"""The IMPES simulator (Chapter 5, sections 5.2 - 5.8).

IMPES = IMplicit Pressure, Explicit Saturation.  Every time step does this:

 1. Look up fluid properties at the current pressure and saturation.
 2. Build the transmissibilities (upstream mobility x geometry), the storage
    coefficients and the gravity and capillary terms, all from the *old* state.
    The upstream block of each phase is chosen by its *potential* (pressure minus
    gravity head), see ``Simulator._neighbour_is_upstream``.
 3. Combine the oil and water equations into one pressure equation for the new
    oil pressures (eq. 5-58), add the well terms, and solve it.
 4. With the new pressures, compute well rates and bottom-hole pressures.
 5. Update the water saturation explicitly from the oil equation (eq. 5-68).

The code keeps the names of the thesis.  A small glossary (all arrays have one
value per cell unless noted):

    lam_o, lam_w     phase mobility  kr / (mu * B)
    T_o[d], T_w[d]   transmissibility to the neighbour in direction d
    cpoo, cpow       storage coefficients for the pressure change (eq. 5-46, 5-52)
    cswo, csww       storage coefficients for the saturation change (eq. 5-47, 5-53)
    alpha            factor that eliminates Sw when the equations are added (5-59)
    b, d             diagonal coefficient and right-hand side of the pressure equation
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

from .constants import BBL_TO_FT3, DARCY, PSI_PER_FT
from .grid import DIRECTIONS
from .model import Equilibrium, ReservoirModel, UniformStart
from .results import Results
from .solver import METHODS, solve_linear_system
from .wells import INJECTOR_MODES, PRODUCER_MODES, Producer, well_index

UPSTREAM_RULES = ("potential", "pressure")


def _share(weights):
    """Fractions that sum to 1, in proportion to ``weights`` (an equal split if all are zero)."""
    total = weights.sum()
    return weights / total if total > 0 else np.full(len(weights), 1.0 / len(weights))


def _safe_ratio(a, b):
    """a / b where b > 0, and 0 elsewhere."""
    return np.divide(a, b, out=np.zeros(np.shape(a)), where=np.asarray(b) > 0)


# Which block-size array and which permeability array belong to each direction.
_SIZE_AND_PERM = {
    "x": ("dx", "kx"),
    "y": ("dy", "ky"),
    "z": ("dz", "kz"),
}


@dataclass
class _WellState:
    """A well while the simulation runs (control mode can change)."""

    name: str
    well: object
    indices: np.ndarray     # cell index of every perforation
    pids: np.ndarray        # well index of every perforation
    heads: np.ndarray       # wellbore pressure increase from the top perforation down to each one (psi)
    mode: str
    target: float


SATURATION_CLIP_TOLERANCE = 0.01          # a saturation moved back by more than this (of the range 0 to 1) is a problem


def diagnostics_healthy(d: dict) -> bool:
    """True when ``Simulator.diagnostics()`` shows no sign of trouble: pressure stayed inside the PVT table, no stop
    condition ended the run, no well flowed backwards, and no saturation had to be moved back by more than
    ``SATURATION_CLIP_TOLERANCE``. (Moves of 1e-4 happen at connate saturation where the oil expands near a producer; an
    unstable time step moves saturations by tenths: ``max_saturation_clip`` tells which.)"""
    return bool(not d["outside_pvt_range"] and d["stop_reason"] is None and d["n_backflow_well_steps"] == 0
                and d["max_saturation_clip"] <= SATURATION_CLIP_TOLERANCE)


@dataclass
class SimState:
    """Everything needed to continue a run from here: time, pressures, saturations, well controls and volumes.

    Get one with ``Simulator.get_state()`` and continue from it with ``Simulator.set_state(state)``, also in a
    simulator built from a *different* model (for example one with updated permeabilities: that is how an
    ensemble member is restarted after a data-assimilation step).
    """

    time: float
    p_o: np.ndarray
    s_w: np.ndarray
    well_controls: dict       # well name -> (mode, target)
    cum: dict                 # well name -> cumulative volumes (STB)

    def copy(self):
        return copy.deepcopy(self)

    def to_dict(self) -> dict:
        """The state as a JSON-ready dictionary (see ``kusim.spec.state_to_dict``)."""
        from .spec import state_to_dict

        return state_to_dict(self)

    @classmethod
    def from_dict(cls, data) -> "SimState":
        from .spec import state_from_dict

        return state_from_dict(data)

    def to_json(self, path):
        from .spec import save_state

        save_state(self, path)

    @classmethod
    def from_json(cls, path) -> "SimState":
        from .spec import load_state

        return load_state(path)


class Simulator:
    """Run the model.

    >>> sim = Simulator(model)
    >>> results = sim.run()

    or take the steps yourself with ``sim.step()`` and look at ``sim.p_o`` and
    ``sim.s_w`` in between.

    Parameters
    ----------
    model : ReservoirModel
    linear_solver : {"sparse", "gauss"}
        How the pressure equation is solved (see ``kusim.solver``).
    save_every : int
        Keep a pressure/saturation snapshot every this many steps (the first and
        last are always kept).
    upstream : {"potential", "pressure"}
        How the upstream block of a phase is chosen (see ``_neighbour_is_upstream``).
        ``"potential"`` (default) compares pressure minus gravity head, which is right
        with gravity.  ``"pressure"`` compares pressure only, as in thesis eq. 5-28 to 5-33
        and in the original FORTRAN; it suppresses gravity segregation in vertical flow
        (porting-notes decision 18) and is kept to reproduce the legacy behaviour.
    """

    def __init__(self, model: ReservoirModel, linear_solver: str = "sparse", save_every: int = 1,
                 upstream: str = "potential"):
        if linear_solver not in METHODS:
            raise ValueError(f"linear_solver must be one of {METHODS}")
        if upstream not in UPSTREAM_RULES:
            raise ValueError(f"upstream must be one of {UPSTREAM_RULES}")
        self.upstream = upstream
        self.model = model
        self.grid = model.grid
        self.linear_solver = linear_solver
        self.save_every = max(1, int(save_every))
        self._build_geometric_factors()
        self.reset()

    # ------------------------------------------------------------------
    # Set-up
    # ------------------------------------------------------------------
    def _build_geometric_factors(self):
        """Geometry part of each transmissibility, which never changes.

        For the neighbour in direction d (eq. 5-16 to 5-27):

            T = 0.006328 * 2/(dL) * 1/(dL_nb/k_nb + dL/k) * lambda_upstream

        The part before ``lambda_upstream`` is stored here as ``geometry[d]``.
        It is zero on the edges of the grid, which makes them no-flow boundaries.
        """
        g, m = self.grid, self.model
        self.geometry = {}
        for direction in DIRECTIONS:
            size_name, perm_name = _SIZE_AND_PERM[direction[0]]
            size, perm = getattr(g, size_name), getattr(m, perm_name)
            nb = g.neighbour[direction]
            harmonic = 1.0 / (size[nb] / perm[nb] + size / perm)
            self.geometry[direction] = DARCY * (2.0 / size) * harmonic * g.is_open[direction]

    def _setup_wells(self):
        g, m = self.grid, self.model
        self.producers, self.injectors = [], []
        for n, w in enumerate(m.producers, start=1):
            self.producers.append(self._make_well_state(f"P{n}", w))
        for n, w in enumerate(m.injectors, start=1):
            self.injectors.append(self._make_well_state(f"I{n}", w))

    def _make_well_state(self, name, well):
        g, m = self.grid, self.model
        indices = np.array([g.cell_index(b) for b in well.blocks])
        pids = np.array([well_index(m.kx[c], m.ky[c], g.dx[c], g.dy[c], g.dz[c], well.radius, well.skin)
                         for c in indices])
        gravity = well.wellbore_gravity
        if gravity is None:
            gravity = m.gamma_o if isinstance(well, Producer) else m.gamma_w
        depths = g.depth[indices]
        heads = PSI_PER_FT * gravity * (depths - depths.min())
        return _WellState(name, well, indices, pids, heads, well.mode, well.target)

    def reset(self):
        """Go back to time zero and set the initial pressure and saturation."""
        g, m = self.grid, self.model
        init = m.initial
        if isinstance(init, Equilibrium):
            # Capillary pressure at each block's height above the water-oil contact,
            # then the water saturation that has that capillary pressure (5-94 .. 5-97).
            pcow = PSI_PER_FT * (m.gamma_w - m.gamma_o) * (init.woc_depth - g.depth)
            self.s_w = m.rock_fluid.sw_from_pcow(pcow)
            self.p_o = init.datum_pressure + PSI_PER_FT * m.gamma_o * g.depth
        elif isinstance(init, UniformStart):
            self.s_w = np.full(g.n_cells, float(init.sw))
            self.p_o = np.full(g.n_cells, float(init.pressure))
        else:
            raise TypeError("model.initial must be Equilibrium or UniformStart")
        self.time = 0.0
        self.step_number = 0
        self.stop_reason = None
        self._setup_wells()
        self._cum = {w.name: {"oil": 0.0, "water": 0.0} for w in self.producers}
        self._cum.update({w.name: {"water": 0.0} for w in self.injectors})
        self._well_rows = []
        self._perf_rows = []
        self._times = [0.0]
        self._pressure = [self.p_o.copy()]
        self._sw = [self.s_w.copy()]
        self._max_clip = 0.0                      # the largest amount by which Sw had to be moved back (saturation units)
        self._n_clipped = 0                       # cell-steps where Sw had to be moved back inside [swc, 1 - sor] (by more than 1e-6)
        self._n_backflow = 0                      # well-steps where a producer injected or an injector produced
        self._step_wells = {}
        self._p_min, self._p_max = float(self.p_o.min()), float(self.p_o.max())

    @property
    def finished(self) -> bool:
        """True when the end time is reached or a stop condition was met."""
        return self.stop_reason is not None or self.step_number >= self.model.n_steps

    # ------------------------------------------------------------------
    # Fluid properties
    # ------------------------------------------------------------------
    def _fluid_properties(self, p_o, s_w):
        """Properties at the current state, including mobilities (Chapter 5, 5.8 steps 5-6)."""
        m = self.model
        rf, pvt = m.rock_fluid, m.pvt
        pr = SimpleNamespace()
        pr.pcow = rf.pcow_at(s_w)
        pr.dpcow = rf.d_pcow(s_w)
        pr.p_w = p_o - pr.pcow                      # eq. 5-8:  Pw = Po - Pcow
        pr.inv_bo = pvt.inv_bo(p_o)
        pr.d_inv_bo = pvt.d_inv_bo(p_o)
        pr.inv_bw = pvt.inv_bw(pr.p_w)
        pr.d_inv_bw = pvt.d_inv_bw(pr.p_w)
        pr.lam_o = rf.relperm_oil(s_w) * pr.inv_bo / pvt.viscosity_oil(p_o)
        pr.lam_w = rf.relperm_water(s_w) * pr.inv_bw / pvt.viscosity_water(pr.p_w)
        return pr

    # ------------------------------------------------------------------
    # One time step
    # ------------------------------------------------------------------
    def step(self):
        """Advance the simulation by one time step ``dt``."""
        if self.finished:
            raise RuntimeError("the simulation has already finished")
        g, m = self.grid, self.model
        dt = m.dt
        p_old, s_old = self.p_o, self.s_w
        pr = self._fluid_properties(p_old, s_old)

        # gravity head (psi) from each block to its neighbour, per unit specific gravity
        head = {d: PSI_PER_FT * (g.depth[g.neighbour[d]] - g.depth) for d in DIRECTIONS}

        # --- transmissibilities with upstream mobility (eq. 5-28 .. 5-33, by potential: decision 18) ---
        t_o, t_w = {}, {}
        for d in DIRECTIONS:
            nb = g.neighbour[d]
            mob_o = np.where(self._neighbour_is_upstream(p_old, m.gamma_o, head[d], nb), pr.lam_o[nb], pr.lam_o)
            mob_w = np.where(self._neighbour_is_upstream(pr.p_w, m.gamma_w, head[d], nb), pr.lam_w[nb], pr.lam_w)
            t_o[d] = self.geometry[d] * mob_o
            t_w[d] = self.geometry[d] * mob_w

        # --- storage coefficients (eq. 5-46, 5-47, 5-52, 5-53) ---
        # (The code stores 1/B; the thesis writes the same terms with B.)
        phi, cr = m.porosity, m.rock_compressibility
        cpoo = phi * (1.0 - s_old) / dt * (cr * pr.inv_bo + pr.d_inv_bo)
        cswo = -phi * pr.inv_bo / dt
        cpow = s_old * phi / dt * (cr * pr.inv_bw + pr.d_inv_bw)
        csww = phi * pr.inv_bw / dt - cpow * pr.dpcow
        alpha = -cswo / csww                                          # eq. 5-59

        # --- gravity and capillary terms moved to the right-hand side ---
        gra_o = np.zeros(g.n_cells)
        gra_w = np.zeros(g.n_cells)
        cap = np.zeros(g.n_cells)
        sum_t_o = np.zeros(g.n_cells)
        sum_t_w = np.zeros(g.n_cells)
        for d in DIRECTIONS:
            nb = g.neighbour[d]
            gra_o += m.gamma_o * head[d] * t_o[d]
            gra_w += m.gamma_w * head[d] * t_w[d]
            cap += t_w[d] * (pr.pcow[nb] - pr.pcow)
            sum_t_o += t_o[d]
            sum_t_w += t_w[d]
        cap *= alpha

        # --- the pressure equation  b*P_i + sum(coef_d * P_nb) = d_i  (eq. 5-60) ---
        coeff = {dd: t_o[dd] + alpha * t_w[dd] for dd in DIRECTIONS}  # eq. 5-61
        b = -((sum_t_o + cpoo) + alpha * (sum_t_w + cpow))
        rhs = cap + (gra_o + alpha * gra_w) - (cpoo + alpha * cpow) * p_old

        # --- wells (eq. 5-62 .. 5-91) ---
        self._add_well_terms(b, rhs, pr, alpha)

        # --- assemble the 7-diagonal matrix and solve ---
        n = g.n_cells
        rows, cols, vals = [np.arange(n)], [np.arange(n)], [b]
        for d in DIRECTIONS:
            r = np.flatnonzero(g.is_open[d])
            rows.append(r)
            cols.append(g.neighbour[d][r])
            vals.append(coeff[d][r])
        matrix = coo_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)
        )
        p_new = solve_linear_system(matrix, rhs, self.linear_solver)

        # --- well rates and bottom-hole pressures from the new pressures ---
        qop_cell = self._finish_wells(p_new, pr)
        for name, data in self._step_wells.items():
            total = data["q_w"].sum() + (data["q_o"].sum() if data["kind"] == "producer" else 0.0)
            self._n_backflow += int(total < 0.0)

        # --- explicit saturation update (eq. 5-68) ---
        flow = np.zeros(n)
        for d in DIRECTIONS:
            flow += t_o[d] * (p_new[g.neighbour[d]] - p_new)
        s_new = s_old + (flow - qop_cell - cpoo * (p_new - p_old) - gra_o) / cswo
        s_unclipped = s_new
        s_new = np.clip(s_new, m.swc, 1.0 - m.sor)
        moved = np.abs(s_new - s_unclipped)
        self._n_clipped += int(np.count_nonzero(moved > 1e-6))
        self._max_clip = max(self._max_clip, float(moved.max()))

        # --- advance ---
        self.p_o, self.s_w = p_new, s_new
        self._p_min, self._p_max = min(self._p_min, float(p_new.min())), max(self._p_max, float(p_new.max()))
        self.time += dt
        self.step_number += 1
        self._record_wells(dt)
        self._apply_well_limits()
        if self.step_number % self.save_every == 0 or self.finished:
            self._times.append(self.time)
            self._pressure.append(self.p_o.copy())
            self._sw.append(self.s_w.copy())

    def _neighbour_is_upstream(self, p_phase, gamma, head, nb):
        """True where the neighbour ``nb`` is the upstream block for one phase, block by block.

        Fluid flows towards the block with the lower *potential* ``p - 0.433 gamma z``.  The flow
        into a block from its neighbour is proportional to

            (p_nb - p) - gamma * head,        head = 0.433 * (z_nb - z)

        so the neighbour is upstream (the mobility is taken from it) when that is >= 0.
        ``p_phase`` is the pressure of the phase (oil or water), ``gamma`` its specific gravity,
        ``head`` the array of 0.433 * (z_nb - z) and ``nb`` the neighbour index array.

        With ``upstream="pressure"`` the test is ``p_nb >= p`` only (thesis eq. 5-28 to 5-33): the same
        when the depths are equal, but wrong in vertical flow, where it takes the lower block's
        (often nearly zero) water mobility for water that is sinking.
        """
        if self.upstream == "pressure":
            return p_phase[nb] >= p_phase
        return (p_phase[nb] - p_phase) - gamma * head >= 0

    # ------------------------------------------------------------------
    # State, controls and diagnostics (history matching, ensembles)
    # ------------------------------------------------------------------
    def get_state(self) -> SimState:
        """A snapshot of the current state (a copy: later steps do not change it)."""
        wells = [*self.producers, *self.injectors]
        return SimState(self.time, self.p_o.copy(), self.s_w.copy(),
                        {w.name: (w.mode, w.target) for w in wells}, copy.deepcopy(self._cum))

    def set_state(self, state: SimState, clip: bool = True) -> dict:
        """Continue from ``state`` (see ``SimState``).  Returns what had to be adjusted.

        Saturations outside ``[swc, 1 - sor]`` (a data-assimilation update can produce them) are put back
        inside when ``clip`` is true, and counted in ``"n_saturation_clipped"``; with ``clip=False`` they raise.
        Pressures are taken as given; ``diagnostics()`` tells whether they leave the PVT table.
        The time must be a multiple of the model's ``dt``.  The saved fields of the run start again from here.
        """
        m, n = self.model, self.grid.n_cells
        p, s = np.asarray(state.p_o, dtype=float), np.asarray(state.s_w, dtype=float)
        if p.shape != (n,) or s.shape != (n,):
            raise ValueError(f"the state needs {n} pressures and {n} saturations")
        if not (np.all(np.isfinite(p)) and np.all(np.isfinite(s))):
            raise ValueError("the state contains NaN or infinite values")
        lo, hi = m.swc, 1.0 - m.sor
        n_out = int(np.count_nonzero((s < lo) | (s > hi)))
        if n_out and not clip:
            raise ValueError(f"{n_out} saturations are outside [{lo}, {hi}]")
        steps = state.time / m.dt
        if abs(steps - round(steps)) > 1e-6:
            raise ValueError(f"the state time {state.time} is not a multiple of dt = {m.dt}")
        known = {w.name for w in [*self.producers, *self.injectors]}
        if set(state.well_controls) != known:
            raise ValueError(f"the state has wells {sorted(state.well_controls)}, the model has {sorted(known)}")
        self.p_o, self.s_w = p.copy(), np.clip(s, lo, hi)
        self.time, self.step_number, self.stop_reason = float(state.time), int(round(steps)), None
        for name, (mode, target) in state.well_controls.items():
            self.set_well_control(name, mode, target)
        self._cum = copy.deepcopy(state.cum)
        self._well_rows, self._perf_rows = [], []
        self._times, self._pressure, self._sw = [self.time], [self.p_o.copy()], [self.s_w.copy()]
        self._n_clipped = 0
        self._max_clip = 0.0
        self._n_backflow = 0
        self._step_wells = {}
        self._p_min, self._p_max = float(self.p_o.min()), float(self.p_o.max())
        return {"n_saturation_clipped": n_out}

    def set_well_control(self, name, mode=None, target=None):
        """Change the control of one well from the next step on (``mode`` and/or ``target``).

        Modes are those of ``Producer`` ("oil_rate", "liquid_rate", "bhp") and ``Injector``
        ("water_rate", "bhp"); rates are STB/day and positive, a bottom-hole pressure is in psi.
        A producer whose ``min_bhp`` was reached has switched to "bhp"; setting a rate mode again undoes that.
        """
        wells = {w.name: w for w in [*self.producers, *self.injectors]}
        if name not in wells:
            raise KeyError(f"no well {name!r}; the wells are {sorted(wells)}")
        w = wells[name]
        allowed = PRODUCER_MODES if isinstance(w.well, Producer) else INJECTOR_MODES
        if mode is not None:
            if mode not in allowed:
                raise ValueError(f"well {name}: mode must be one of {allowed}, got {mode!r}")
            w.mode = mode
        if target is not None:
            if not np.isfinite(target) or target < 0:
                raise ValueError(f"well {name}: the target must be a number >= 0, got {target!r}")
            w.target = float(target)

    def well_data(self) -> dict:
        """The wells in the last step: ``{name: {kind, mode, target, bhp, oil_rate, water_rate}}`` (rates STB/day).

        Empty before the first step."""
        wells = {w.name: w for w in [*self.producers, *self.injectors]}
        return {
            name: dict(kind=d["kind"], mode=wells[name].mode, target=wells[name].target, bhp=float(d["bhp"]),
                       oil_rate=float(d["q_o"].sum()), water_rate=float(d["q_w"].sum()))
            for name, d in self._step_wells.items()
        }

    def well_block_throughput(self) -> float:
        """The largest fraction of the *mobile* pore volume of a well block that the wells moved in the last step.

        For every perforation: reservoir volume produced or injected in the step (rates times formation volume
        factors, times ``dt``) divided by ``porosity * volume * (1 - swc - sor)`` of its block. The explicit saturation
        update is stable only if this stays modest: with the two cases it was calibrated on (the thesis case and a
        layered waterflood) saturations began to be clipped from about 3.5. A heuristic, not a bound: it uses the rates
        of the last step, and fronts moving between blocks are not included. NaN before the first step."""
        if not self._step_wells:
            return float("nan")
        g, m, worst = self.grid, self.model, 0.0
        for w in [*self.producers, *self.injectors]:
            s = self._step_wells[w.name]
            idx = w.indices
            reservoir = np.abs(s["q_o"]) / m.pvt.inv_bo(s["pressure"]) + np.abs(s["q_w"]) / m.pvt.inv_bw(s["pressure"])
            mobile = g.volume[idx] * m.porosity[idx] * (1.0 - m.swc - m.sor)
            worst = max(worst, float(np.max(reservoir * BBL_TO_FT3 * m.dt / mobile)))
        return worst

    def diagnostics(self) -> dict:
        """Signs that a run is not to be trusted, as numbers and flags (for loops that run many models).

        ``n_saturation_clipped``: cell-steps where Sw had to be moved back inside ``[swc, 1 - sor]`` by more than 1e-6;
        ``max_saturation_clip``: the largest such move (1e-4 is harmless, tenths mean an unstable time step).
        ``outside_pvt_range``: the pressure left the PVT table at some time (both lookups are flat there and
        the compressibility is not consistent: see ``docs/porting-notes.md``, open items).
        ``n_backflow_well_steps``: well-steps where a producer injected or an injector produced (a bottom-hole pressure
        beyond the pressure of the block).  A commercial simulator would shut such a well in; here the well just flows
        backwards, which an optimiser can exploit.
        """
        lo, hi = float(self.model.pvt.pressure[0]), float(self.model.pvt.pressure[-1])
        return dict(time=self.time, n_saturation_clipped=self._n_clipped, max_saturation_clip=self._max_clip, n_backflow_well_steps=self._n_backflow,
                    p_min=self._p_min, p_max=self._p_max,
                    pvt_range=(lo, hi), outside_pvt_range=bool(self._p_min < lo or self._p_max > hi),
                    stop_reason=self.stop_reason)

    # ------------------------------------------------------------------
    # Wells
    # ------------------------------------------------------------------
    def _add_well_terms(self, b, rhs, pr, alpha):
        """Add the well terms to the pressure equation.  Rates are STB/day.

        A cell's well term is  q' = Q / (cell volume): a rate per unit volume.
        The 5.615 ft^3/bbl factor puts it in the same units as the flow and
        storage terms (which use 0.006328 = 0.001127 * 5.615).

        A well may have several perforations (one per completion block).

        * Rate-controlled wells: the rate is shared between the perforations in
          proportion to their flow capacity (well index x mobility, evaluated at the
          start of the step).  Each perforation then adds a known rate to the
          right-hand side only.
        * Bottom-hole pressure wells: every perforation flows against the same
          pressure.  The block pressure is an unknown, so the diagonal ``b`` changes
          as well (eq. 5-65, 5-66).

        Note: the original FORTRAN divides the rate by the cell volume but omits
        the 5.615 factor, which makes the well terms 5.615 times too small
        compared with the flow terms.  KUSIM-Python applies the factor.
        """
        g = self.grid
        self._plan = {}        # rate-controlled wells: per-perforation (oil, water) rates

        for w in self.producers:
            idx = w.indices
            per_volume = BBL_TO_FT3 / g.volume[idx]
            lam_o, lam_w = pr.lam_o[idx], pr.lam_w[idx]
            if w.mode == "bhp":
                coef = w.pids * per_volume * (lam_o + alpha[idx] * lam_w)     # eq. 5-90, 5-91
                b[idx] -= coef
                rhs[idx] -= coef * (w.target + w.heads)
                continue
            if w.mode == "oil_rate":
                q_o = w.target * _share(w.pids * lam_o)
                q_w = q_o * _safe_ratio(lam_w, lam_o)                         # eq. 5-79
            else:                                                            # "liquid_rate"
                lam_t = lam_o + lam_w
                q_l = w.target * _share(w.pids * lam_t)
                q_o = q_l * _safe_ratio(lam_o, lam_t)                         # eq. 5-83
                q_w = q_l * _safe_ratio(lam_w, lam_t)                         # eq. 5-84
            rhs[idx] += (q_o + alpha[idx] * q_w) * per_volume
            self._plan[w.name] = (q_o, q_w)

        for w in self.injectors:
            idx = w.indices
            per_volume = BBL_TO_FT3 / g.volume[idx]
            mobility = self._injectivity_mobility(pr, idx)
            if w.mode == "water_rate":
                q_w = w.target * _share(w.pids * mobility)                    # eq. 5-70
                rhs[idx] -= alpha[idx] * q_w * per_volume
                self._plan[w.name] = (None, q_w)
            else:                                                            # "bhp"
                coef = w.pids * per_volume * mobility
                b[idx] -= alpha[idx] * coef
                rhs[idx] -= alpha[idx] * coef * (w.target + w.heads)

    @staticmethod
    def _injectivity_mobility(pr, c):
        """Mobility used in the injector well equation (eq. 5-74): (Bo/Bw)*lam_o + lam_w.

        ``c`` is a cell index or an array of them.  The oil is displaced by the injected
        water, so the oil mobility counts too, converted to water-barrel units with
        Bo/Bw.  In terms of the reciprocals used in the code, Bo/Bw = (1/Bw)/(1/Bo).

        Note: the FORTRAN uses BO/BW with BO, BW already inverted, which is the
        ratio upside down; this uses the form in the thesis.
        """
        return pr.lam_o[c] * pr.inv_bw[c] / pr.inv_bo[c] + pr.lam_w[c]

    def _finish_wells(self, p_new, pr):
        """After the solve: perforation rates, bottom-hole pressures, and q'_o for the saturation update.

        For a rate-controlled well the bottom-hole pressure follows from the well
        equation summed over the perforations:  Q = sum(PI_k * lam_k * (P_k - Pwf - head_k)),
        so  Pwf = (sum(PI_k * lam_k * (P_k - head_k)) - Q) / sum(PI_k * lam_k).
        ``head_k`` is the wellbore pressure increase between the top perforation and perforation k.
        """
        g = self.grid
        qop_cell = np.zeros(g.n_cells)
        self._step_wells = {}

        for w in self.producers:
            idx, p = w.indices, p_new[w.indices]
            lam_o, lam_w = pr.lam_o[idx], pr.lam_w[idx]
            if w.mode == "bhp":
                bhp = w.target
                q_o = w.pids * lam_o * (p - bhp - w.heads)
                q_w = w.pids * lam_w * (p - bhp - w.heads)
            else:
                q_o, q_w = self._plan[w.name]
                capacity = float(np.sum(w.pids * lam_o))
                bhp = (np.sum(w.pids * lam_o * (p - w.heads)) - q_o.sum()) / capacity if capacity > 0 else np.nan
            qop_cell[idx] = q_o * BBL_TO_FT3 / g.volume[idx]
            self._step_wells[w.name] = dict(kind="producer", bhp=bhp, q_o=q_o, q_w=q_w, pressure=p)

        for w in self.injectors:
            idx, p = w.indices, p_new[w.indices]
            mobility = self._injectivity_mobility(pr, idx)
            if w.mode == "bhp":
                bhp = w.target
                q_w = w.pids * mobility * (bhp + w.heads - p)
            else:
                q_w = self._plan[w.name][1]
                bhp = (q_w.sum() + np.sum(w.pids * mobility * (p - w.heads))) / np.sum(w.pids * mobility)
            self._step_wells[w.name] = dict(kind="injector", bhp=bhp, q_o=np.zeros(len(idx)), q_w=q_w, pressure=p)
        return qop_cell

    def _record_wells(self, dt):
        """Store the well totals and the rate of every perforation for this step."""
        g = self.grid
        for w in [*self.producers, *self.injectors]:
            s = self._step_wells[w.name]
            cum = self._cum[w.name]
            produced = s["kind"] == "producer"
            oil, water = float(s["q_o"].sum()), float(s["q_w"].sum())
            if produced:
                cum["oil"] += oil * dt
            cum["water"] += water * dt
            liquid = oil + water
            self._well_rows.append(
                dict(
                    time=self.time,
                    well=w.name,
                    kind=s["kind"],
                    block=w.well.blocks[0],
                    n_perforations=len(w.indices),
                    mode=w.mode,
                    bhp=s["bhp"],
                    oil_rate=oil,
                    water_rate=water,
                    liquid_rate=liquid,
                    water_cut=(water / liquid) if produced and liquid > 0 else np.nan,
                    cum_oil=cum.get("oil", np.nan),
                    cum_water=cum["water"],
                )
            )
            for n, c in enumerate(w.indices):
                self._perf_rows.append(
                    dict(time=self.time, well=w.name, kind=s["kind"], block=int(c) + 1,
                         i=int(g.i[c]) + 1, j=int(g.j[c]) + 1, k=int(g.k[c]) + 1,
                         oil_rate=float(s["q_o"][n]), water_rate=float(s["q_w"][n]),
                         pressure=float(s["pressure"][n]))
                )

    def _apply_well_limits(self):
        """Producer limits, checked at the end of each step (as in the original program).

        * Bottom-hole pressure at or below ``min_bhp``: the well switches to
          constant bottom-hole pressure at ``min_bhp``.
        * Oil rate at or below ``min_oil_rate``: the run stops.
        """
        for w in self.producers:
            s = self._step_wells[w.name]
            lim = w.well
            if w.mode != "bhp" and lim.min_bhp is not None and s["bhp"] <= lim.min_bhp:
                w.mode, w.target = "bhp", lim.min_bhp
            oil = float(s["q_o"].sum())
            if lim.min_oil_rate is not None and oil <= lim.min_oil_rate:
                self.stop_reason = f"{w.name}: oil rate fell to {lim.min_oil_rate} STB/day at t = {self.time:g} days"

    # ------------------------------------------------------------------
    def run(self) -> Results:
        """Run to the end time (or until a stop condition) and return the results."""
        while not self.finished:
            self.step()
        return self.results()

    def results(self) -> Results:
        """Package what has been simulated so far."""
        wells = pd.DataFrame(self._well_rows)
        return Results(
            grid=self.grid,
            times=np.array(self._times),
            pressure=np.array(self._pressure),
            sw=np.array(self._sw),
            wells=wells,
            perforations=pd.DataFrame(self._perf_rows),
            completions={w.name: [int(c) + 1 for c in w.indices] for w in [*self.producers, *self.injectors]},
            stop_reason=self.stop_reason,
        )
