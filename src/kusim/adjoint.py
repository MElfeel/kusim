"""Adjoint gradients of the IMPES simulator: the gradient of a history-matching (or production) objective with respect to
permeability, porosity and well controls, for the cost of about two to three simulations.

What this is.  ``forward_step`` is a *separate, functional copy* of ``Simulator.step`` (the same arithmetic, no state, no
bookkeeping) and ``backward_step`` is its transpose, derived by hand: it runs the forward operations in reverse order, each
with its local derivative, so the code reads like the forward step backwards.  Tests check that the copy reproduces the
simulator and that the gradients agree with finite differences.

Why it is simple here.  One IMPES step is  x^{n+1} = (p^{n+1}, S^{n+1})  with

    A(x^n) p^{n+1} = d(x^n, u^n)                (the pressure equation: the matrix and the right-hand side use the OLD state)
    S^{n+1} = clip( S^n + F(p^{n+1}, x^n) )     (explicit saturation update)

so the adjoint (Intersect's eq. 2.1046-2.1047 for this scheme) needs no Newton Jacobian: going backwards, every step costs
one *transposed sparse solve* with the pressure matrix, plus products with the derivatives of the coefficients, which
come from the slopes of the property tables.  The states are stored, and the intermediate quantities of a step are
recomputed during the backward sweep ("reconstructed from saved simulator state").

What is differentiated, and what is not.  The gradient is that of the algorithm as implemented.  Discrete decisions are
frozen: the choice of the upstream block, the saturation clip (a clipped block passes no gradient), the table segment,
a producer switching to its ``min_bhp``, and a stop at ``min_oil_rate``.  Where such a decision changes within a small
change of the parameters the objective has a kink, and the gradient is only one of its one-sided slopes; the counts in
``AdjointResult.diagnostics`` say how often decisions flipped (``backflow_well_steps`` counts well-steps in which a producer
injected or an injector produced: an optimiser can exploit that, which no real well does).  Outside the PVT table the lookups are flat, and so are
their derivatives (the storage slope used by the scheme is differentiated as a constant).  The initial state is not a
function of the parameters here.  Not covered: scalar parameters such as skin or rock compressibility (use finite
differences), and objective vectors other than the well vectors, ``FPRP`` and the cumulative ``FOPT``, ``FWPT``, ``FWIT``.

Usage (most users want ``kusim.gradients.HistoryProblem``, which also handles the parameter vector):

    result = adjoint_gradient(model, observations, schedule)
    result.value                 # J
    result.fields["kx"]          # dJ/d kx, one value per block (also "ky", "kz", "porosity")
    result.controls["I1"]        # dJ/d(target of well I1) for every step
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from .constants import BBL_TO_FT3, DARCY, PSI_PER_FT
from .grid import DIRECTIONS
from .schedule import plan_advance
from .simulator import Simulator
from .wells import well_index

SUPPORTED_VECTORS = ("WBHP", "WOPR", "WWPR", "WWCT", "WWIR", "FPRP", "FOPT", "FWPT", "FWIT")


# ====================================================================== small helpers
def _lookup(table, x):
    """Table value and derivative.  The derivative is the slope of the segment inside the table and 0 where the lookup
    is flat (outside), which is what the value does; ``Table.slope`` itself keeps the end slope there."""
    x = np.asarray(x, dtype=float)
    seg = np.clip(np.searchsorted(table.x, x, side="right") - 1, 0, len(table.x) - 2)
    slope = (table.y[seg + 1] - table.y[seg]) / (table.x[seg + 1] - table.x[seg])
    inside = (x >= table.x[0]) & (x <= table.x[-1])
    return table.lookup(x), np.where(inside, slope, 0.0)


def _share(weights):
    """Fractions in proportion to ``weights`` (an equal split if they are all zero), and their total."""
    total = weights.sum()
    return (weights / total if total > 0 else np.full(len(weights), 1.0 / len(weights))), total


def _share_backward(share_bar, share, total):
    """Gradient with respect to the weights, from the gradient with respect to the shares."""
    if total > 0:
        return (share_bar - np.sum(share_bar * share)) / total
    return np.zeros_like(share_bar)


def _safe_ratio(a, b):
    return np.divide(a, b, out=np.zeros(np.shape(a)), where=np.asarray(b) > 0)


def _upstream(rule, p_phase, gamma, head, nb):
    """The upstream test of ``Simulator._neighbour_is_upstream`` (potential or pressure)."""
    if rule == "pressure":
        return p_phase[nb] >= p_phase
    return (p_phase[nb] - p_phase) - gamma * head >= 0


def _well_index_derivatives(kx, ky, dx, dy, dz, radius, skin):
    """Well index and its derivatives with respect to kx and ky (Peaceman, thesis eq. 5-72, 5-73)."""
    pid = well_index(kx, ky, dx, dy, dz, radius, skin)
    r = ky / kx
    num = np.sqrt(r) * dx**2 + dy**2 / np.sqrt(r)
    bottom = r**0.25 + r**-0.25
    dln_re_dr = 0.5 * (0.5 * r**-0.5 * dx**2 - 0.5 * r**-1.5 * dy**2) / num - (0.25 * r**-0.75 - 0.25 * r**-1.25) / bottom
    re = 0.28 * np.sqrt(num) / bottom
    length = np.log(re / radius) + skin
    return pid, pid * (0.5 / kx + dln_re_dr * r / (kx * length)), pid * (0.5 / ky - dln_re_dr / (kx * length))


# ====================================================================== constants of a model
@dataclass
class _WellConst:
    name: str
    kind: str                  # "producer" or "injector"
    well: object
    idx: np.ndarray            # cell index of each perforation
    pid: np.ndarray            # well index of each perforation
    heads: np.ndarray          # wellbore pressure head of each perforation
    per_vol: np.ndarray        # 5.615 / block volume
    vol: np.ndarray


class _Const:
    """Everything of a model that does not change during a run, taken from a ``Simulator`` built from it (so that the
    geometry, the well indices and the heads are the simulator's own)."""

    def __init__(self, sim):
        g, m = sim.grid, sim.model
        self.model, self.grid, self.n, self.dt, self.upstream = m, g, g.n_cells, m.dt, sim.upstream
        self.G = {d: sim.geometry[d] for d in DIRECTIONS}
        self.head = {d: PSI_PER_FT * (g.depth[g.neighbour[d]] - g.depth) for d in DIRECTIONS}
        self.pvt_tables = SimpleNamespace(ibo=m.pvt._inv_bo, ibw=m.pvt._inv_bw, muo=m.pvt._mu_o, muw=m.pvt._mu_w)
        self.rf_tables = SimpleNamespace(kro=m.rock_fluid._kro, krw=m.rock_fluid._krw, pc=m.rock_fluid._pcow)
        self.wells = []
        for ws in [*sim.producers, *sim.injectors]:
            kind = "producer" if ws.name.startswith("P") else "injector"
            vol = g.volume[ws.indices]
            self.wells.append(_WellConst(ws.name, kind, ws.well, ws.indices, ws.pids, ws.heads, BBL_TO_FT3 / vol, vol))


# ====================================================================== the forward step (a functional copy)
def forward_step(c, p_old, s_old, controls):
    """One step from ``(p_old, s_old)`` with the controls ``{well: (mode, target)}``.

    Returns ``(p_new, s_new, outputs, k)``: ``outputs[well]`` has the bottom-hole pressure ``bhp``, the perforation rates
    ``q_o``, ``q_w`` and their sums ``qo``, ``qw``; ``k`` holds the intermediate quantities for the backward sweep.
    The arithmetic is that of ``Simulator.step`` (the order of operations too, so that the results agree to rounding)."""
    g, m, dt, n = c.grid, c.model, c.dt, c.n
    T, R = c.pvt_tables, c.rf_tables
    k = SimpleNamespace(p_old=p_old, s_old=s_old, controls=controls)

    # --- fluid properties (value and derivative of every table lookup)
    k.pcow, k.dpcow_ds = _lookup(R.pc, s_old)
    k.dpc = R.pc.slope(s_old)                                          # the slope the scheme uses in csww
    k.pw = p_old - k.pcow
    k.ibo, k.dibo_dp = _lookup(T.ibo, p_old)
    k.dibo = T.ibo.slope(p_old)
    k.ibw, k.dibw_dpw = _lookup(T.ibw, k.pw)
    k.dibw = T.ibw.slope(k.pw)
    k.muo, k.dmuo_dp = _lookup(T.muo, p_old)
    k.muw, k.dmuw_dpw = _lookup(T.muw, k.pw)
    k.kro, k.dkro_ds = _lookup(R.kro, s_old)
    k.krw, k.dkrw_ds = _lookup(R.krw, s_old)
    k.lam_o = k.kro * k.ibo / k.muo
    k.lam_w = k.krw * k.ibw / k.muw

    # --- transmissibilities with the upstream mobility
    k.flag_o, k.flag_w, k.mob_o, k.mob_w, k.t_o, k.t_w = {}, {}, {}, {}, {}, {}
    for d in DIRECTIONS:
        nb = g.neighbour[d]
        k.flag_o[d] = _upstream(c.upstream, p_old, m.gamma_o, c.head[d], nb)
        k.flag_w[d] = _upstream(c.upstream, k.pw, m.gamma_w, c.head[d], nb)
        k.mob_o[d] = np.where(k.flag_o[d], k.lam_o[nb], k.lam_o)
        k.mob_w[d] = np.where(k.flag_w[d], k.lam_w[nb], k.lam_w)
        k.t_o[d] = c.G[d] * k.mob_o[d]
        k.t_w[d] = c.G[d] * k.mob_w[d]

    # --- storage coefficients
    k.phi, k.cr = m.porosity, m.rock_compressibility
    k.cpoo = k.phi * (1.0 - s_old) / dt * (k.cr * k.ibo + k.dibo)
    k.cswo = -k.phi * k.ibo / dt
    k.cpow = s_old * k.phi / dt * (k.cr * k.ibw + k.dibw)
    k.csww = k.phi * k.ibw / dt - k.cpow * k.dpc
    k.alpha = -k.cswo / k.csww

    # --- gravity and capillary terms, the pressure equation
    k.gra_o, k.gra_w, k.cap_raw = np.zeros(n), np.zeros(n), np.zeros(n)
    k.sum_t_o, k.sum_t_w = np.zeros(n), np.zeros(n)
    for d in DIRECTIONS:
        nb = g.neighbour[d]
        k.gra_o += m.gamma_o * c.head[d] * k.t_o[d]
        k.gra_w += m.gamma_w * c.head[d] * k.t_w[d]
        k.cap_raw += k.t_w[d] * (k.pcow[nb] - k.pcow)
        k.sum_t_o += k.t_o[d]
        k.sum_t_w += k.t_w[d]
    cap = k.cap_raw * k.alpha
    k.coeff = {d: k.t_o[d] + k.alpha * k.t_w[d] for d in DIRECTIONS}
    b = -((k.sum_t_o + k.cpoo) + k.alpha * (k.sum_t_w + k.cpow))
    rhs = cap + (k.gra_o + k.alpha * k.gra_w) - (k.cpoo + k.alpha * k.cpow) * p_old

    # --- wells in the matrix, then solve
    k.well_store = {wc.name: _well_matrix(c, wc, controls[wc.name], k, b, rhs) for wc in c.wells}
    rows, cols, vals = [np.arange(n)], [np.arange(n)], [b]
    for d in DIRECTIONS:
        r = np.flatnonzero(g.is_open[d])
        rows.append(r)
        cols.append(g.neighbour[d][r])
        vals.append(k.coeff[d][r])
    matrix = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    k.A = matrix.tocsc()
    p_new = spsolve(k.A, rhs)

    # --- well rates and bottom-hole pressures
    qop_cell = np.zeros(n)
    outputs = {}
    for wc in c.wells:
        outputs[wc.name] = _well_finish(c, wc, controls[wc.name], k.well_store[wc.name], p_new, qop_cell)

    # --- explicit saturation update
    k.flow = np.zeros(n)
    for d in DIRECTIONS:
        k.flow += k.t_o[d] * (p_new[g.neighbour[d]] - p_new)
    k.u = k.flow - qop_cell - k.cpoo * (p_new - p_old) - k.gra_o
    s_unclipped = s_old + k.u / k.cswo
    s_new = np.clip(s_unclipped, m.swc, 1.0 - m.sor)
    k.clip_mask = ((s_unclipped >= m.swc) & (s_unclipped <= 1.0 - m.sor)).astype(float)
    k.clipped = np.abs(s_new - s_unclipped) > 1e-6
    k.p_new = p_new
    return p_new, s_new, outputs, k


def _well_matrix(c, wc, ctrl, k, b, rhs):
    """The well terms in the pressure equation (``Simulator._add_well_terms``); changes ``b`` and ``rhs`` in place."""
    idx, pid, pv = wc.idx, wc.pid, wc.per_vol
    mode, target = ctrl
    lo, lw, al = k.lam_o[idx], k.lam_w[idx], k.alpha[idx]
    st = SimpleNamespace(mode=mode, T=target, lo=lo, lw=lw, al=al, ibo=k.ibo[idx], ibw=k.ibw[idx])
    if wc.kind == "producer":
        if mode == "bhp":
            st.coef = pid * pv * (lo + al * lw)
            b[idx] -= st.coef
            rhs[idx] -= st.coef * (target + wc.heads)
        elif mode == "oil_rate":
            st.w = pid * lo
            st.share, st.S = _share(st.w)
            st.q_o = target * st.share
            st.ratio = _safe_ratio(lw, lo)
            st.q_w = st.q_o * st.ratio
            rhs[idx] += (st.q_o + al * st.q_w) * pv
        else:                                                               # "liquid_rate"
            st.lt = lo + lw
            st.w = pid * st.lt
            st.share, st.S = _share(st.w)
            st.q_l = target * st.share
            st.ro, st.rw = _safe_ratio(lo, st.lt), _safe_ratio(lw, st.lt)
            st.q_o, st.q_w = st.q_l * st.ro, st.q_l * st.rw
            rhs[idx] += (st.q_o + al * st.q_w) * pv
    else:
        st.mob = lo * st.ibw / st.ibo + lw
        if mode == "water_rate":
            st.w = pid * st.mob
            st.share, st.S = _share(st.w)
            st.q_w = target * st.share
            rhs[idx] -= al * st.q_w * pv
        else:                                                               # "bhp"
            st.coef = pid * pv * st.mob
            b[idx] -= al * st.coef
            rhs[idx] -= al * st.coef * (target + wc.heads)
    return st


def _well_finish(c, wc, ctrl, st, p_new, qop_cell):
    """Rates and bottom-hole pressure after the solve (``Simulator._finish_wells``)."""
    idx, pid, heads = wc.idx, wc.pid, wc.heads
    mode, target = ctrl
    p = p_new[idx]
    if wc.kind == "producer":
        lo, lw = st.lo, st.lw
        if mode == "bhp":
            bhp = target
            q_o = pid * lo * (p - bhp - heads)
            q_w = pid * lw * (p - bhp - heads)
        else:
            q_o, q_w = st.q_o, st.q_w
            st.capacity = float(np.sum(pid * lo))
            bhp = (np.sum(pid * lo * (p - heads)) - q_o.sum()) / st.capacity if st.capacity > 0 else np.nan
        qop_cell[idx] = q_o * BBL_TO_FT3 / wc.vol
    else:
        if mode == "bhp":
            bhp = target
            q_w = pid * st.mob * (bhp + heads - p)
        else:
            q_w = st.q_w
            st.capacity = float(np.sum(pid * st.mob))
            bhp = (q_w.sum() + np.sum(pid * st.mob * (p - heads))) / st.capacity
        q_o = np.zeros(len(idx))
    return dict(bhp=bhp, q_o=q_o, q_w=q_w, qo=float(q_o.sum()), qw=float(q_w.sum()))


# ====================================================================== the backward step (the transpose, by hand)
def _new_accumulators(c):
    return SimpleNamespace(
        G_bar={d: np.zeros(c.n) for d in DIRECTIONS}, phi_bar=np.zeros(c.n),
        pid_bar={wc.name: np.zeros(len(wc.idx)) for wc in c.wells}, ctrl_bar={wc.name: 0.0 for wc in c.wells})


def _mobility_backward(st, idx, mob_bar, B):
    """injector mobility  mob = lam_o * ibw / ibo + lam_w"""
    B.lam_o_bar[idx] += mob_bar * st.ibw / st.ibo
    B.ibw_bar[idx] += mob_bar * st.lo / st.ibo
    B.ibo_bar[idx] += -mob_bar * st.lo * st.ibw / st.ibo**2
    B.lam_w_bar[idx] += mob_bar


def _well_finish_backward(c, wc, st, p_new, out_bar, qop_bar_idx, B, acc):
    """Backward through ``_well_finish``: pressure and property gradients, and the gradients of the planned rates."""
    idx, pid, heads = wc.idx, wc.pid, wc.heads
    p, m = p_new[idx], len(idx)
    bhp_bar, qo_bar_out, qw_bar_out = out_bar.get("bhp", 0.0), out_bar.get("qo", 0.0), out_bar.get("qw", 0.0)
    T = st.T
    if wc.kind == "producer":
        qo_bar = np.full(m, qo_bar_out) + qop_bar_idx * BBL_TO_FT3 / wc.vol
        qw_bar = np.full(m, qw_bar_out)
        lo, lw = st.lo, st.lw
        if st.mode == "bhp":
            dp = p - T - heads
            acc.pid_bar[wc.name] += qo_bar * lo * dp + qw_bar * lw * dp
            B.lam_o_bar[idx] += qo_bar * pid * dp
            B.lam_w_bar[idx] += qw_bar * pid * dp
            dp_bar = qo_bar * pid * lo + qw_bar * pid * lw
            B.p_new_bar[idx] += dp_bar
            acc.ctrl_bar[wc.name] += -dp_bar.sum() + bhp_bar
        else:
            if st.capacity > 0 and np.isfinite(bhp_bar):
                bhp = (np.sum(pid * lo * (p - heads)) - st.q_o.sum()) / st.capacity
                n_bar, cap_bar = bhp_bar / st.capacity, -bhp_bar * bhp / st.capacity
                qo_bar += -bhp_bar / st.capacity
                B.p_new_bar[idx] += n_bar * pid * lo
                B.lam_o_bar[idx] += n_bar * pid * (p - heads) + cap_bar * pid
                acc.pid_bar[wc.name] += n_bar * lo * (p - heads) + cap_bar * lo
            st.qo_bar, st.qw_bar = qo_bar, qw_bar
    else:
        qw_bar = np.full(m, qw_bar_out)
        if st.mode == "bhp":
            drive = T + heads - p
            acc.pid_bar[wc.name] += qw_bar * st.mob * drive
            mob_bar = qw_bar * pid * drive
            B.p_new_bar[idx] -= qw_bar * pid * st.mob
            acc.ctrl_bar[wc.name] += np.sum(qw_bar * pid * st.mob) + bhp_bar
            _mobility_backward(st, idx, mob_bar, B)
        else:
            if np.isfinite(bhp_bar):
                bhp = (st.q_w.sum() + np.sum(pid * st.mob * (p - heads))) / st.capacity
                qw_bar += bhp_bar / st.capacity
                n_bar, cap_bar = bhp_bar / st.capacity, -bhp_bar * bhp / st.capacity
                B.p_new_bar[idx] += n_bar * pid * st.mob
                acc.pid_bar[wc.name] += n_bar * st.mob * (p - heads) + cap_bar * st.mob
                _mobility_backward(st, idx, n_bar * pid * (p - heads) + cap_bar * pid, B)
            st.qw_bar = qw_bar


def _well_matrix_backward(c, wc, st, b_bar, rhs_bar, B, acc):
    """Backward through ``_well_matrix``; ``b_bar`` and ``rhs_bar`` are the gradients at the well blocks."""
    idx, pid, pv, heads = wc.idx, wc.pid, wc.per_vol, wc.heads
    T, lo, lw, al = st.T, st.lo, st.lw, st.al
    if wc.kind == "producer":
        if st.mode == "bhp":
            coef_bar = -b_bar - rhs_bar * (T + heads)
            acc.ctrl_bar[wc.name] += np.sum(-rhs_bar * st.coef)
            acc.pid_bar[wc.name] += coef_bar * pv * (lo + al * lw)
            B.lam_o_bar[idx] += coef_bar * pid * pv
            B.lam_w_bar[idx] += coef_bar * pid * pv * al
            B.alpha_bar[idx] += coef_bar * pid * pv * lw
            return
        qo_bar = st.qo_bar + rhs_bar * pv
        qw_bar = st.qw_bar + rhs_bar * pv * al
        B.alpha_bar[idx] += rhs_bar * pv * st.q_w
        lo_bar, lw_bar = np.zeros_like(lo), np.zeros_like(lo)
        if st.mode == "oil_rate":
            qo_bar = qo_bar + qw_bar * st.ratio                              # q_w = q_o * ratio
            ratio_bar = qw_bar * st.q_o
            pos = lo > 0
            lw_bar[pos] += ratio_bar[pos] / lo[pos]
            lo_bar[pos] += -ratio_bar[pos] * lw[pos] / lo[pos] ** 2
            acc.ctrl_bar[wc.name] += np.sum(qo_bar * st.share)               # q_o = T * share
            w_bar = _share_backward(qo_bar * T, st.share, st.S)
            acc.pid_bar[wc.name] += w_bar * lo                               # w = pid * lam_o
            lo_bar += w_bar * pid
        else:                                                                # liquid_rate
            ql_bar = qo_bar * st.ro + qw_bar * st.rw
            ro_bar, rw_bar = qo_bar * st.q_l, qw_bar * st.q_l
            pos = st.lt > 0
            lo_bar[pos] += ro_bar[pos] / st.lt[pos]
            lw_bar[pos] += rw_bar[pos] / st.lt[pos]
            lt_bar = np.zeros_like(lo)
            lt_bar[pos] += -ro_bar[pos] * lo[pos] / st.lt[pos] ** 2 - rw_bar[pos] * lw[pos] / st.lt[pos] ** 2
            acc.ctrl_bar[wc.name] += np.sum(ql_bar * st.share)               # q_l = T * share
            w_bar = _share_backward(ql_bar * T, st.share, st.S)
            acc.pid_bar[wc.name] += w_bar * st.lt                            # w = pid * lt
            lt_bar += w_bar * pid
            lo_bar += lt_bar                                                 # lt = lo + lw
            lw_bar += lt_bar
        B.lam_o_bar[idx] += lo_bar
        B.lam_w_bar[idx] += lw_bar
        return
    # injectors
    if st.mode == "bhp":
        coef_bar = -al * (b_bar + rhs_bar * (T + heads))
        B.alpha_bar[idx] += -st.coef * (b_bar + rhs_bar * (T + heads))
        acc.ctrl_bar[wc.name] += np.sum(-al * st.coef * rhs_bar)
        acc.pid_bar[wc.name] += coef_bar * pv * st.mob
        _mobility_backward(st, idx, coef_bar * pid * pv, B)
    else:
        qw_bar = st.qw_bar - rhs_bar * al * pv
        B.alpha_bar[idx] += -rhs_bar * st.q_w * pv
        acc.ctrl_bar[wc.name] += np.sum(qw_bar * st.share)
        w_bar = _share_backward(qw_bar * T, st.share, st.S)
        acc.pid_bar[wc.name] += w_bar * st.mob
        _mobility_backward(st, idx, w_bar * pid, B)


def backward_step(c, k, p_new_bar, s_new_bar, out_bars, acc):
    """The transpose of ``forward_step``.

    Input: the gradients of the objective with respect to the new pressure and saturation (``p_new_bar``, ``s_new_bar``,
    from the later steps and from the objective at this time) and with respect to the outputs of the wells
    (``out_bars[well] = {"bhp": .., "qo": .., "qw": ..}``).  Adds the gradients with respect to the geometry factors,
    the porosity, the well indices and the controls to ``acc`` and returns the gradients with respect to the old state.
    """
    g, m, dt, n = c.grid, c.model, c.dt, c.n
    p_old, s_old, p_new = k.p_old, k.s_old, k.p_new
    B = SimpleNamespace(p_new_bar=np.array(p_new_bar, dtype=float), lam_o_bar=np.zeros(n), lam_w_bar=np.zeros(n),
                        ibo_bar=np.zeros(n), ibw_bar=np.zeros(n), alpha_bar=np.zeros(n))
    t_o_bar = {d: np.zeros(n) for d in DIRECTIONS}
    t_w_bar = {d: np.zeros(n) for d in DIRECTIONS}

    # ---- 1. the explicit saturation update:  s_new = clip(s_old + u / cswo),  u = flow - qop - cpoo (p_new - p_old) - gra_o
    s_unc_bar = s_new_bar * k.clip_mask
    s_old_bar = s_unc_bar.copy()
    u_bar = s_unc_bar / k.cswo
    cswo_bar = -s_unc_bar * k.u / k.cswo**2
    qop_bar = -u_bar
    cpoo_bar = -u_bar * (p_new - p_old)
    B.p_new_bar += -k.cpoo * u_bar
    p_old_bar = k.cpoo * u_bar
    gra_o_bar = -u_bar
    for d in DIRECTIONS:                                                     # flow = sum t_o * (p_new[nb] - p_new)
        nb = g.neighbour[d]
        t_o_bar[d] += (p_new[nb] - p_new) * u_bar
        np.add.at(B.p_new_bar, nb, k.t_o[d] * u_bar)
        B.p_new_bar -= k.t_o[d] * u_bar

    # ---- 2. the wells after the solve
    for wc in c.wells:
        _well_finish_backward(c, wc, k.well_store[wc.name], p_new, out_bars.get(wc.name, {}), qop_bar[wc.idx], B, acc)

    # ---- 3. the pressure equation  A p_new = rhs :  lambda = A^-T p_new_bar
    lam = spsolve(k.A.T.tocsc(), B.p_new_bar)
    rhs_bar = lam
    b_bar = -lam * p_new
    coeff_bar = {d: np.where(g.is_open[d], -lam * p_new[g.neighbour[d]], 0.0) for d in DIRECTIONS}

    # ---- 4. back through the assembly of the coefficients
    alpha, cpoo, cpow = k.alpha, k.cpoo, k.cpow
    sum_t_o_bar = -b_bar                                                     # b = -((sum_t_o + cpoo) + alpha (sum_t_w + cpow))
    cpoo_bar += -b_bar
    B.alpha_bar += -b_bar * (k.sum_t_w + cpow)
    sum_t_w_bar = -alpha * b_bar
    cpow_bar = -alpha * b_bar
    gra_o_bar = gra_o_bar + rhs_bar                                          # rhs = cap + gra_o + alpha gra_w - (cpoo + alpha cpow) p_old
    B.alpha_bar += k.gra_w * rhs_bar
    gra_w_bar = alpha * rhs_bar
    cpoo_bar += -p_old * rhs_bar
    B.alpha_bar += -cpow * p_old * rhs_bar
    cpow_bar += -alpha * p_old * rhs_bar
    p_old_bar += -(cpoo + alpha * cpow) * rhs_bar
    B.alpha_bar += k.cap_raw * rhs_bar                                       # cap = alpha * cap_raw
    cap_raw_bar = alpha * rhs_bar
    pcow_bar = np.zeros(n)
    for d in DIRECTIONS:
        nb = g.neighbour[d]
        t_o_bar[d] += coeff_bar[d] + sum_t_o_bar + m.gamma_o * c.head[d] * gra_o_bar      # coeff = t_o + alpha t_w, the sums, gravity
        t_w_bar[d] += (alpha * coeff_bar[d] + sum_t_w_bar + m.gamma_w * c.head[d] * gra_w_bar
                       + (k.pcow[nb] - k.pcow) * cap_raw_bar)
        B.alpha_bar += k.t_w[d] * coeff_bar[d]
        np.add.at(pcow_bar, nb, k.t_w[d] * cap_raw_bar)
        pcow_bar -= k.t_w[d] * cap_raw_bar

    # ---- 5. the wells in the matrix
    for wc in c.wells:
        _well_matrix_backward(c, wc, k.well_store[wc.name], b_bar[wc.idx], rhs_bar[wc.idx], B, acc)

    # ---- 6. the transmissibilities  t = G * (upstream mobility)
    for d in DIRECTIONS:
        nb = g.neighbour[d]
        acc.G_bar[d] += t_o_bar[d] * k.mob_o[d] + t_w_bar[d] * k.mob_w[d]
        mob_o_bar, mob_w_bar = c.G[d] * t_o_bar[d], c.G[d] * t_w_bar[d]
        np.add.at(B.lam_o_bar, nb, np.where(k.flag_o[d], mob_o_bar, 0.0))
        B.lam_o_bar += np.where(k.flag_o[d], 0.0, mob_o_bar)
        np.add.at(B.lam_w_bar, nb, np.where(k.flag_w[d], mob_w_bar, 0.0))
        B.lam_w_bar += np.where(k.flag_w[d], 0.0, mob_w_bar)

    # ---- 7. the storage coefficients
    phi, cr = k.phi, k.cr
    cswo_bar += -B.alpha_bar / k.csww                                        # alpha = -cswo / csww
    csww_bar = B.alpha_bar * k.cswo / k.csww**2
    acc.phi_bar += csww_bar * k.ibw / dt                                     # csww = phi ibw / dt - cpow dpc
    B.ibw_bar += csww_bar * phi / dt
    cpow_bar += -csww_bar * k.dpc
    fw = cr * k.ibw + k.dibw                                                 # cpow = s phi / dt (cr ibw + dibw)
    s_old_bar += cpow_bar * phi / dt * fw
    acc.phi_bar += cpow_bar * s_old / dt * fw
    B.ibw_bar += cpow_bar * s_old * phi / dt * cr
    acc.phi_bar += -cswo_bar * k.ibo / dt                                    # cswo = -phi ibo / dt
    B.ibo_bar += -cswo_bar * phi / dt
    fo = cr * k.ibo + k.dibo                                                 # cpoo = phi (1 - s) / dt (cr ibo + dibo)
    s_old_bar += -cpoo_bar * phi / dt * fo
    acc.phi_bar += cpoo_bar * (1.0 - s_old) / dt * fo
    B.ibo_bar += cpoo_bar * phi * (1.0 - s_old) / dt * cr

    # ---- 8. the fluid properties
    kro_bar = B.lam_o_bar * k.ibo / k.muo                                    # lam_o = kro ibo / muo
    B.ibo_bar += B.lam_o_bar * k.kro / k.muo
    muo_bar = -B.lam_o_bar * k.kro * k.ibo / k.muo**2
    krw_bar = B.lam_w_bar * k.ibw / k.muw                                    # lam_w = krw ibw / muw
    B.ibw_bar += B.lam_w_bar * k.krw / k.muw
    muw_bar = -B.lam_w_bar * k.krw * k.ibw / k.muw**2
    s_old_bar += kro_bar * k.dkro_ds + krw_bar * k.dkrw_ds
    p_old_bar += muo_bar * k.dmuo_dp + B.ibo_bar * k.dibo_dp
    pw_bar = muw_bar * k.dmuw_dpw + B.ibw_bar * k.dibw_dpw
    p_old_bar += pw_bar                                                      # pw = p - pcow
    pcow_bar += -pw_bar
    s_old_bar += pcow_bar * k.dpcow_ds                                       # pcow = Pc(s)
    return p_old_bar, s_old_bar


# ====================================================================== the trajectory
@dataclass
class Trajectory:
    """A forward run of the copy: states, the controls in force and the well outputs of every step."""

    times: np.ndarray                  # start time of every step
    p: list
    s: list                            # states: p[n], s[n] at the start of step n (and the last at the end)
    controls: list                     # per step: {well: (mode, target)}
    scheduled: list                    # per step: {well: is the target one of the schedule (not the min_bhp limit)?}
    outputs: list                      # per step: the outputs of the wells
    cum: list                          # per step: cumulative volumes at its end {"oil", "water_p", "water_i"}
    stop_reason: object = None
    diagnostics: dict = field(default_factory=dict)


def forward_trajectory(model, until, schedule=None, state=None, upstream="potential", tol=1e-6):
    """Run the functional copy from time 0 (or from ``state``) to day ``until`` and keep what the adjoint needs.

    Controls change as in ``kusim.schedule.advance`` (events) and as in the simulator (a producer that reaches its
    ``min_bhp`` goes on bottom-hole-pressure control; ``min_oil_rate`` ends the run)."""
    sim = Simulator(model, save_every=10**9, upstream=upstream)
    if state is not None:
        sim.set_state(state)
    c = _Const(sim)
    wells = {ws.name: ws for ws in [*sim.producers, *sim.injectors]}
    steps, pending = plan_advance(sim.time, until, model.dt, model.t_end, schedule, tol)
    ctl = {name: SimpleNamespace(mode=ws.mode, target=ws.target, scheduled=True) for name, ws in wells.items()}
    cum0 = sim.get_state().cum
    cum = {"oil": sum(v.get("oil", 0.0) for k_, v in cum0.items() if k_.startswith("P")),
           "water_p": sum(v["water"] for k_, v in cum0.items() if k_.startswith("P")),
           "water_i": sum(v["water"] for k_, v in cum0.items() if k_.startswith("I"))}
    p, s, t = sim.p_o.copy(), sim.s_w.copy(), sim.time
    traj = Trajectory(times=[], p=[p], s=[s], controls=[], scheduled=[], outputs=[], cum=[])
    flips = switches = clipped = backflow = 0
    used, previous_flags = 0, None
    for _ in range(steps):
        while used < len(pending) and pending[used].time <= t + tol:
            e = pending[used]
            sim.set_well_control(e.well, e.mode, e.target)                  # validates the event
            ctl[e.well].mode, ctl[e.well].target = wells[e.well].mode, wells[e.well].target
            if e.target is not None:
                ctl[e.well].scheduled = True
            used += 1
        controls = {name: (x.mode, x.target) for name, x in ctl.items()}
        p_new, s_new, outputs, k = forward_step(c, p, s, controls)
        flags = [k.flag_o[d] for d in DIRECTIONS] + [k.flag_w[d] for d in DIRECTIONS]
        if previous_flags is not None:
            flips += int(sum(np.count_nonzero(a != b) for a, b in zip(flags, previous_flags)))
        previous_flags = flags
        clipped += int(np.count_nonzero(k.clipped))
        backflow += sum(int((o["qo"] + o["qw"] if w.kind == "producer" else o["qw"]) < 0.0)
                        for w, o in ((wc, outputs[wc.name]) for wc in c.wells))
        traj.times.append(t)
        traj.controls.append(controls)
        traj.scheduled.append({name: x.scheduled for name, x in ctl.items()})
        traj.outputs.append(outputs)
        for wc in c.wells:
            o = outputs[wc.name]
            if wc.kind == "producer":
                cum["oil"] += o["qo"] * model.dt
                cum["water_p"] += o["qw"] * model.dt
            else:
                cum["water_i"] += o["qw"] * model.dt
        traj.cum.append(dict(cum))
        p, s, t = p_new, s_new, t + model.dt
        traj.p.append(p)
        traj.s.append(s)
        stop = False
        for wc in c.wells:                                                   # the limits of Simulator._apply_well_limits
            if wc.kind != "producer":
                continue
            x, lim = ctl[wc.name], wc.well
            if x.mode != "bhp" and lim.min_bhp is not None and outputs[wc.name]["bhp"] <= lim.min_bhp:
                x.mode, x.target, x.scheduled = "bhp", lim.min_bhp, False
                switches += 1
            if lim.min_oil_rate is not None and outputs[wc.name]["qo"] <= lim.min_oil_rate:
                traj.stop_reason = f"{wc.name}: oil rate fell to {lim.min_oil_rate} STB/day at t = {t:g} days"
                stop = True
        if stop:
            break
    traj.times = np.array(traj.times)
    traj.diagnostics = dict(steps=len(traj.times), upstream_flips=flips, mode_switches=switches, saturation_clipped=clipped,
                            backflow_well_steps=backflow)
    traj.const = c
    return traj


# ====================================================================== the objective and the gradient
@dataclass
class AdjointResult:
    """Objective value and gradients.  ``fields[name]`` is dJ/d(field) per block for kx, ky, kz and porosity;
    ``controls[well]`` is dJ/d(target) for every step where the target is a scheduled one (0 elsewhere);
    ``step_times`` are the step start times; ``initial_pressure`` and ``initial_saturation`` are the gradients with
    respect to the initial state; ``diagnostics`` count the discrete decisions that make the objective non-smooth."""

    value: float
    fields: dict
    controls: dict
    step_times: np.ndarray
    initial_pressure: np.ndarray
    initial_saturation: np.ndarray
    diagnostics: dict
    simulated: dict = field(default_factory=dict)         # (time, vector) -> simulated value
    trajectory: object = None


def _vector_value(name, outputs, p_new, cum, c):
    """Simulated value of a summary vector at the end of a step."""
    kind, _, well = name.partition(":")
    if kind == "WBHP":
        return outputs[well]["bhp"]
    if kind == "WOPR":
        return outputs[well]["qo"]
    if kind in ("WWPR", "WWIR"):
        return outputs[well]["qw"]
    if kind == "WWCT":
        o = outputs[well]
        liquid = o["qo"] + o["qw"]
        return o["qw"] / liquid if liquid > 0 else float("nan")
    pv = c.grid.volume * c.model.porosity
    if name == "FPRP":
        return float((p_new * pv).sum() / pv.sum())
    return {"FOPT": cum["oil"], "FWPT": cum["water_p"], "FWIT": cum["water_i"]}[name]


def adjoint_gradient(model, observations=None, schedule=None, state=None, upstream="potential", linear_terms=None) -> AdjointResult:
    """Objective and gradient.  The objective is ``J = sum weight * ((simulated - observed) / sigma)**2`` over the
    ``observations`` (history matching) plus ``sum weight * simulated`` over the ``linear_terms`` of
    ``kusim.history.linear_terms`` (for example an economic objective); either may be omitted.

    The run covers the times of the terms (from time 0, or from ``state``) with the controls of ``schedule``.  The times
    must be multiples of ``dt`` after the start.  See the module docstring for what the gradient does not include."""
    terms = []                                          # (time, vector, observed, sigma, weight, linear?)
    if observations is not None:
        terms += [(r.time, r.vector, r.value, r.sigma, r.weight, False) for r in observations.table.itertuples()]
    if linear_terms is not None:
        terms += [(r.time, r.vector, 0.0, 1.0, r.weight, True) for r in linear_terms.itertuples()]
    if not terms:
        raise ValueError("give observations and/or linear_terms: there is no objective")
    for v in {t[1] for t in terms}:
        if v.partition(":")[0] not in SUPPORTED_VECTORS and v not in SUPPORTED_VECTORS:
            raise ValueError(f"the adjoint does not support the vector {v!r}; supported: {SUPPORTED_VECTORS}")
    until = float(max(t[0] for t in terms))
    traj = forward_trajectory(model, until, schedule, state, upstream)
    c, dt, n = traj.const, model.dt, traj.const.n
    t0 = traj.times[0] if len(traj.times) else 0.0
    by_step, simulated, value = {}, {}, 0.0
    for time, vector, observed, sigma, weight, linear in terms:
        step = (time - t0) / dt
        if abs(step - round(step)) > 1e-6 or round(step) < 1:
            raise ValueError(f"the term at day {time} is not at the end of a step of the run (steps of {dt} days after day {t0})")
        step = int(round(step)) - 1
        if step >= len(traj.times):
            raise ValueError(f"the run stopped before day {time}: {traj.stop_reason}")
        y = _vector_value(vector, traj.outputs[step], traj.p[step + 1], traj.cum[step], c)
        simulated[(time, vector)] = y
        if np.isnan(observed) or np.isnan(y):
            continue
        if linear:
            value += weight * y
            by_step.setdefault(step, []).append((vector, weight))
        else:
            res = (y - observed) / sigma
            value += weight * res**2
            by_step.setdefault(step, []).append((vector, 2.0 * weight * res / sigma))

    acc = _new_accumulators(c)
    pv = c.grid.volume * c.model.porosity
    p_bar, s_bar = np.zeros(n), np.zeros(n)
    cum_bar = {"oil": 0.0, "water_p": 0.0, "water_i": 0.0}
    step_gradients = {wc.name: np.zeros(len(traj.times)) for wc in c.wells}
    for step in range(len(traj.times) - 1, -1, -1):
        out_bars = {wc.name: {} for wc in c.wells}
        p_new_bar = p_bar.copy()
        p_new = traj.p[step + 1]
        for vec, ybar in by_step.get(step, []):
            kind, _, well = vec.partition(":")
            if kind == "WBHP":
                out_bars[well]["bhp"] = out_bars[well].get("bhp", 0.0) + ybar
            elif kind == "WOPR":
                out_bars[well]["qo"] = out_bars[well].get("qo", 0.0) + ybar
            elif kind in ("WWPR", "WWIR"):
                out_bars[well]["qw"] = out_bars[well].get("qw", 0.0) + ybar
            elif kind == "WWCT":
                o = traj.outputs[step][well]
                liquid = o["qo"] + o["qw"]
                out_bars[well]["qw"] = out_bars[well].get("qw", 0.0) + ybar * o["qo"] / liquid**2
                out_bars[well]["qo"] = out_bars[well].get("qo", 0.0) - ybar * o["qw"] / liquid**2
            elif vec == "FPRP":
                fprp = float((p_new * pv).sum() / pv.sum())
                p_new_bar += ybar * pv / pv.sum()
                acc.phi_bar += ybar * c.grid.volume * (p_new - fprp) / pv.sum()
            else:
                cum_bar[{"FOPT": "oil", "FWPT": "water_p", "FWIT": "water_i"}[vec]] += ybar
        for wc in c.wells:                                                   # cumulative volumes: d cum / d rate = dt
            key = ("oil", "water_p") if wc.kind == "producer" else (None, "water_i")
            if key[0] is not None:
                out_bars[wc.name]["qo"] = out_bars[wc.name].get("qo", 0.0) + dt * cum_bar["oil"]
            out_bars[wc.name]["qw"] = out_bars[wc.name].get("qw", 0.0) + dt * cum_bar[key[1]]
        _, _, _, k = forward_step(c, traj.p[step], traj.s[step], traj.controls[step])        # reconstruct this step
        before = {name: x for name, x in acc.ctrl_bar.items()}
        p_bar, s_bar = backward_step(c, k, p_new_bar, s_bar, out_bars, acc)
        for name in acc.ctrl_bar:
            gradient = acc.ctrl_bar[name] - before[name]
            step_gradients[name][step] = gradient if traj.scheduled[step][name] else 0.0

    fields = _field_gradients(model, c, acc)
    diagnostics = dict(traj.diagnostics, stop_reason=traj.stop_reason)
    return AdjointResult(value, fields, step_gradients, traj.times, p_bar, s_bar, diagnostics, simulated, traj)


def _field_gradients(model, c, acc):
    """From the gradients of the geometry factors, the well indices and the porosity to the gradients of kx, ky, kz."""
    g = c.grid
    perm_of = {"x": "kx", "y": "ky", "z": "kz"}
    size_of = {"x": g.dx, "y": g.dy, "z": g.dz}
    out = {name: np.zeros(c.n) for name in ("kx", "ky", "kz")}
    for d in DIRECTIONS:
        axis = d[0]
        perm, size, nb = getattr(model, perm_of[axis]), size_of[axis], g.neighbour[d]
        a = size[nb] / perm[nb] + size / perm
        scale = DARCY * (2.0 / size) * g.is_open[d] * acc.G_bar[d] / a**2          # G = DARCY (2 / size) / a, a = size_nb / k_nb + size / k
        out[perm_of[axis]] += scale * size / perm**2
        np.add.at(out[perm_of[axis]], nb, scale * size[nb] / perm[nb] ** 2)
    for wc in c.wells:
        for j, cell in enumerate(wc.idx):
            _, dkx, dky = _well_index_derivatives(model.kx[cell], model.ky[cell], g.dx[cell], g.dy[cell], g.dz[cell],
                                                  wc.well.radius, wc.well.skin)
            out["kx"][cell] += acc.pid_bar[wc.name][j] * dkx
            out["ky"][cell] += acc.pid_bar[wc.name][j] * dky
    out["porosity"] = acc.phi_bar
    return out
