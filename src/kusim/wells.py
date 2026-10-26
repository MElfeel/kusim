"""Wells: producers and injectors, and the well productivity index.

A well is completed in one grid block (given by its thesis block number, from 1) or in
several blocks, usually the blocks of one vertical column.  Use ``grid.column_blocks(i, j)``
to list the blocks of a column:

    Producer(block=grid.column_blocks(10, 10), mode="bhp", target=2500.0)

Each completion block ("perforation") has its own well index.  A well with a fixed
bottom-hole pressure draws from every perforation against that pressure.  A well with a
fixed rate shares the rate between its perforations in proportion to how easily each one
can flow (well index times mobility); the bottom-hole pressure then follows from the rate.
The pressure inside the wellbore increases with depth by the weight of the fluid in it
(``wellbore_gravity``, a specific gravity; by default oil for a producer and water for an
injector).  The bottom-hole pressure of a well is the pressure at its shallowest perforation.
Without this, the deeper blocks of a column, which are naturally at higher pressure, would
appear to flow backwards into the well.

Producer control modes (``PBC`` in the original input):

    "oil_rate"     constant surface oil rate               (PBC = 1)
    "liquid_rate"  constant surface oil + water rate       (PBC = 2)
    "bhp"          constant bottom-hole pressure           (PBC = 3)

Injector control modes (``IBC`` in the original input):

    "water_rate"   constant surface water injection rate   (IBC = 1)
    "bhp"          constant bottom-hole pressure           (IBC = 2)

Rates are in STB/day and are always entered as positive numbers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

from .constants import WELL_INDEX_CONSTANT

PRODUCER_MODES = ("oil_rate", "liquid_rate", "bhp")
INJECTOR_MODES = ("water_rate", "bhp")


@dataclass
class Producer:
    """A producing well.

    Parameters
    ----------
    block : int or list of int
        Block number (from 1) the well is completed in, or a list of block numbers for a
        well completed in several layers (see ``Grid.column_blocks``).
    mode : str
        "oil_rate", "liquid_rate" or "bhp".
    target : float
        STB/day for the rate modes, psi for "bhp".
    radius : float
        Wellbore radius (ft).
    skin : float
        Skin factor.
    min_bhp : float, optional
        If the bottom-hole pressure falls to this value (psi), the well
        switches to constant bottom-hole pressure at this value.
    min_oil_rate : float, optional
        When the oil rate falls to this value (STB/day) the run stops.
    wellbore_gravity : float, optional
        Specific gravity of the fluid in the wellbore, which sets how the well pressure changes
        between perforations at different depths.  Default: the oil gravity.  Use 0 to ignore it.
    """

    block: Union[int, Sequence[int]]
    mode: str
    target: float
    radius: float = 0.25
    skin: float = 0.0
    min_bhp: Optional[float] = None
    min_oil_rate: Optional[float] = None
    wellbore_gravity: Optional[float] = None

    def __post_init__(self):
        if self.mode not in PRODUCER_MODES:
            raise ValueError(f"producer mode must be one of {PRODUCER_MODES}, got {self.mode!r}")
        self.blocks = _as_block_list(self.block)


@dataclass
class Injector:
    """A water injection well (see ``Producer`` for the shared parameters)."""

    block: Union[int, Sequence[int]]
    mode: str
    target: float
    radius: float = 0.25
    skin: float = 0.0
    wellbore_gravity: Optional[float] = None       # default: the water gravity

    def __post_init__(self):
        if self.mode not in INJECTOR_MODES:
            raise ValueError(f"injector mode must be one of {INJECTOR_MODES}, got {self.mode!r}")
        self.blocks = _as_block_list(self.block)


def _as_block_list(block) -> List[int]:
    """One block number or a sequence of them, as a list without repeats."""
    if hasattr(block, "__index__"):          # a single number (int or NumPy integer)
        blocks = [int(block)]
    else:
        blocks = [int(b) for b in block]
    if not blocks:
        raise ValueError("a well needs at least one completion block")
    if len(set(blocks)) != len(blocks):
        raise ValueError(f"a block is listed twice in the completions {blocks}")
    return blocks


def equivalent_radius(kx, ky, dx, dy):
    """Peaceman drainage radius re of a well block (Chapter 5, eq. 5-73), in ft."""
    ratio = ky / kx
    top = math.sqrt(math.sqrt(ratio) * dx**2 + math.sqrt(1.0 / ratio) * dy**2)
    bottom = ratio**0.25 + (1.0 / ratio) ** 0.25
    return 0.28 * top / bottom


def well_index(kx, ky, dx, dy, dz, radius, skin):
    """Well productivity index PID (Chapter 5, eq. 5-72).

    Surface rate (STB/day) = PID * mobility * (P_block - P_bottom_hole), where
    the mobility is kr / (mu * B).  Units: md, ft, ft, ft, ft, skin dimensionless.
    """
    k_abs = math.sqrt(kx * ky)
    re = equivalent_radius(kx, ky, dx, dy)
    return WELL_INDEX_CONSTANT * k_abs * dz / (math.log(re / radius) + skin)
