"""Unit-conversion constants (oilfield units).

Units used throughout KUSIM:

    length      ft          permeability   md
    pressure    psi         viscosity      cp
    time        day         rates          STB/day (surface barrels)
    volumes     ft^3        B_o, B_w       rb/STB
"""

# Gradient of a fluid column: psi/ft per unit specific gravity (water = 1.0).
# Gravity term of a phase:  gamma * 0.433 * (depth difference)   [psi]
PSI_PER_FT = 0.433

# Darcy's law in oilfield units, written per unit *bulk volume* of a cell.
#   0.001127 converts  md*ft^2*psi / (cp*ft)  into  bbl/day,
#   and 5.615 ft^3/bbl converts bbl into ft^3, so 0.001127 * 5.615 = 0.006328.
BBL_TO_FT3 = 5.615
DARCY = 0.006328

# Peaceman well index constant: 2*pi*0.001127 = 0.00708 (rb/day per md*ft/cp/psi).
WELL_INDEX_CONSTANT = 0.00708
