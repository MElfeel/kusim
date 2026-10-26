# Chapter 5: KUSIM 3-D Two-Phase Flow Equations

> Transcribed from `legacy/thesis/Chapeter V-KUSIM Equations.pdf` (page images in
> `legacy/thesis/chapter V-KUSIM Equations images/`). Original equation numbers are kept.
> Items marked **[Note]** are transcription remarks (typos or inconsistencies in the original
> that matter when porting). Equations are field units: pressure in psi, depth in ft,
> and the constant 0.433 converts a fluid gradient from specific gravity to psi/ft.

## 5.1 Derivation of basic equations used in KUSIM

Continuity equation for each fluid phase in a 3-D Cartesian system:

$$-\left(\frac{\partial}{\partial x}(\rho_o u_x)+\frac{\partial}{\partial y}(\rho_o u_y)+\frac{\partial}{\partial z}(\rho_o u_z)\right)=\frac{\partial}{\partial t}(\phi\rho_o S_o)\tag{5-1}$$

$$-\left(\frac{\partial}{\partial x}(\rho_w u_x)+\frac{\partial}{\partial y}(\rho_w u_y)+\frac{\partial}{\partial z}(\rho_w u_z)\right)=\frac{\partial}{\partial t}(\phi\rho_w S_w)\tag{5-2}$$

The corresponding Darcy equations for each phase $l$ ($l = o, w$):

$$u_x=-\frac{K_x k_{rl}}{\mu_l}\left(\frac{\partial P_l}{\partial x}-\gamma\frac{\partial D}{\partial x}\right)\tag{5-3}$$

$$u_y=-\frac{K_y k_{rl}}{\mu_l}\left(\frac{\partial P_l}{\partial y}-\gamma\frac{\partial D}{\partial y}\right)\tag{5-4}$$

$$u_z=-\frac{K_z k_{rl}}{\mu_l}\left(\frac{\partial P_l}{\partial z}-\gamma\frac{\partial D}{\partial z}\right)\tag{5-5}$$

where $\gamma=\rho g$ is the hydrostatic gradient of the fluid.

Substituting Darcy's equations and standard black-oil fluid descriptions into the continuity
equations, and including production/injection terms, gives the flow equations for the two phases.

### 5.1.1 Oil phase equation

$$\frac{\partial}{\partial x}\left(\frac{k_x k_{ro}}{\mu_o B_o}\left(\frac{\partial P_o}{\partial x}-\gamma_o\frac{\partial D}{\partial x}\right)\right)+\frac{\partial}{\partial y}\left(\frac{k_y k_{ro}}{\mu_o B_o}\left(\frac{\partial P_o}{\partial y}-\gamma_o\frac{\partial D}{\partial y}\right)\right)+\frac{\partial}{\partial z}\left(\frac{k_z k_{ro}}{\mu_o B_o}\left(\frac{\partial P_o}{\partial z}-\gamma_o\frac{\partial D}{\partial z}\right)\right)-q_o'=\frac{\partial}{\partial t}\left(\frac{\phi S_o}{B_o}\right)\tag{5-6}$$

### 5.2.1 Water phase equation

$$\frac{\partial}{\partial x}\left(\frac{k_x k_{rw}}{\mu_w B_w}\left(\frac{\partial P_w}{\partial x}-\gamma_w\frac{\partial D}{\partial x}\right)\right)+\frac{\partial}{\partial y}\left(\frac{k_y k_{rw}}{\mu_w B_w}\left(\frac{\partial P_w}{\partial y}-\gamma_w\frac{\partial D}{\partial y}\right)\right)+\frac{\partial}{\partial z}\left(\frac{k_z k_{rw}}{\mu_w B_w}\left(\frac{\partial P_w}{\partial z}-\gamma_w\frac{\partial D}{\partial z}\right)\right)-q_w'=\frac{\partial}{\partial t}\left(\frac{\phi S_w}{B_w}\right)\tag{5-7}$$

Auxiliary equations:

### 5.3.1 Capillary pressure

$$P_w = P_o - P_{cow}\tag{5-8}$$

### 5.4.1 Saturations

$$S_o + S_w = 1\tag{5-9}$$

Relative permeabilities and capillary pressure are functions of water saturation, while
formation volume factors, viscosities and porosity are functions of pressure.

**Drainage and imbibition curves** (Figures 5.1 and 5.2 in the original):

- *Drainage* (Fig. 5.1): rock initially $S_w = 1$, oil displaces water. Oil relative permeability
  rises from 0 as $S_w$ falls toward $S_{wir}$; capillary pressure rises as $S_w$ decreases,
  with a displacement pressure $P_{cd}$ at $S_w = 1$.
- *Imbibition* (Fig. 5.2): rock at $S_w = S_{wir}$, water displaces oil. Water relative
  permeability rises as $S_w$ increases up to $1 - S_{or}$ (the figure labels it "1-").
- The curves are typical of a completely water-wet system. For less water-wet systems the
  capillary pressure curve has a negative part at high water saturation. The curve shapes
  depend on rock and wetting characteristics.

## 5.2 KUSIM discretization: solution by the IMPES method

For two-phase oil/water 3-D flow the governing equations are (5-6) and (5-7) above.

### A. Left-side flow terms

#### The oil phase

**1. On the x axis:**

$$\left.\frac{\partial}{\partial x}\left(\frac{k_x k_{ro}}{\mu_o B_o}\left(\frac{\partial P_o}{\partial x}-0.433\gamma_o\frac{\partial D}{\partial x}\right)\right)\right|_{i,j,k}\approx T_{xo\,i+\frac12,j,k}\left[\left(P_{o\,i+1,j,k}-P_{o\,i,j,k}\right)_i-0.433\gamma_o\left(D_{i+1,j,k}-D_{i,j,k}\right)_i\right]+T_{xo\,i-\frac12,j,k}\left[\left(P_{o\,i-1,j,k}-P_{o\,i,j,k}\right)_i-0.433\gamma_w\left(D_{i-1,j,k}-D_{i,j,k}\right)_i\right]\tag{5-10}$$

> **[Note]** In the original the second bracket of (5-10) uses $\gamma_w$ although this is the oil
> phase. It should be $\gamma_o$.

**2. On the y axis:**

$$\left.\frac{\partial}{\partial y}\left(\frac{k_z k_{ro}}{\mu_o B_o}\left(\frac{\partial P_o}{\partial y}-0.433\gamma_o\frac{\partial D}{\partial y}\right)\right)\right|_{i,j,k}\approx T_{yo\,i,j+\frac12,k}\left[\left(P_{o\,i,j+1,k}-P_{o\,i,j,k}\right)_k-0.433\gamma_o\left(D_{i,j+1,k}-D_{i,j,k}\right)_k\right]+T_{yo\,i,j-\frac12,k}\left[\left(P_{o\,i,j-1,k}-P_{o\,i,j,k}\right)_k-0.433\gamma_o\left(D_{i,j+1,k}-D_{i,j,k}\right)_k\right]\tag{5-11}$$

> **[Note]** Typos in the original (5-11): the coefficient is written $k_z$ (should be $k_y$), and
> the second bracket uses $D_{i,j+1,k}$ (should be $D_{i,j-1,k}$). Equations (5-14) repeats the same typos for water.

**3. On the z axis:**

$$\left.\frac{\partial}{\partial z}\left(\frac{k_z k_{ro}}{\mu_o B_o}\left(\frac{\partial P_o}{\partial z}-0.433\gamma_o\frac{\partial D}{\partial z}\right)\right)\right|_{i,j,k}\approx T_{zo\,i,j,k+\frac12}\left[\left(P_{o\,i,j,k+1}-P_{o\,i,j,k}\right)_k-0.433\gamma_o\left(D_{i,j,k+1}-D_{i,j,k}\right)_k\right]+T_{zo\,i,j,k-\frac12}\left[\left(P_{o\,i,j,k-1}-P_{o\,i,j,k}\right)_k-0.433\gamma_o\left(D_{i,j,k-1}-D_{i,j,k}\right)_k\right]\tag{5-12}$$

#### The water phase

**4. On the x axis:**

$$\left.\frac{\partial}{\partial x}\left(\frac{k_x k_{rw}}{\mu_w B_w}\left(\frac{\partial P_w}{\partial x}-0.433\gamma_w\frac{\partial D}{\partial x}\right)\right)\right|_{i,j,k}\approx T_{xw\,i+\frac12,j,k}\left[\left(P_{w\,i+1,j,k}-P_{w\,i,j,k}\right)_i-0.433\gamma_w\left(D_{i+1,j,k}-D_{i,j,k}\right)_i\right]+T_{xw\,i-\frac12,j,k}\left[\left(P_{w\,i-1,j,k}-P_{w\,i,j,k}\right)_i-0.433\gamma_w\left(D_{i-1,j,k}-D_{i,j,k}\right)_i\right]\tag{5-13}$$

**5. On the y axis:**

$$\left.\frac{\partial}{\partial y}\left(\frac{k_z k_{rw}}{\mu_w B_w}\left(\frac{\partial P_w}{\partial y}-0.433\gamma_w\frac{\partial D}{\partial y}\right)\right)\right|_{i,j,k}\approx T_{yw\,i,j+\frac12,k}\left[\left(P_{w\,i,j+1,k}-P_{w\,i,j,k}\right)_k-0.433\gamma_w\left(D_{i,j+1,k}-D_{i,j,k}\right)_k\right]+T_{yw\,i,j-\frac12,k}\left[\left(P_{w\,i,j-1,k}-P_{w\,i,j,k}\right)_k-0.433\gamma_w\left(D_{i,j+1,k}-D_{i,j,k}\right)_k\right]\tag{5-14}$$

**6. On the z axis:**

$$\left.\frac{\partial}{\partial z}\left(\frac{k_z k_{rw}}{\mu_w B_w}\left(\frac{\partial P_w}{\partial z}-0.433\gamma_w\frac{\partial D}{\partial z}\right)\right)\right|_{i,j,k}\approx T_{zw\,i,j,k+\frac12}\left[\left(P_{w\,i,j,k+1}-P_{w\,i,j,k}\right)_k-0.433\gamma_w\left(D_{i,j,k+1}-D_{i,j,k}\right)_k\right]+T_{zw\,i,j,k-\frac12}\left[\left(P_{w\,i,j,k-1}-P_{w\,i,j,k}\right)_k-0.433\gamma_w\left(D_{i,j,k-1}-D_{i,j,k}\right)_k\right]\tag{5-15}$$

### Transmissibilities

Each transmissibility uses the phase mobility at the interface $\lambda_{l}$ (see "Mobility terms")
and a harmonic average of the neighbouring block permeabilities weighted by block size.
Written for the x direction (the y and z forms replace $\Delta x$ by $\Delta y$ or $\Delta z$).

**I. Oil phase**

$$T_{xo\,i+\frac12,j,k}=\frac{2\lambda_{o\,i+\frac12,j,k}}{\Delta x_{i,j,k}\left(\dfrac{\Delta x_{i+1,j,k}}{k_{i+1,j,k}}+\dfrac{\Delta x_{i,j,k}}{k_{i,j,k}}\right)}\tag{5-16}$$

$$T_{xo\,i-\frac12,j,k}=\frac{2\lambda_{o\,i-\frac12,j,k}}{\Delta x_{i,j,k}\left(\dfrac{\Delta x_{i-1,j,k}}{k_{i-1,j,k}}+\dfrac{\Delta x_{i,j,k}}{k_{i,j,k}}\right)}\tag{5-17}$$

$$T_{yo\,i,j+\frac12,k}=\frac{2\lambda_{o\,i,j+\frac12,k}}{\Delta y_{i,j,k}\left(\dfrac{\Delta y_{i,j+1,k}}{k_{i,j+1,k}}+\dfrac{\Delta y_{i,j,k}}{k_{i,j,k}}\right)}\tag{5-18}$$

$$T_{yo\,i,j-\frac12,k}=\frac{2\lambda_{o\,i,j-\frac12,k}}{\Delta y_{i,j,k}\left(\dfrac{\Delta y_{i,j-1,k}}{k_{i,j-1,k}}+\dfrac{\Delta y_{i,j,k}}{k_{i,j,k}}\right)}\tag{5-19}$$

$$T_{zo\,i,j,k+\frac12}=\frac{2\lambda_{o\,i,j,k+\frac12}}{\Delta z_{i,j,k}\left(\dfrac{\Delta z_{i,j,k+1}}{k_{i,j,k+1}}+\dfrac{\Delta z_{i,j,k}}{k_{i,j,k}}\right)}\tag{5-20}$$

$$T_{zo\,i,j,k-\frac12}=\frac{2\lambda_{o\,i,j,k-\frac12}}{\Delta z_{i,j,k}\left(\dfrac{\Delta z_{i,j,k-1}}{k_{i,j,k-1}}+\dfrac{\Delta z_{i,j,k}}{k_{i,j,k}}\right)}\tag{5-21}$$

**II. Water phase**

$$T_{xw\,i+\frac12,j,k}=\frac{2\lambda_{w\,i+\frac12,j,k}}{\Delta x_{i,j,k}\left(\dfrac{\Delta x_{i+1,j,k}}{k_{i+1,j,k}}+\dfrac{\Delta x_{i,j,k}}{k_{i,j,k}}\right)}\tag{5-22}$$

$$T_{xw\,i-\frac12,j,k}=\frac{2\lambda_{w\,i-\frac12,j,k}}{\Delta x_{i,j,k}\left(\dfrac{\Delta x_{i-1,j,k}}{k_{i-1,j,k}}+\dfrac{\Delta x_{i,j,k}}{k_{i,j,k}}\right)}\tag{5-23}$$

$$T_{yw\,i,j+\frac12,k}=\frac{2\lambda_{w\,i,j+\frac12,k}}{\Delta y_{i,j,k}\left(\dfrac{\Delta y_{i,j+1,k}}{k_{i,j+1,k}}+\dfrac{\Delta y_{i,j,k}}{k_{i,j,k}}\right)}\tag{5-24}$$

$$T_{yw\,i,j-\frac12,k}=\frac{2\lambda_{w\,i,j-\frac12,k}}{\Delta y_{i,j,k}\left(\dfrac{\Delta y_{i,j-1,k}}{k_{i,j-1,k}}+\dfrac{\Delta y_{i,j,k}}{k_{i,j,k}}\right)}\tag{5-25}$$

$$T_{zw\,i,j,k+\frac12}=\frac{2\lambda_{w\,i,j,k+\frac12}}{\Delta z_{i,j,k}\left(\dfrac{\Delta z_{i,j,k+1}}{k_{i,j,k+1}}+\dfrac{\Delta z_{i,j,k}}{k_{i,j,k}}\right)}\tag{5-26}$$

$$T_{zw\,i,j,k-\frac12}=\frac{2\lambda_{w\,i,j,k-\frac12}}{\Delta z_{i,j,k}\left(\dfrac{\Delta z_{i,j,k-1}}{k_{i,j,k-1}}+\dfrac{\Delta z_{i,j,k}}{k_{i,j,k}}\right)}\tag{5-27}$$

### Mobility terms

$$\lambda_o=\frac{k_{ro}}{\mu_o B_o}\qquad\qquad\lambda_w=\frac{k_{rw}}{\mu_w B_w}$$

> **[Note]** The printed definitions write $K_{ro}$ / $K_{rw}$ (capital K). These are the relative
> permeabilities $k_{ro}$, $k_{rw}$; absolute permeability enters through the transmissibility.

The mobility is now a function of saturation as well as pressure, so the solution is strongly
influenced by how it is evaluated at block interfaces. The Buckley-Leverett solution
for water displacing oil in 1-D (a sharp front, for negligible capillary pressure) is used to
illustrate this. Two approximations for flow between blocks $i$ and $i+1$ are compared:

**a. Weighted-average selection**

$$\lambda_{o\,i+\frac12}=\frac{\Delta x_i\lambda_{oi}+\Delta x_{i+1}\lambda_{o\,i+1}}{\Delta x_i+\Delta x_{i+1}}$$

**b. Upstream selection**

$$\lambda_{o\,i+\frac12}=\lambda_{oi}$$

> **[Note]** The original labels these the other way round: the first formula is headed
> "Upstream selection" and the second "Weighted average selection", but the formulas are
> clearly the weighted average and the upstream value respectively, and the figure and following
> text agree with that.

Figure (original p. 85) compares both against the exact Buckley-Leverett profile: the upstream
selection tracks the exact solution closely, while the weighted average misplaces the front (in the figure
it lags behind the exact front) and smears it. The text explains that the oil outflow from a block depends mainly on that block's own $k_{ro}$:
with averaging, block $i$ may have reached residual oil saturation while block $i+1$ still has
non-zero mobility, so flow out of block $i$ does not stop when it should. For small blocks the
difference is small, but for blocks of practical size it is a significant error.

**In reservoir simulation, upstream mobilities are normally used.** The upstream block is the one
with the higher phase pressure:

x plus direction:

$$\lambda_{o\,i+\frac12,j,k}=\begin{cases}\lambda_{o\,i+1,j,k}&\text{if }P_{o\,i+1,j,k}\ge P_{o\,i,j,k}\\\lambda_{o\,i,j,k}&\text{if }P_{o\,i+1,j,k}<P_{o\,i,j,k}\end{cases}\qquad\lambda_{w\,i+\frac12,j,k}=\begin{cases}\lambda_{w\,i+1,j,k}&\text{if }P_{w\,i+1,j,k}\ge P_{w\,i,j,k}\\\lambda_{w\,i,j,k}&\text{if }P_{w\,i+1,j,k}<P_{w\,i,j,k}\end{cases}\tag{5-28}$$

x minus direction:

$$\lambda_{o\,i-\frac12,j,k}=\begin{cases}\lambda_{o\,i-1,j,k}&\text{if }P_{o\,i-1,j,k}\ge P_{o\,i,j,k}\\\lambda_{o\,i,j,k}&\text{if }P_{o\,i-1,j,k}<P_{o\,i,j,k}\end{cases}\qquad\lambda_{w\,i-\frac12,j,k}=\begin{cases}\lambda_{w\,i-1,j,k}&\text{if }P_{w\,i-1,j,k}\ge P_{w\,i,j,k}\\\lambda_{w\,i,j,k}&\text{if }P_{w\,i-1,j,k}<P_{w\,i,j,k}\end{cases}\tag{5-29}$$

y plus direction:

$$\lambda_{o\,i,j+\frac12,k}=\begin{cases}\lambda_{o\,i,j+1,k}&\text{if }P_{o\,i,j+1,k}\ge P_{o\,i,j,k}\\\lambda_{o\,i,j,k}&\text{if }P_{o\,i,j+1,k}<P_{o\,i,j,k}\end{cases}\qquad\lambda_{w\,i,j+\frac12,k}=\begin{cases}\lambda_{w\,i,j+1,k}&\text{if }P_{w\,i,j+1,k}\ge P_{w\,i,j,k}\\\lambda_{w\,i,j,k}&\text{if }P_{w\,i,j+1,k}<P_{w\,i,j,k}\end{cases}\tag{5-30}$$

y minus direction:

$$\lambda_{o\,i,j-\frac12,k}=\begin{cases}\lambda_{o\,i,j-1,k}&\text{if }P_{o\,i,j-1,k}\ge P_{o\,i,j,k}\\\lambda_{o\,i,j,k}&\text{if }P_{o\,i,j-1,k}<P_{o\,i,j,k}\end{cases}\qquad\lambda_{w\,i,j-\frac12,k}=\begin{cases}\lambda_{w\,i,j-1,k}&\text{if }P_{w\,i,j-1,k}\ge P_{w\,i,j,k}\\\lambda_{w\,i,j,k}&\text{if }P_{w\,i,j-1,k}<P_{w\,i,j,k}\end{cases}\tag{5-31}$$

z plus direction:

$$\lambda_{o\,i,j,k+\frac12}=\begin{cases}\lambda_{o\,i,j,k+1}&\text{if }P_{o\,i,j,k+1}\ge P_{o\,i,j,k}\\\lambda_{o\,i,j,k}&\text{if }P_{o\,i,j,k+1}<P_{o\,i,j,k}\end{cases}\qquad\lambda_{w\,i,j,k+\frac12}=\begin{cases}\lambda_{w\,i,j,k+1}&\text{if }P_{w\,i,j,k+1}\ge P_{w\,i,j,k}\\\lambda_{w\,i,j,k}&\text{if }P_{w\,i,j,k+1}<P_{w\,i,j,k}\end{cases}\tag{5-32}$$

z minus direction:

$$\lambda_{o\,i,j,k-\frac12}=\begin{cases}\lambda_{o\,i,j,k-1}&\text{if }P_{o\,i,j,k-1}\ge P_{o\,i,j,k}\\\lambda_{o\,i,j,k}&\text{if }P_{o\,i,j,k-1}<P_{o\,i,j,k}\end{cases}\qquad\lambda_{w\,i,j,k-\frac12}=\begin{cases}\lambda_{w\,i,j,k-1}&\text{if }P_{w\,i,j,k-1}\ge P_{w\,i,j,k}\\\lambda_{w\,i,j,k}&\text{if }P_{w\,i,j,k-1}<P_{w\,i,j,k}\end{cases}\tag{5-33}$$

> **[Note]** This matches the FORTRAN upstream-mobility block (`IF (PO(I+1).GE.PO(I)) ...`,
> lines ~312-390 of `3DTP(11).FOR`).
>
> **[Deviation in `kusim`]** The rule above (higher phase pressure) is only right without gravity. `kusim` now
> chooses the upstream block by the sign of the phase potential difference
> $(P_{\alpha,nb}-P_\alpha)-\gamma_\alpha\,0.433\,(z_{nb}-z)\ge 0$, which is the same as the thesis rule for
> blocks at equal depth and differs in vertical flow. See `docs/porting-notes.md`, decision 18. The thesis rule is
> available as `Simulator(model, upstream="pressure")`.

### B. Right-side (accumulation) terms

The right-hand side of the oil equation expands as

$$\frac{\partial}{\partial t}\left(\frac{\phi S_o}{B_o}\right)=\frac{\phi}{B_o}\frac{\partial S_o}{\partial t}+S_o\frac{\partial}{\partial t}\left(\frac{\phi}{B_o}\right)\tag{5-34}$$

For the first term, replace oil saturation by water saturation:

$$\frac{\partial S_w}{\partial t}=-\frac{\partial S_o}{\partial t}\tag{5-35}$$

and use a backward difference in time:

$$\left(\frac{\phi}{B_o}\frac{\partial S_o}{\partial t}\right)_i\approx-\frac{\phi_i}{B_{o\,i}\Delta t_i}\left(S_{w\,i}-S_{w\,i}^{t}\right)\tag{5-36}$$

The second term is identical to the right-hand side of the single-phase equation:

$$\frac{\partial}{\partial t}\left(\frac{\phi}{B_o}\right)\tag{5-37}$$

Chain rule:

$$\frac{\partial}{\partial t}\left(\frac{\phi}{B}\right)=\frac{1}{B}\frac{\partial\phi}{\partial P}+\phi\frac{\partial(1/B)}{\partial t}\tag{5-38}$$

> **[Note]** As printed, (5-38) is missing the $\partial P/\partial t$ factor on the first term.
> The following lines (5-41 and the unnumbered result) restore it:
> $\frac{\partial}{\partial t}\left(\frac{\phi}{B}\right)=\frac{1}{B}\frac{\partial\phi}{\partial P}\frac{\partial P}{\partial t}+\phi\frac{\partial(1/B)}{\partial t}$.

Rock compressibility (porosity dependence on pressure at constant temperature):

$$c_r=\frac{1}{\phi}\frac{d\phi}{dP}\quad\Rightarrow\quad\frac{d\phi}{dP}=\phi c_r\tag{5-39}$$

Fluid model: constant mass density, $\rho=\text{constant}/B$, which implies $B=f(P)$ only:

$$\phi\frac{\partial(1/B)}{\partial t}=\phi\frac{d(1/B)}{dP}\frac{\partial P}{\partial t}\tag{5-40, 5-41}$$

Substituting:

$$\frac{\partial}{\partial t}\left(\frac{\phi}{B}\right)=\frac{1}{B}\frac{\partial\phi}{\partial P}\frac{\partial P}{\partial t}+\phi\frac{d(1/B)}{dP}\frac{\partial P}{\partial t}=\left(\frac{\phi c_r}{B}+\phi\frac{d(1/B)}{dP}\right)\frac{\partial P}{\partial t}$$

Full discretized form:

$$\frac{\partial}{\partial t}\left(\frac{\phi}{B_o}\right)_i\approx\frac{\phi_i}{\Delta t}\left[\frac{c_r}{B_o}+\frac{d(1/B_o)}{dP_o}\right]_i\left(P_{o\,i}-P_{o\,i}^{t}\right)\tag{5-42}$$

$$S_o\frac{\partial}{\partial t}\left(\frac{\phi}{B_o}\right)\approx S_o\frac{\phi_i}{\Delta t}\left[\frac{c_r}{B_o}+\frac{d(1/B_o)}{dP_o}\right]_i\left(P_{o\,i}-P_{o\,i}^{t}\right)\tag{5-43}$$

$$\approx(1-S_w)\frac{\phi_i}{\Delta t}\left[\frac{c_r}{B_o}+\frac{d(1/B_o)}{dP_o}\right]_i\left(P_{o\,i}-P_{o\,i}^{t}\right)\tag{5-44}$$

Right side of the oil equation:

$$\left.\frac{\partial}{\partial t}\left(\frac{\phi S_o}{B_o}\right)\right|_i\approx C_{poo\,i}\left(P_{o\,i}-P_{o\,i}^{t}\right)+C_{swo\,i}\left(S_{w\,i}-S_{w\,i}^{t}\right)\tag{5-45}$$

where

$$C_{poo\,i}=\frac{\phi_i(1-S_{w\,i})}{\Delta t}\left[\frac{c_r}{B_o}+\frac{d(1/B_o)}{dP_o}\right]_i\tag{5-46}$$

$$C_{swo\,i}=-\frac{\phi_i}{B_{o\,i}\Delta t_i}\tag{5-47}$$

For the water equation a similar expansion is carried out:

$$\frac{\partial}{\partial t}\left(\frac{\phi S_w}{B_w}\right)=\frac{\phi}{B_w}\frac{\partial S_w}{\partial t}+S_w\frac{\partial}{\partial t}\left(\frac{\phi}{B_w}\right)\tag{5-48}$$

The second term expands further. Since $P_w=P_o-P_{cow}$:

$$\frac{\partial}{\partial t}\left(\frac{\phi}{B_w}\right)=\frac{\partial}{\partial P_w}\left(\frac{\phi}{B_w}\right)\frac{\partial P_w}{\partial t}=\frac{\partial}{\partial P_w}\left(\frac{\phi}{B_w}\right)\left(\frac{\partial P_o}{\partial t}-\frac{\partial P_{cow}}{\partial t}\right)\tag{5-49}$$

Capillary pressure is a function of water saturation only:

$$\frac{\partial P_{cow}}{\partial t}=\frac{dP_{cow}}{dS_w}\frac{\partial S_w}{\partial t}\tag{5-50}$$

Right side of the water equation:

$$\left.\frac{\partial}{\partial t}\left(\frac{\phi S_w}{B_w}\right)\right|_i\approx C_{pow\,i}\left(P_{o\,i}-P_{o\,i}^{t}\right)+C_{sww\,i}\left(S_{w\,i}-S_{w\,i}^{t}\right)\tag{5-51}$$

where

$$C_{pow\,i}=\frac{\phi_i S_{w\,i}}{\Delta t}\left[\frac{c_r}{B_o}+\frac{d(1/B_w)}{dP_w}\right]_i\tag{5-52}$$

$$C_{sww\,i}=\frac{\phi_i}{B_{w\,i}\Delta t_i}-\left(\frac{dP_{cow}}{dS_w}\right)_i C_{pow\,i}\tag{5-53}$$

> **[Note]** In (5-52) the original prints $c_r/B_o$ inside the water bracket. By analogy with (5-46) this
> is probably meant to be $c_r/B_w$. Check against the FORTRAN storage-coefficient block
> (`DO 28`, ~line 464) before porting.

The terms $C_{poo}$, $C_{swo}$, $C_{pow}$, $C_{sww}$ are called the oil and water **storage coefficients**.

### Discrete forms of the equations

**Oil phase:**

$$\begin{aligned}
&T_{xo\,i+\frac12,j,k}\left[\left(P_{o\,i+1,j,k}-P_{o\,i,j,k}\right)-0.433\gamma_o\left(D_{i+1,j,k}-D_{i,j,k}\right)\right]\\
+&T_{xo\,i-\frac12,j,k}\left[\left(P_{o\,i-1,j,k}-P_{o\,i,j,k}\right)-0.433\gamma_o\left(D_{i-1,j,k}-D_{i,j,k}\right)\right]\\
+&T_{yo\,i,j+\frac12,k}\left[\left(P_{o\,i,j+1,k}-P_{o\,i,j,k}\right)-0.433\gamma_o\left(D_{i,j+1,k}-D_{i,j,k}\right)\right]\\
+&T_{yo\,i,j-\frac12,k}\left[\left(P_{o\,i,j-1,k}-P_{o\,i,j,k}\right)-0.433\gamma_o\left(D_{i,j-1,k}-D_{i,j,k}\right)\right]\\
+&T_{zo\,i,j,k+\frac12}\left[\left(P_{o\,i,j,k+1}-P_{o\,i,j,k}\right)-0.433\gamma_o\left(D_{i,j,k+1}-D_{i,j,k}\right)\right]\\
+&T_{zo\,i,j,k-\frac12}\left[\left(P_{o\,i,j,k-1}-P_{o\,i,j,k}\right)-0.433\gamma_o\left(D_{i,j,k-1}-D_{i,j,k}\right)\right]\\
=&\,C_{poo\,i,j,k}\left(P_{o\,i,j,k}-P^{t}_{o\,i,j,k}\right)+C_{swo\,i,j,k}\left(S_{w\,i,j,k}-S^{t}_{w\,i,j,k}\right)
\end{aligned}\tag{5-54}$$

**Water phase** (note the added capillary-pressure difference in every flow term):

$$\begin{aligned}
&T_{xw\,i+\frac12,j,k}\left[\left(P_{w\,i+1,j,k}-P_{w\,i,j,k}\right)-\left(P_{cow\,i+1,j,k}-P_{cow\,i,j,k}\right)-0.433\gamma_w\left(D_{i+1,j,k}-D_{i,j,k}\right)\right]\\
+&T_{xw\,i-\frac12,j,k}\left[\left(P_{w\,i-1,j,k}-P_{w\,i,j,k}\right)-\left(P_{cow\,i-1,j,k}-P_{cow\,i,j,k}\right)-0.433\gamma_w\left(D_{i-1,j,k}-D_{i,j,k}\right)\right]\\
+&T_{yw\,i,j+\frac12,k}\left[\left(P_{w\,i,j+1,k}-P_{w\,i,j,k}\right)-\left(P_{cow\,i,j+1,k}-P_{cow\,i,j,k}\right)-0.433\gamma_w\left(D_{i,j+1,k}-D_{i,j,k}\right)\right]\\
+&T_{yw\,i,j-\frac12,k}\left[\left(P_{w\,i,j-1,k}-P_{w\,i,j,k}\right)-\left(P_{cow\,i,j-1,k}-P_{cow\,i,j,k}\right)-0.433\gamma_w\left(D_{i,j-1,k}-D_{i,j,k}\right)\right]\\
+&T_{zw\,i,j,k+\frac12}\left[\left(P_{w\,i,j,k+1}-P_{w\,i,j,k}\right)-\left(P_{cow\,i,j,k+1}-P_{cow\,i,j,k}\right)-0.433\gamma_w\left(D_{i,j,k+1}-D_{i,j,k}\right)\right]\\
+&T_{zw\,i,j,k-\frac12}\left[\left(P_{w\,i,j,k-1}-P_{w\,i,j,k}\right)-\left(P_{cow\,i,j,k-1}-P_{cow\,i,j,k}\right)-0.433\gamma_w\left(D_{i,j,k-1}-D_{i,j,k}\right)\right]\\
=&\,C_{pow\,i,j,k}\left(P_{o\,i,j,k}-P^{t}_{o\,i,j,k}\right)+C_{sww\,i,j,k}\left(S_{w\,i,j,k}-S^{t}_{w\,i,j,k}\right)
\end{aligned}\tag{5-55}$$

> **[Note]** The printed form of (5-54) and (5-55) omits the well terms $q'_o$, $q'_w$ that
> appear in (5-56) and (5-57); and the right side of (5-55) prints $P_\alpha$ where $P_o$ is meant.
> Inside the water-phase brackets, $P_w$ differences can be rewritten as
> $P_o$ differences minus $P_{cow}$ differences, which is what the pressure equation below does.

## 5.3 Solution by the IMPES method

In the equations above, oil pressure $P_o$ and water saturation $S_w$ are the primary
unknowns. All coefficients (transmissibilities and storage coefficients) are functions of these
unknowns, and so are the capillary pressures on the left side of the water equation. We therefore
cannot solve the equations before the coefficients are computed, and cannot compute the
coefficients before the solution is known. A method is needed that either iterates on the
solution until convergence or estimates the coefficients another way.

**IMPES** (**IM**plicit **P**ressure, **E**xplicit **S**aturation) is a simple method, still used quite
extensively. Its key idea is to evaluate the coefficients and capillary pressures at the
*old* time level $t$, so that pressures and saturations can be solved without iteration:

$$T_{yo}^{t},\,T_{yw}^{t};\qquad T_{xo}^{t},\,T_{xw}^{t};\qquad C_{swo}^{t},\,C_{sww}^{t};\qquad P_{cow}^{t}$$

(and likewise $T_{zo}^{t}$, $T_{zw}^{t}$, $C_{poo}^{t}$, $C_{pow}^{t}$).

With these approximations the oil equation becomes:

$$\begin{aligned}
&T^{t}_{xo\,i+\frac12,j,k}\left[\left(P_{o\,i+1,j,k}-P_{o\,i,j,k}\right)-0.433\gamma_o\left(D_{i+1,j,k}-D_{i,j,k}\right)\right]\\
+&T^{t}_{xo\,i-\frac12,j,k}\left[\left(P_{o\,i-1,j,k}-P_{o\,i,j,k}\right)-0.433\gamma_o\left(D_{i-1,j,k}-D_{i,j,k}\right)\right]\\
+&T^{t}_{yo\,i,j+\frac12,k}\left[\left(P_{o\,i,j+1,k}-P_{o\,i,j,k}\right)-0.433\gamma_o\left(D_{i,j+1,k}-D_{i,j,k}\right)\right]\\
+&T^{t}_{yo\,i,j-\frac12,k}\left[\left(P_{o\,i,j-1,k}-P_{o\,i,j,k}\right)-0.433\gamma_o\left(D_{i,j-1,k}-D_{i,j,k}\right)\right]\\
+&T^{t}_{zo\,i,j,k+\frac12}\left[\left(P_{o\,i,j,k+1}-P_{o\,i,j,k}\right)-0.433\gamma_o\left(D_{i,j,k+1}-D_{i,j,k}\right)\right]\\
+&T^{t}_{zo\,i,j,k-\frac12}\left[\left(P_{o\,i,j,k-1}-P_{o\,i,j,k}\right)-0.433\gamma_o\left(D_{i,j,k-1}-D_{i,j,k}\right)\right]\\
-&\,q'_{o\,i}=C^{t}_{poo\,i,j,k}\left(P_{o\,i,j,k}-P^{t}_{o\,i,j,k}\right)+C^{t}_{swo\,i,j,k}\left(S_{w\,i,j,k}-S^{t}_{w\,i,j,k}\right)
\end{aligned}\tag{5-56}$$

And the water equation becomes (with $P^{t}_{cow}$ explicit):

$$\begin{aligned}
&T^{t}_{xw\,i+\frac12,j,k}\left[\left(P_{w\,i+1,j,k}-P_{w\,i,j,k}\right)-\left(P_{cow\,i+1,j,k}-P_{cow\,i,j,k}\right)^{t}-0.433\gamma_w\left(D_{i+1,j,k}-D_{i,j,k}\right)\right]\\
+&T^{t}_{xw\,i-\frac12,j,k}\left[\left(P_{w\,i-1,j,k}-P_{w\,i,j,k}\right)-\left(P_{cow\,i-1,j,k}-P_{cow\,i,j,k}\right)^{t}-0.433\gamma_w\left(D_{i-1,j,k}-D_{i,j,k}\right)\right]\\
+&T^{t}_{yw\,i,j+\frac12,k}\left[\left(P_{w\,i,j+1,k}-P_{w\,i,j,k}\right)-\left(P_{cow\,i,j+1,k}-P_{cow\,i,j,k}\right)^{t}-0.433\gamma_w\left(D_{i,j+1,k}-D_{i,j,k}\right)\right]\\
+&T^{t}_{yw\,i,j-\frac12,k}\left[\left(P_{w\,i,j-1,k}-P_{w\,i,j,k}\right)-\left(P_{cow\,i,j-1,k}-P_{cow\,i,j,k}\right)^{t}-0.433\gamma_w\left(D_{i,j-1,k}-D_{i,j,k}\right)\right]\\
+&T^{t}_{zw\,i,j,k+\frac12}\left[\left(P_{w\,i,j,k+1}-P_{w\,i,j,k}\right)-\left(P_{cow\,i,j,k+1}-P_{cow\,i,j,k}\right)^{t}-0.433\gamma_w\left(D_{i,j,k+1}-D_{i,j,k}\right)\right]\\
+&T^{t}_{zw\,i,j,k-\frac12}\left[\left(P_{w\,i,j,k-1}-P_{w\,i,j,k}\right)-\left(P_{cow\,i,j,k-1}-P_{cow\,i,j,k}\right)^{t}-0.433\gamma_w\left(D_{i,j,k-1}-D_{i,j,k}\right)\right]\\
-&\,q'_{w\,i}=C^{t}_{pow\,i,j,k}\left(P_{o\,i,j,k}-P^{t}_{o\,i,j,k}\right)+C^{t}_{sww\,i,j,k}\left(S_{w\,i,j,k}-S^{t}_{w\,i,j,k}\right)
\end{aligned}\tag{5-57}$$

### I. KUSIM pressure solution

Since water saturation appears only as $S_{w\,i}$ on the right-hand sides of the two equations,
they can be combined to eliminate $S_w$ as an unknown. Multiply the water equation by
$\alpha_i$ and add it to the oil equation to get the **pressure equation**:

$$\begin{aligned}
&\left(T^{t}_{xo\,i+\frac12,j,k}+\alpha_i T^{t}_{xw\,i+\frac12,j,k}\right)\left(P_{o\,i+1,j,k}-P_{o\,i,j,k}\right)+\left(T^{t}_{xo\,i-\frac12,j,k}+\alpha_i T^{t}_{xw\,i-\frac12,j,k}\right)\left(P_{o\,i-1,j,k}-P_{o\,i,j,k}\right)\\
+&\left(T^{t}_{yo\,i,j+\frac12,k}+\alpha_i T^{t}_{yw\,i,j+\frac12,k}\right)\left(P_{o\,i,j+1,k}-P_{o\,i,j,k}\right)+\left(T^{t}_{yo\,i,j-\frac12,k}+\alpha_i T^{t}_{yw\,i,j-\frac12,k}\right)\left(P_{o\,i,j-1,k}-P_{o\,i,j,k}\right)\\
+&\left(T^{t}_{zo\,i,j,k+\frac12}+\alpha_i T^{t}_{zw\,i,j,k+\frac12}\right)\left(P_{o\,i,j,k+1}-P_{o\,i,j,k}\right)+\left(T^{t}_{zo\,i,j,k-\frac12}+\alpha_i T^{t}_{zw\,i,j,k-\frac12}\right)\left(P_{o\,i,j,k-1}-P_{o\,i,j,k}\right)\\
-&\,\alpha_i T^{t}_{xw\,i+\frac12,j,k}\left(P_{cow\,i+1,j,k}-P_{cow\,i,j,k}\right)^{t}-\alpha_i T^{t}_{xw\,i-\frac12,j,k}\left(P_{cow\,i-1,j,k}-P_{cow\,i,j,k}\right)\\
-&\,\alpha_i T^{t}_{yw\,i,j+\frac12,k}\left(P_{cow\,i,j+1,k}-P_{cow\,i,j,k}\right)^{t}-\alpha_i T^{t}_{yw\,i,j-\frac12,k}\left(P_{cow\,i,j-1,k}-P_{cow\,i,j,k}\right)\\
-&\,\alpha_i T^{t}_{zw\,i,j,k+\frac12}\left(P_{cow\,i,j,k+1}-P_{cow\,i,j,k}\right)^{t}-\alpha_i T^{t}_{zw\,i,j,k-\frac12}\left(P_{cow\,i,j,k-1}-P_{cow\,i,j,k}\right)\\
-&\,q'_{o\,i}-\alpha_i q'_{w\,i}=\left(C^{t}_{poo\,i}+\alpha_i C^{t}_{pow\,i}\right)\left(P_{o\,i}-P^{t}_{o\,i}\right)
\end{aligned}\tag{5-58}$$

> **[Note]** The printed (5-58) has no gravity terms, and the printed $D$ in (5-63) has none either,
> yet (5-56) and (5-57) include them. The FORTRAN carries gravity arrays (`GRAO`, `GRAW`, ~line 475).
> Treat the gravity contribution as missing from this chapter's pressure equation and take it from the code.

$$\alpha=-\frac{C_{sww\,i,j,k}}{C_{swo\,i,j,k}}\tag{5-59}$$

> **[Note]** As printed. Adding $\alpha\times$(water equation) to the oil equation removes $S_w$ only if
> $\alpha=-C_{swo}/C_{sww}$, and that is what the FORTRAN computes (`ALFA(I)=-CSWO(I)/CSWW(I)`).

The pressure equation can then be written as a seven-point system for each cell, with unknowns
in the cell and its six neighbours:

$$G\,P_{i,j,k-1}+E\,P_{i,j-1,k}+A\,P_{i-1,j,k}+B\,P_{i,j,k}+C\,P_{i+1,j,k}+F\,P_{i,j+1,k}+H\,P_{i,j,k+1}=D_{i,j,k}\tag{5-60}$$

> **[Note]** As printed, (5-60) pairs each coefficient with a neighbour as G: k-1, E: j-1,
> A: i-1, B: centre, C: i+1, F: j+1, H: k+1. But (5-61) below defines $A$ from the $i+\tfrac12$
> transmissibility (and $C$ from $i-\tfrac12$, $E$ from $j+\tfrac12$, $F$ from $j-\tfrac12$,
> $G$ from $k+\tfrac12$, $H$ from $k-\tfrac12$). Those two pages disagree. The physically
> consistent pairing is the coefficient built from the $+\tfrac12$ face multiplying the $+1$ neighbour.

Off-diagonal coefficients, as defined in the original:

$$\begin{aligned}
A&=T^{t}_{xo\,i+\frac12,j,k}+\alpha_i T^{t}_{xw\,i+\frac12,j,k}\\
C&=T^{t}_{xo\,i-\frac12,j,k}+\alpha_i T^{t}_{xw\,i-\frac12,j,k}\\
E&=T^{t}_{yo\,i,j+\frac12,k}+\alpha_i T^{t}_{yw\,i,j+\frac12,k}\\
F&=T^{t}_{yo\,i,j-\frac12,k}+\alpha_i T^{t}_{yw\,i,j-\frac12,k}\\
G&=T^{t}_{zo\,i,j,k+\frac12}+\alpha_i T^{t}_{zw\,i,j,k+\frac12}\\
H&=T^{t}_{zo\,i,j,k-\frac12}+\alpha_i T^{t}_{zw\,i,j,k-\frac12}
\end{aligned}\tag{5-61}$$

> **[Note]** Settled against the FORTRAN (lines 569-690): `A = TOXM + alpha*TWXM` multiplies `P(i-1)` and
> `C = TOXP + alpha*TWXP` multiplies `P(i+1)`; likewise `E`, `F` (y) and `G`, `H` (z) with the minus and plus
> faces. So the pairing printed in (5-60) is right and the definitions in (5-61) have the labels swapped.

### Other coefficients depend on the well boundary conditions

**1. Constant production rate:**

$$B=-\left[\left(T^{t}_{xo\,i+\frac12}+T^{t}_{xo\,i-\frac12}+T^{t}_{yo\,i+\frac12}+T^{t}_{yo\,i-\frac12}+T^{t}_{zo\,k+\frac12}+T^{t}_{zo\,k-\frac12}+C^{t}_{poo\,i}\right)+\alpha_i\left(T^{t}_{xw\,i+\frac12}+T^{t}_{xw\,i-\frac12}+T^{t}_{yw\,j+\frac12}+T^{t}_{yw\,j-\frac12}+T^{t}_{zw\,k+\frac12}+T^{t}_{zw\,k-\frac12}+C^{t}_{pow\,i}\right)\right]\tag{5-62}$$

$$D=\alpha_i\left[\begin{aligned}&T^{t}_{xw\,i+\frac12}\left(P_{cow\,i+1}-P_{cow\,i}\right)^{t}+T^{t}_{xw\,i-\frac12}\left(P_{cow\,i-1}-P_{cow\,i}\right)\\&+T^{t}_{yw\,j+\frac12}\left(P_{cow\,j+1}-P_{cow\,j}\right)^{t}+T^{t}_{yw\,j-\frac12}\left(P_{cow\,j-1}-P_{cow\,j}\right)\\&+T^{t}_{zw\,k+\frac12}\left(P_{cow\,k+1}-P_{cow\,k}\right)^{t}+T^{t}_{zw\,k-\frac12}\left(P_{cow\,k-1}-P_{cow\,k}\right)\end{aligned}\right]-\left(C^{t}_{poo\,i}+\alpha_i C^{t}_{pow\,i}\right)P_{o\,i}^{t}+q'_{o\,i}+\alpha_i q'_{w\,i}\tag{5-63}$$

> **[Note]** The final line of (5-63) prints $-(C_{poo}+\alpha_i P_{cow\,i,j,k})+q'_{oi}+\alpha_i q'_{wi}$. From the
> structure of (5-58) the term should be $-(C^t_{poo}+\alpha_i C^t_{pow})P^t_{o\,i}$, as written above;
> the printed $P_{cow}$ appears to be a typo for $C_{pow}$ and the $P^t_o$ factor was dropped.
> Verify against the FORTRAN `D` array (`DO 30`, ~line 596).

**2. Constant bottom-hole pressure.** For an oil/water producing well with productivity index
$PID_i$ and cell volume $\Delta V_i$:

$$\begin{aligned}
q'_{oi}&=\frac{PID_i}{\Delta V_i}\lambda_{oi}\left(P_{oi}-P_{bhi}\right)=\frac{PID_i}{\Delta V_i}\lambda_{oi}P_{oi}-\frac{PID_i}{\Delta V_i}\lambda_{oi}P_{bhi}\\
q'_{wi}&=\frac{PID_i}{\Delta V_i}\left(\frac{B_{oi}}{B_{wi}}\lambda_{oi}+\lambda_{wi}\right)\left(P_{wi}-P_{bhi}\right)\\
\text{assume }&P_{wi}=P_{oi}:\\
q'_{wi}&=\frac{PID_i}{\Delta V_i}\left(\frac{B_{oi}}{B_{wi}}\lambda_{oi}+\lambda_{wi}\right)\left(P_{oi}-P_{bhi}\right)\\
&=\frac{PID_i}{\Delta V_i}\left(\frac{B_{oi}}{B_{wi}}\lambda_{oi}+\lambda_{wi}\right)P_{oi}-\frac{PID_i}{\Delta V_i}\left(\frac{B_{oi}}{B_{wi}}\lambda_{oi}+\lambda_{wi}\right)P_{bhi}
\end{aligned}\tag{5-64}$$

> **[Note]** The $q'_{wi}$ expression as printed uses the combined mobility
> $\left(\tfrac{B_o}{B_w}\lambda_o+\lambda_w\right)$ rather than just $\lambda_w$. This is the form the
> authors used in the derivation; confirm how the FORTRAN well block (~lines 546-565, 727-790) defines
> water-rate for producers before porting.

Substituting $q'_{oi}$, $q'_{wi}$ into the pressure equation changes coefficients $B$ and $D$:

$$B=-\left[\left(T^{t}_{xo\,i+\frac12}+\dots+T^{t}_{zo\,k-\frac12}+C^{t}_{poo\,i}\right)+\alpha_i\left(T^{t}_{xw\,i+\frac12}+\dots+T^{t}_{zw\,k-\frac12}+C^{t}_{pow\,i}\right)+\frac{PID_i}{\Delta V_i}\lambda_{oi}+\alpha_i\frac{PID_i}{\Delta V_i}\left(\frac{B_{oi}}{B_{wi}}\lambda_{oi}+\lambda_{wi}\right)\right]\tag{5-65}$$

$$D=\alpha_i\left[\sum\text{capillary terms as in (5-63)}\right]-\left(C^{t}_{poo\,i}+\alpha_i C_{pow\,i}\right)P^{t}_{o\,i}-\frac{PID_i}{\Delta V_i}\lambda_{oi}P_{bhi}-\alpha_i\frac{PID_i}{\Delta V_i}\left(\frac{B_{oi}}{B_{wi}}\lambda_{oi}+\lambda_{wi}\right)P_{bhi}\tag{5-66}$$

### II. KUSIM saturation solution

Having obtained the oil pressures, solve for water saturation using either the oil or the water
equation. KUSIM uses the oil equation:

$$\begin{aligned}
&T^{t}_{xo\,i+\frac12,j,k}\left[\left(P_{o\,i+1,j,k}-P_{o\,i,j,k}\right)-0.433\gamma_o\left(D_{i+1,j,k}-D_{i,j,k}\right)\right]+T^{t}_{xo\,i-\frac12,j,k}\left[\left(P_{o\,i-1,j,k}-P_{o\,i,j,k}\right)-0.433\gamma_o\left(D_{i-1,j,k}-D_{i,j,k}\right)\right]\\
+&T^{t}_{yo\,i,j+\frac12,k}\left[\left(P_{o\,i,j+1,k}-P_{o\,i,j,k}\right)-0.433\gamma_o\left(D_{i,j+1,k}-D_{i,j,k}\right)\right]+T^{t}_{yo\,i,j-\frac12,k}\left[\left(P_{o\,i,j-1,k}-P_{o\,i,j,k}\right)-0.433\gamma_o\left(D_{i,j-1,k}-D_{i,j,k}\right)\right]\\
+&T^{t}_{zo\,i,j,k+\frac12}\left[\left(P_{o\,i,j,k+1}-P_{o\,i,j,k}\right)-0.433\gamma_o\left(D_{i,j,k+1}-D_{i,j,k}\right)\right]+T^{t}_{zo\,i,j,k-\frac12}\left[\left(P_{o\,i,j,k-1}-P_{o\,i,j,k}\right)-0.433\gamma_o\left(D_{i,j,k-1}-D_{i,j,k}\right)\right]\\
-&\,q'_{o\,i}=C^{t}_{poo\,i,j,k}\left(P_{o\,i,j,k}-P^{t}_{o\,i,j,k}\right)+C^{t}_{swo\,i,j,k}\left(S_{w\,i,j,k}-S^{t}_{w\,i,j,k}\right)
\end{aligned}\tag{5-67}$$

Since $S_w$ appears as an unknown only in the last term on the right side, solve for it explicitly:

$$S_{w\,i,j,k}=S^{t}_{w\,i,j,k}+\frac{1}{C^{t}_{swo\,i,j,k}}\left[\begin{aligned}&\sum_{\text{6 faces}}T^{t}_{xo/yo/zo}\left[\left(P_{o\,\text{nbr}}-P_{o\,i,j,k}\right)-0.433\gamma_o\left(D_{\text{nbr}}-D_{i,j,k}\right)\right]\\&-q'_{o\,i}-C^{t}_{poo\,i,j,k}\left(P_{o\,i,j,k}-P^{t}_{o\,i,j,k}\right)\end{aligned}\right]\tag{5-68}$$

where the sum runs over the six face neighbours ($i\pm1$, $j\pm1$, $k\pm1$), each with the
corresponding oil transmissibility from (5-16)-(5-21), exactly as in (5-67).

For grid blocks with pressure-specified wells, the well term is replaced as discussed earlier:

$$S_{w\,i,j,k}=S^{t}_{w\,i,j,k}+\frac{1}{C^{t}_{swo\,i,j,k}}\left[\sum_{\text{6 faces}}\cdots-\frac{PID_i}{\Delta V_i}\lambda_{oi}\left(P_{o\,i,j,k}-P_{bh_i}\right)-C^{t}_{poo\,i,j,k}\left(P_{o\,i,j,k}-P^{t}_{o\,i,j,k}\right)\right]\tag{5-69}$$

### Applicability of the IMPES method

Evaluating coefficients at the old time level when solving for the new pressures and saturations
restricts the solution, sometimes severely. The greatest implications are on the saturation-dependent
parameters (relative permeability and capillary pressure), which change rapidly with changing saturation.
IMPES may therefore not be well suited to problems with rapid variations. It is mainly used for
field-scale systems with relatively large grid blocks and slow rates of change, and is normally not
suited to rapid changes close to wells (such as coning studies). Provided time steps are kept small,
IMPES gives accurate and stable solutions to a long range of reservoir problems.

## 5.4 Boundary conditions

The boundary conditions for multiphase flow are as for one-phase flow, but rates and pressures can
be specified for each phase. Normally water is injected into a grid block at constant surface rate or
constant bottom-hole pressure, and oil and water are produced from a grid block at constant
bottom-hole pressure, constant surface oil rate, or constant surface liquid rate. Sometimes a constant
reservoir voidage rate is wanted: injection of water matches a specified liquid-production rate so that
average reservoir pressure stays constant, or liquid production matches a specified water injection rate.

### 5.4.1 Constant water injection rate

The simplest condition, since a water rate term is already in the water equation. For a constant
surface water injection rate $Q_{wi}$ (negative) in grid block $i$:

$$q'_{wi}=\frac{Q_{wi}}{\Delta V_i}\tag{5-70}$$

At the end of the time step the bottom-hole injection pressure can be calculated from the well equation:

$$Q_{wi}=PID_i\,\lambda_{wi}\left(P_{w\,i}-P_{bh_i}\right)\tag{5-71}$$

The well constant PID is defined as for one-phase flow (Peaceman-type equivalent radius):

$$PID_i=\frac{0.00708\,K_{abs}\,h_{net}}{\ln(r_e/r_w)+S}\tag{5-72}$$

$$r_e\approx0.28\,\frac{\left[\left(K_y/K_x\right)^{1/2}\Delta x^2+\left(K_x/K_y\right)^{1/2}\Delta y^2\right]^{1/2}}{\left(K_y/K_x\right)^{1/4}+\left(K_x/K_y\right)^{1/4}}\tag{5-73}$$

The injected fluid meets resistance from the fluids it displaces, so as a better approximation the
sum of the mobilities of the fluids present in the injection block is used. For water injection in
an oil-water system:

$$Q_{wi}=PID_i\left(\frac{B_{oi}}{B_{wi}}\lambda_{oi}+\lambda_{wi}\right)\left(P_{w\,i}-P_{bh_i}\right)\tag{5-74}$$

Injection is controlled by the oil mobility initially (little water in the block), and the water mobility
takes over later. Injection wells are often constrained by a maximum bottom-hole pressure to avoid
fracturing the formation; this should be checked at the end of each time step and, if necessary,
the rate reduced or the well converted to constant bottom-hole pressure.

Frequently capillary pressure is neglected in the well equation (particularly in field-scale simulation):

$$Q_{wi}=PID_i\left(\frac{B_{oi}}{B_{wi}}\lambda_{oi}+\lambda_{wi}\right)\left(P_{o\,i}-P_{bh_i}\right)\tag{5-75}$$

### 5.4.2 Injection at constant bottom-hole pressure

Injection at constant bottom-hole pressure is achieved either with constant pressure at the surface pump
or by letting the hydrostatic pressure of the water-filled well control it. The well equation above is
used again:

$$Q_{wi}=PID_i\left(\frac{B_{oi}}{B_{wi}}\lambda_{oi}+\lambda_{wi}\right)\left(P_{o\,i}-P_{bh_i}\right)\tag{5-76}$$

Since the (unknown) grid-block pressure is in the well equation, the corresponding terms must be
included in the appropriate coefficients of the pressure solution.

### 5.4.3 Constant oil production rate

Handled as for constant water injection rate. For a constant surface oil rate $Q_{oi}$ (positive):

$$q'_{oi}=\frac{Q_{oi}}{\Delta V_i}\tag{5-77}$$

Oil production is generally accompanied by water production, so the water equation has a
water production term:

$$q'_{wi}=q'_{oi}\,\frac{\lambda_{wi}\left(P_{w\,i}-P_{bh_i}\right)}{\lambda_{oi}\left(P_{o\,i}-P_{bh_i}\right)}\tag{5-78}$$

If capillary pressure is neglected around the producer:

$$q'_{wi}=q'_{oi}\,\frac{\lambda_{wi}}{\lambda_{oi}}\tag{5-79}$$

At the end of the time step, the bottom-hole pressure follows from the oil well equation:

$$Q_{oi}=WC_i\,\lambda_{oi}\left(P_{o\,i}-P_{bh_i}\right)\tag{5-80}$$

> **[Note]** The original writes the well constant as $WC_i$ in (5-80) and (5-89) and as $PID_i$
> elsewhere (5-71, 5-88, 5-90). These are the same quantity (eq. 5-72).

Producers are normally constrained by a minimum bottom-hole pressure (for lifting). If it is reached,
the well should be converted to constant bottom-hole pressure. Rates are also often constrained by
a maximum water cut: if exceeded, the highest-water-cut grid block may be shut in, or the production
rate reduced. Surface water cut:

$$f_{ws_i}=\frac{q'_{wi}}{q'_{wi}+q'_{oi}}\tag{5-81}$$

### 5.4.4 Constant liquid production rate

For a total surface liquid production rate $Q_{Li}$ (positive) in grid block $i$:

$$Q_{Li}=Q_{oi}+Q_{wi}\tag{5-82}$$

If capillary pressure is neglected:

$$q'_{oi}=\frac{\dfrac{\lambda_{oi}}{\lambda_{oi}+\lambda_{wi}}\,Q_{Li}}{\Delta V_i}\tag{5-83}$$

$$q'_{wi}=\frac{\dfrac{\lambda_{wi}}{\lambda_{oi}+\lambda_{wi}}\,Q_{Li}}{\Delta V_i}\tag{5-84}$$

### 5.4.5 Production at constant reservoir voidage rate

Illustrated by a constant surface water injection rate $Q_{w_{inj}}$ in some grid block, with the total
liquid production of the producing block matched to the reservoir injection volume so that reservoir
pressure stays approximately constant:

$$Q_{oi}B_{oi}+Q_{wi}B_{wi}=-Q_{w_{inj}}B_{w_{inj}}\tag{5-85}$$

If capillary pressure is neglected:

$$q'_{oi}=\left(\frac{\lambda_{oi}}{\lambda_{oi}B_{oi}+\lambda_{wi}B_{wi}}\right)\left(\frac{-Q_{w_{inj}}B_{w_{inj}}}{\Delta V_i}\right)\tag{5-86}$$

$$q'_{wi}=\left(\frac{\lambda_{wi}}{\lambda_{oi}B_{oi}+\lambda_{wi}B_{wi}}\right)\left(\frac{-Q_{w_{inj}}B_{w_{inj}}}{\Delta V_i}\right)\tag{5-87}$$

### 5.4.6 Production at constant bottom-hole pressure

For a production well in grid block $i$ at constant bottom-hole pressure $p_{bh_i}$:

$$Q_{oi}=PID_i\,\lambda_{oi}\left(P_{o\,i}-P_{bh_i}\right)\tag{5-88}$$

$$Q_{wi}=WC_i\,\lambda_{wi}\left(P_{w\,i}-P_{bh_i}\right)\tag{5-89}$$

Substituting into the flow terms in the flow equations:

$$q'_{oi}=\frac{PID_i}{\Delta V_i}\lambda_{oi}\left(P_{o\,i}-P_{bh_i}\right)\tag{5-90}$$

$$q'_{wi}=\frac{PID_i}{\Delta V_i}\lambda_{wi}\left(P_{w\,i}-P_{bh_i}\right)\tag{5-91}$$

Since the rate terms contain unknown block pressures, they must be included in the matrix coefficients
when solving for pressures. At the end of each time step the actual rates are computed from
these equations and water cut is computed as before.

> **[Note]** Compare (5-91) with the derivation in (5-64) (previous section), which used
> $\left(\tfrac{B_o}{B_w}\lambda_o+\lambda_w\right)$ for the water rate of a producer. Pages 96-97 and
> page 105 are inconsistent. The FORTRAN decides which applies.

## 5.5 KUSIM matrix solver

Once the coefficients are calculated, a set of linear equations is formulated and the matrix shape is
determined. Solution methods vary in the number of operations needed and so in solve time. In all
3-D models such as KUSIM, the matrix has 7 non-zero diagonals (heptadiagonal). The matrix form is
affected by the ordering of block numbers, and the simplicity of some solution methods depends on that form.

KUSIM uses Gauss elimination to solve these heptadiagonal matrices. Gauss elimination does not
depend on the matrix shape since it deals with all the coefficients of the matrix. Ordering
scheme therefore becomes meaningless here, and only one ordering system is considered in KUSIM.

### General review of Gauss elimination

For three equations:

$$a_{11}x_1+a_{12}x_2+a_{13}x_3=d_1\tag{1}$$
$$a_{21}x_1+a_{22}x_2+a_{23}x_3=d_2\tag{2}$$
$$a_{31}x_1+a_{32}x_2+a_{33}x_3=d_3\tag{3}$$

Multiply (1) by $-a_{21}/a_{11}$ and add to (2); then multiply (1) by $-a_{31}/a_{11}$ and add to (3):

$$a'_{22}x_2+a'_{23}x_3=d'_2\tag{5}\qquad\qquad a'_{32}x_2+a'_{33}x_3=d'_3\tag{6}$$

Next multiply (5) by $-a'_{32}/a'_{22}$ and add to (6), giving the triangular system

$$a_{11}x_1+a_{12}x_2+a_{13}x_3=d_1\tag{7}$$
$$a'_{22}x_2+a'_{23}x_3=d'_2\tag{8}$$
$$a''_{33}x_3=d''_3\tag{9}$$

This is **forward elimination**. Equation (9) gives $x_3$ directly, then **backward substitution** follows:

$$x_3=\frac{d''_3}{a''_{33}}\tag{10}$$
$$x_2=\frac{d'_2-a'_{23}x_3}{a'_{22}}\tag{11}$$
$$x_1=\frac{d_1-a_{12}x_2-a_{13}x_3}{a_{11}}\tag{12}$$

**In general**, for $n$ equations and $n$ unknowns:

1. Forward elimination:
$$a_{ij}=a_{ij}+a_{kj}\left(-\frac{a_{ik}}{a_{kk}}\right),\qquad i=k+1,n;\ j=k+1,n;\ k=1,n-1$$
$$d_i=d_i+d_k\left(-\frac{a_{ik}}{a_{kk}}\right),\qquad k=1,n-1;\ i=k+1,n$$

2. Backward substitution:
$$x_i=\frac{1}{a_{ii}}\left(d_i-\sum_{j=i+1}^{n}a_{ij}x_j\right),\qquad i=n,\dots,1$$

> **[Note]** This is the loop structure in the FORTRAN (`DO 50 K=1,U-1`, `DO 60 I=K+1,U`,
> `DO 70 J=K,U`, then `DO 80 I=U-1,1,-1`, lines ~699-718). The original solver is dense, with
> no pivoting and no use of the band structure, which is why the plan replaces it with a sparse solver.

### Three-dimensional grid and matrix structure

(Original figure on p. 108: a 3-D block grid with indices $i=1..N_x$, $j=1..N_y$, $k=1..N_z$,
showing neighbours $(i,j,k\pm1)$, $(i\pm1,j,k)$ and the upwind rule for $\lambda_w$.)

The linear system for pressures has seven non-zero diagonals on the left side:

$$g_{i,j,k}P_{i,j,k-1}+e_{i,j,k}P_{i,j-1,k}+a_{i,j,k}P_{i-1,j,k}+b_{i,j,k}P_{i,j,k}+c_{i,j,k}P_{i+1,j,k}+f_{i,j,k}P_{i,j+1,k}+h_{i,j,k}P_{i,j,k+1}=d_{i,j,k}$$

with $i=1,N_x$; $j=1,N_y$; $k=1,N_z$.

(Original figure on p. 109: a 12-row sparse matrix with $b$ on the diagonal, $a$ and $c$ immediately
below and above it, $e$ and $f$ at an offset of one row (plane of cells) on either side, and $g$ and $h$
at an offset of one layer.)

> **[Note]** Here the left-neighbour coefficient is $a$ (multiplies $P_{i-1}$) and the right-neighbour
> coefficient is $c$ (multiplies $P_{i+1}$), consistent with the matrix figure, with $e\to j-1$, $f\to j+1$,
> $g\to k-1$ and $h\to k+1$. This is the printed ordering in (5-60). It conflicts with how
> (5-61) defines $A$ and $C$; see the note under (5-61).

The bandwidth of the 3-D system, numbered along the x and y directions first:

$$N_b=2N_xN_z+1\tag{5-92}$$

> **[Note]** For x-then-y numbering, the plane size is $N_xN_y$, so the half-bandwidth is $N_xN_y$ and
> the bandwidth is $2N_xN_y+1$. The printed $N_xN_z$ appears to be a typo. The FORTRAN
> uses `W = NX*NY` and `2*W` (lines ~59, 685).

To reduce the bandwidth (and thus the work) one should number the smallest plane first. The number
of operations for direct solution is approximately

$$\text{no. of operations}\approx4N_x^3N_y^3N_z^3\tag{5-93}$$

For a small 1000-block system with $N_x=N_y=N_z=10$ this is on the order of 40 million operations per solve
(the original's figure; dense elimination is $O(n^3)$ with $n=N_xN_yN_z$), so direct solution is normally
limited to small systems, and iterative methods are needed for large ones.

## 5.6 KUSIM main features

- The maximum number of blocks is initially set to about 9999, to be increased in later versions.
  (The FORTRAN as provided is dimensioned for 4000 blocks.)
- Handles both regular and irregular gridding.
- Only one PVT region table is used.
- Only the IMPES solution method is considered: block pressure is calculated implicitly, then
  saturation explicitly, at the end of each time step.
- Designed to deal effectively with complex reservoir shape.
- Restricted to only one type of boundary condition.
- Main sections of the code:
  1. Definition of variables
  2. Reading the user's input data file (`input.dat`)
  3. Generalizing the input data
  4. Initialization
  5. Time loop
  6. Calculating pressure-dependent properties at each grid block
  7. Averaging properties
  8. Transmissibilities calculation
  9. Calculation of gravity effect terms
  10. Storage coefficients calculation
  11. Matrix coefficients determination (including source/sink boundary conditions)
  12. Matrix solution
  13. Pressure update
  14. Saturation solution
  15. Interpolation subroutine

### 1. Definition of variables

All variables used in the program are identified. There are two types: **real** variables (most of the
model's variables, some of them arrays) and **integer** variables (mainly used for loops).

### 2. Reading the user's input data file

`input.dat` contains all data required to run the program, in these categories:

1. geological model data (form, dimensions, number of blocks)
2. rock properties of each block
3. time step and run termination (end time)
4. PVT properties
5. saturation vs. relative permeability and saturation vs. capillary pressure
6. boundary conditions

### 3. Generalizing the input data

Some data such as $D_x$, $D_y$, $D_z$ must be generalized to describe each block.

### 4. Initialization

There are two ways to initialize the model: equilibrium initialization and gravity-segregation
initialization. KUSIM uses **equilibrium initialization**.

First obtain the capillary pressure in each block:

$$P_{cow}=0.433\,(\gamma_w-\gamma_o)\,(WOC-EL)\tag{5-94}$$

$$EL=DL+DDH\tag{5-95}$$

- $WOC$: water-oil contact depth, ft
- $EL$: elevation of each grid block from the reference datum, ft
- $DL$: reference datum depth or datum level (normally the depth to the top of the structure), ft
- $DDH$: elevation of each block from datum (at the mid-point centre)
- $\gamma_w$, $\gamma_o$: water and oil specific gravity

The equation becomes:

$$P_{cow}=0.433\,(\gamma_w-\gamma_o)\,(WOC-DL-DDH)\tag{5-96}$$

The initial water saturation $S_{wi}$ is then calculated for each block at its mid-point elevation
using the $S_w$ vs. $P_{cow}$ data from `input.dat`:

1. If $P_{cow}\ge P_{cow}(S_w=S_{wir})$ (irreducible water saturation), set $S_{wi}=S_{wir}$.
2. If $P_{cow}\le P_{cow}(S_w=1)$, set $S_{wi}=1$.
3. If $P_{cow}(S_w=S_{wir})<P_{cow}<P_{cow}(S_w=1)$, interpolate the value of $S_{wi}$.

Oil saturation follows from $S_o+S_w=1$.

For pressure, one input is the pressure at datum level, and each block gets its initial pressure
using the elevation $DDH$:

$$P_o=P_{DATUM}+0.433\,\gamma_o\,(DDH)\tag{5-97}$$

> **[Note]** The printed (5-97) is $P_o=P_{DATUM}+0.433\,\gamma_o$; the factor $DDH$ is
> implied by the text ("using the elevation DDH") and is not printed. The sign convention of $DDH$
> (positive downward or upward from datum) should be confirmed in the FORTRAN initialization
> block (`DO 12`-`DO 700`, lines ~225-256).

### 5. Time loop

Constrained by the beginning time (usually zero), the time step and the end time. The time step
affects accuracy: the smaller it is, the more accurate the results and the higher the cost, so some
optimization is needed to choose a suitable time step.

### 6. Calculating pressure-dependent properties at each grid block

Each time step begins with newly updated block pressures, so the interpolation subroutine is called
each step to find the PVT properties at those pressures.

### 7. Averaging properties

There are two transmissibilities in each dimension (minus and plus direction). The properties of the
two blocks they connect have to be averaged. KUSIM uses upstream mobility weighting to calculate the
average properties for each transmissibility.

### 8. Calculation of transmissibilities

A transmissibility has three factors: (8.1) block dimensions constant, (8.2) the mobility term, a function of
pressure, and (8.3) weighted-average permeability. Multiplying the three gives the transmissibility.

### 9. Calculation of gravity effect terms

Gravity has significant effects on the final results, especially in 3-D models, 1-D vertical column
models and 2-D cross-sectional models. Gravity terms are calculated for each phase and each
dimension in the plus and minus directions, in the same manner as transmissibility.

### 10. Storage coefficients calculation

The right-hand side of the flow equations has four storage coefficients in KUSIM. This section also
calculates $\alpha$ (alfa), the multiplication factor.

### 11. Matrix coefficients determination

These consist mainly of transmissibilities, gravities and storage coefficients. All types of boundary
conditions are dealt with in this section.

### 12. Matrix solution

The matrix is always heptadiagonal and is normally solved by special solvers that account for
the internal zero coefficients, which reduces simulation run time. Since KUSIM only orders in the
x direction, starting from the left side of the upper layer, such solvers are not needed and KUSIM
uses Gaussian elimination only.

### 13. Pressure update

After the matrix solve, the new pressures become the old ones in the next time step: $P=P_{new}$.

### 14. Saturation solution

With the new pressures, KUSIM substitutes into the oil equation to solve for saturation explicitly (5-68).

### 15. Interpolation subroutine

KUSIM uses linear interpolation, with three cases:

- If the argument is less than the first table value, the result is the first corresponding value.
- If the argument is greater than the last table value, the result is the last corresponding value.
- Otherwise, linear interpolation between the bracketing table rows.

> **[Note]** This matches the flat extrapolation in `INTERP` (`3DTP(11).FOR`, line 811 onward). The code
> adds two things the text does not mention:
> - When `ISW` is non-zero it also returns the local slope `DY` (the derivative), using the slope of the
>   first or last table segment beyond the ends. This is how $d(1/B)/dP$ and $dP_{cow}/dS_w$ are obtained.
> - `AA=1` handles tables with increasing $x$ (PVT and relative-permeability lookups), and `AA=0`
>   handles tables with decreasing $x$. The `AA=0` form is used at initialization to invert the
>   capillary-pressure table (find $S_w$ from $P_{cow}$, since $P_{cow}$ falls as $S_w$ rises).
> The tables are dimensioned to 100 rows.

## 5.7 Treatment of wells in KUSIM

KUSIM handles these conditions:

1. Constant oil production rate, with/without constant water injection rate.
2. Constant liquid (oil + water) production rate, with/without constant water injection rate.
3. Constant production bottom-hole pressure, with/without constant water injection rate.
4. Constant oil production rate, with/without constant injection bottom-hole pressure.
5. Constant liquid (oil + water) production rate, with/without constant injection bottom-hole pressure.
6. Constant production bottom-hole pressure, with/without constant injection bottom-hole pressure.

A minimum bottom-hole pressure can also be specified. If the first boundary condition is used, the
bottom-hole pressure is calculated with the well model equations above, and if it falls below the minimum
bottom-hole pressure specified in `input.dat`, the condition changes automatically to condition 3.

`INPUT.DAT` is made so the user only has to specify the number of producing and/or injection wells,
the block each one represents, and the production rate or bottom-hole pressure according to the type
of condition needed.

## 5.8 KUSIM algorithm (flowchart, original pp. 115-118)

1. **Start.** Read INPUT.
2. Generalize DX, DY.
3. Calculate drainage radius and well constant.
4. **Begin time loop**, $t=0$: initialize the model (calculate DDH, calculate PCOW, interpolate to find
   $S_{wi}$, calculate initial pressure).
5. Interpolate to find $k_{ro}$, $k_{rw}$, $B_o$, $B_w$, $\mu_o$, $\mu_w$ (**I**).
6. Calculate $\lambda_o$, $\lambda_w$.
7. Calculate transmissibility; average absolute permeabilities; average mobilities (upstream mobility);
   modify transmissibility constant, permeability and averaged values; calculate transmissibilities.
8. Calculate gravities. Calculate storage coefficients (**III**).
9. No-flow boundary modification of transmissibilities and gravities.
10. Prepare the well boundary condition requirements; calculate the well term $q_o$.
11. Calculate matrix coefficients $G, E, A, C, F, H$; calculate the other (boundary) matrix coefficient; formulate the matrix.
12. Solve the matrix (Gaussian elimination) (**IV**).
13. Calculate wells' bottom-hole pressures and rates. Solve for saturation.
14. Print pressure, saturation, flow rates, bottom-hole pressures and water cut for each time step.
15. If time < max time: $T=T+\Delta t$ and return to step 5 (**II**); otherwise **End**.

> **[Note]** This is the exact sequence to follow for the IMPES loop in the Python port. In the original
> flowchart the box "Calculate Wells Bottom hole pressures and" is cut off; the output step lists rates
> and water cut.

## 5.9 INPUT.DAT manual

| Variable | Meaning |
|---|---|
| `NX`, `NY`, `NZ` | Number of blocks in X, Y, Z direction |
| `DX`, `DY`, `DZ` | Block dimensions in X, Y, Z |
| `KX`, `KY`, `KZ` | Absolute permeability in X, Y, Z |
| `PHI` | Porosity |
| `RDATUM` | Reference datum |
| `WOCD` | Distance of WOC from datum |
| `DPRESS` | Datum pressure |
| `DDH` | Mid-point height difference to datum |
| `GAMAO`, `GAMAW` | Oil and water specific gravity |
| `SWC` | Connate water saturation |
| `SOR` | Residual oil saturation |
| `CR` | Compressibility (rock) |
| `DT` | Time step |
| `TMAX` | Maximum time |
| `NPVT` | Number of PVT data rows |
| `NSAT` | Number of saturation data rows |
| `NOPWS` | Number of production wells |
| `PWL` | Production wells location (block number) |
| `PRW` | Production well wellbore radius |
| `PS` | Production well skin factor |
| `MPBHP` | Minimum production bottom-hole pressure |
| `PBC` | Production boundary condition |
| `QO` | Oil production rate |
| `QL` | Liquid production rate |
| `PBHP` | Production bottom-hole pressure |
| `NOIWS` | Number of injection wells |
| `IWL` | Injection wells location (block number) |
| `IRW` | Injection wells wellbore radius |
| `IS` | Injection wells skin factor |
| `IBC` | Injection boundary condition |
| `QW` | Water injection rate |
| `IBHP` | Injection bottom-hole pressure |

> **[Note]** In `INPUT.DAT` the line order is as read by `3DTP(11).FOR` lines 54-121 (see `CLAUDE.md`).
> This table lists the variables but not units or the numeric codes for `PBC` and `IBC`. The code reads
> `IF (PBC.EQ.1)` and `IF (IBC.EQ.1.AND.PBC.EQ.1)`; document those conventions from the FORTRAN.
> Field units apply throughout (ft, psi, md, bbl/D, cp).

## Transcription issues to resolve before relying on this chapter

Collected from the notes above. These are typos or inconsistencies in the printed chapter:

| Where | Issue |
|---|---|
| (5-10) | second bracket uses $\gamma_w$ for the oil phase |
| (5-11), (5-14) | $k_z$ printed for $k_y$; second bracket uses $D_{j+1}$ instead of $D_{j-1}$ |
| p. 84 | "Upstream" / "Weighted average" labels swapped |
| (5-38) | missing $\partial P/\partial t$ |
| (5-52) | $c_r/B_o$ inside the water bracket (probably $B_w$) |
| (5-54), (5-55) | well terms omitted; $P_\alpha$ printed for $P_o$ |
| (5-58) | no gravity terms (they are in 5-56, 5-57) |
| (5-59) | $\alpha$ printed inverted; the code (and the derivation) has $\alpha=-C_{swo}/C_{sww}$ |
| (5-60) vs (5-61) | neighbour pairing of $A, C, E, F, G, H$ inconsistent; the code pairs the minus-face coefficient with the $i-1$ (or $j-1$, $k-1$) unknown (see `docs/porting-notes.md`) |
| (5-63) | last line prints $P_{cow}$ where $C_{pow}\,P_o^t$ is meant |
| (5-64) vs (5-91) | producer water rate: combined mobility vs $\lambda_w$ only |
| (5-80), (5-89) | $WC_i$ used for the well constant ($PID_i$) |
| (5-92) | bandwidth printed with $N_z$; should involve $N_xN_y$ |
| (5-97) | $DDH$ factor missing |

