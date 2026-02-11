# q1: “What does the inside of the proton look like?”

Great question with layers of depth depending on the resolution scale (virtuality $Q^2$) you probe at:

**Static picture (low $Q^2$):** Three valence quarks (uud) bound by gluon fields. The quarks carry only ~30% of the proton's momentum — the rest is gluons and sea quark-antiquark pairs.

**DIS picture (moderate $Q^2$):** PDFs (Parton Distribution Functions) reveal a rich sea. Gluon density dominates at low Bjorken-$x$ and rises steeply — the proton becomes a "gluon machine." Valence quarks peak around $x \sim 0.2$.

**High-resolution/saturation picture (very low $x$, high density):** Gluon occupation numbers grow until nonlinear recombination kicks in → CGC (Color Glass Condensate) framework. The proton looks like a dense, coherent gluon wall characterized by a saturation scale $Q_s(x)$.

**3D imaging (modern frontier):** GPDs (Generalized Parton Distributions) and TMDs (Transverse Momentum Dependent distributions) map out spatial and momentum tomography. Results from HERA, JLab 12 GeV, and soon the EIC will sharpen this picture dramatically.

**Emergent mass:** Most of the proton's ~938 MeV mass isn't from quark masses (~10 MeV total) but from QCD dynamics — gluon field energy and chiral symmetry breaking.

Key references:
- R. Abdul Khalek et al., *"Science Requirements and Detector Concepts for the EIC,"* Nucl. Phys. A **1026**, 122447 (2022)
- H. Mäntysaari, *"Review of proton and nuclear shape fluctuations at high energy,"* Rep. Prog. Phys. **83**, 082201 (2020)
- C.D. Roberts, *"Empirical consequences of emergent mass,"* Symmetry **12**, 1468 (2020)

The EIC is essentially purpose-built to answer your question with unprecedented precision — you'll be well-positioned to see that unfold.

# q2: “Please tell me more about momentum space imaging”

## Momentum-Space Imaging: TMDs

The key framework here is **TMD factorization**, which extends the collinear PDF picture by retaining the parton's transverse momentum $\mathbf{k}_T$ relative to the parent hadron's direction.

**What TMDs encode:** A TMD $f(x, \mathbf{k}_T; Q^2)$ gives the probability density of finding a parton with longitudinal momentum fraction $x$ *and* intrinsic transverse momentum $\mathbf{k}_T$. At leading twist for spin-½ hadrons, there are **8 independent TMDs** (per flavor), compared to just 3 collinear PDFs. The additional ones arise from correlations between parton spin, hadron spin, and $\mathbf{k}_T$ direction.

**The key TMDs:**

- **$f_1(x, k_T)$** — unpolarized TMD (3D generalization of the standard PDF)
- **Sivers function $f_{1T}^\perp$** — correlates unpolarized quark $k_T$ with proton transverse spin; encodes orbital angular momentum (OAM) effects; naively T-odd, with a predicted **sign change** between SIDIS (Semi-Inclusive Deep Inelastic Scattering) and Drell-Yan — a fundamental test of QCD gauge structure
- **Boer-Mulders function $h_1^\perp$** — correlates quark transverse spin with $k_T$ in an unpolarized hadron
- **Pretzelosity $h_{1T}^\perp$** — sensitive to relativistic effects and quark OAM
- **Worm-gear functions ($g_{1T}, h_{1L}^\perp$)** — mix longitudinal and transverse polarizations

**How you measure them:** SIDIS ($\ell p \to \ell' h X$), Drell-Yan ($pp \to \ell^+\ell^- X$), and $e^+e^-$ annihilation. You extract azimuthal modulations (e.g., $\cos\phi_h$, $\sin(\phi_h - \phi_S)$) in the cross section — each modulation maps to a specific TMD convolution.

**TMD evolution** is governed by the Collins-Soper (CS) equation and the rapidity anomalous dimension (Collins-Soper kernel $K(b_T; \mu)$), which is nonperturbative at large $b_T$ (conjugate to $k_T$). Lattice QCD has made significant progress computing this directly — see the LPC collaboration results.

**Current status:**
- HERMES, COMPASS, and JLab 6/12 GeV have provided first extractions, especially of Sivers and Collins asymmetries
- Global TMD fits (MAP, Pavia, JAM collaborations) now extract unpolarized TMDs and fragmentation functions with reasonable precision
- The Sivers sign change remains **not yet definitively confirmed** — COMPASS Drell-Yan data show hints but statistics are limited

**EIC impact:** This is arguably *the* flagship TMD measurement program. Broad $(x, Q^2, z, P_{hT})$ coverage with polarized beams will enable flavor-separated, precision TMD extractions and nail down the Sivers sign change.

Key references:
- R. Angeles-Martinez et al., *"Transverse momentum dependent (TMD) parton distribution functions: status and prospects,"* Acta Phys. Polon. B **46**, 2501 (2015) — comprehensive review
- I. Scimemi & A. Vladimirov, *"Non-perturbative structure of semi-inclusive deep-inelastic and Drell-Yan cross sections,"* JHEP **06**, 137 (2020) — modern TMD evolution/fitting
- M. Bury, A. Prokudin, A. Vladimirov, *"Extraction of the Sivers function from SIDIS, Drell-Yan, and W±/Z data,"* JHEP **05**, 151 (2021)

Want me to go deeper into the Sivers sign-change physics, TMD evolution formalism, or the connection to OAM/proton spin puzzle?

# q3: "go deeper into the Sivers sign-change physics"

## The Sivers Sign Change: A Fundamental Test of QCD Gauge Structure

### The Sivers Function & Its Origin

The Sivers function $f_{1T}^\perp(x, k_T)$ describes a left-right asymmetry in the distribution of unpolarized quarks inside a transversely polarized proton — schematically, a correlation $\mathbf{k}_T \cdot (\mathbf{P} \times \mathbf{S}_T)$.

Naively, this should vanish by time-reversal invariance. In 1993, Collins proved it's zero for any gauge-invariant, T-even distribution. The resolution came from understanding that **TMDs are not universal in the standard PDF sense** — they depend on the gauge link (Wilson line) structure, which differs process by process.

### Gauge Links: The Key Mechanism

In a TMD definition, you need a Wilson line to make the operator gauge-invariant:

$$f_{1T}^{\perp}(x, k_T) \sim \langle P, S_T | \bar{\psi}(0) \, \mathcal{U}[0 \to \xi] \, \psi(\xi) | P, S_T \rangle$$

The path $\mathcal{U}$ of the Wilson line is dictated by the **color flow of the hard process**:

- **SIDIS** ($\ell p \to \ell' h X$): The struck quark propagates into the final state. The gauge link goes to $+\infty$ along the lightcone → **future-pointing** Wilson line $\mathcal{U}^{[+]}$
- **Drell-Yan** ($p p \to \ell^+\ell^- X$): The annihilating quark comes from the initial state. The gauge link goes to $-\infty$ → **past-pointing** Wilson line $\mathcal{U}^{[-]}$

Under time reversal, $\mathcal{U}^{[+]} \leftrightarrow \mathcal{U}^{[-]}$, and the Sivers function, being T-odd, picks up a sign:

$$\boxed{f_{1T}^{\perp\,[\text{SIDIS}]} = - f_{1T}^{\perp\,[\text{DY}]}}$$

This is the **prediction of Collins (2002)** and is sometimes called "modified universality." It's not a perturbative result — it's a direct consequence of non-abelian gauge invariance and factorization. In QED, there would be no sign change.

### Physical Intuition

Think of it as initial-state vs. final-state interactions (ISI/FSI):

- In SIDIS, the active quark undergoes FSI (gluon exchange with the remnant *after* the hard scattering) → generates a phase that produces the asymmetry
- In DY, the active quark undergoes ISI (gluon exchange *before* annihilation) → same magnitude of phase, opposite sign

These are the leading, single-gluon-exchange contributions to the gauge link. The sign flip is a direct manifestation of the **non-abelian color Coulomb phase** being process-dependent.

### Why It Matters So Much

If the sign change is confirmed → **validates TMD factorization** at a foundational level and our understanding of color flow in QCD.

If it's violated → either TMD factorization breaks down, or there's new physics in soft gluon dynamics. This would be a crisis-level result for pQCD.

It's also connected to the **generalized universality** of TMDs. For more complex processes (e.g., dijet production in $pp$, back-to-back hadrons), the gauge link structure becomes more complicated (staple-like, looping paths), and factorization may break entirely — see the work of Rogers & Mulders on factorization-breaking effects in hadron-hadron collisions.

### Experimental Status

| Experiment | Process | Observable | Status |
|---|---|---|---|
| HERMES, COMPASS, JLab | SIDIS | $\sin(\phi_h - \phi_S)$ modulation | Well-established, nonzero Sivers asymmetry for $u$-quarks (~5-10%), small for $d$ |
| COMPASS (2015, 2017) | DY ($\pi^- p^\uparrow$) | $\sin\phi_S$ modulation | Sign change **consistent with prediction** but $\lesssim 2\sigma$ significance |
| RHIC/STAR | $W^{\pm}/Z$ production | Single-spin asymmetry $A_N$ | Sensitive to Sivers but convoluted with sea quark contributions; results consistent with sign change |
| FNAL SpinQuest (E1039) | DY ($p p^\uparrow$) | $\sin\phi_S$ | **Currently running** — aims for definitive measurement with polarized NH₃ target |
| EIC | SIDIS + DY-like | Full TMD program | Will provide ultimate precision, flavor separation, and $Q^2$ lever arm |

The honest summary: **the sign change is supported but not yet definitively proven.** The COMPASS DY measurement is the most direct test to date, and it favors the prediction, but the statistical power isn't there yet for a discovery-level claim.

### Complications

- **TMD evolution effects:** The SIDIS and DY measurements are at different $Q^2$ scales, so you can't just compare raw asymmetries — you must evolve via the CS kernel, introducing model dependence at large $b_T$
- **Process dependence beyond sign:** Higher-order gauge link topologies can introduce corrections beyond the simple sign flip
- **Factorization breaking:** For processes like $pp \to \text{dijet}$, TMD factorization itself may fail (Bomhof, Mulders, Vogelsang; Rogers, Mulders 2010), complicating extraction from hadronic collisions

### Key References

- J.C. Collins, *"Leading-twist single-transverse-spin asymmetries: Drell-Yan and deep-inelastic scattering,"* Phys. Lett. B **536**, 43 (2002) — the sign-change prediction
- S.J. Brodsky, D.S. Hwang, I. Schmidt, *"Final-state interactions and single-spin asymmetries in semi-inclusive deep inelastic scattering,"* Phys. Lett. B **530**, 99 (2002) — FSI mechanism/physical picture
- COMPASS Collaboration, *"First measurement of transverse-spin-dependent azimuthal asymmetries in the Drell-Yan process,"* Phys. Rev. Lett. **119**, 112002 (2017)
- T.C. Rogers & P.J. Mulders, *"No generalized TMD-factorization in dihadron production in hadron-hadron or heavy ion collisions,"* Phys. Rev. D **81**, 094006 (2010)
- SpinQuest/E1039: [https://spinquest.fnal.gov](https://spinquest.fnal.gov)

Want me to dig into the factorization-breaking aspects in hadronic collisions, or the connection between the Sivers function and quark OAM (Ji relation, Burkardt sum rule)?

# q4: Show me a visualization of inferred momentum space images of the proton

I'll create a scientific visualization showing the transverse momentum distributions inside the proton, including the Sivers asymmetry effect.Here's what each panel shows:

**(a)** Unpolarized $u$-quark TMD $f_1^u(x, k_T)$ — azimuthally symmetric Gaussian, $\langle k_T \rangle \sim 0.5$ GeV at $x=0.1$.

**(b)** Unpolarized gluon TMD — notably broader ($\langle k_T^2 \rangle_g > \langle k_T^2 \rangle_q$) and larger in magnitude, reflecting gluon dominance at low $x$.

**(c)** The Sivers asymmetry $\Delta f_{\text{Sivers}}^u$ isolated (SIDIS). The dipole pattern ($\propto k_x$ for $\vec{S}_T \parallel \hat{y}$) shows the left-right asymmetry — quarks are preferentially found on one side.

**(d)** Full $u$-quark distribution (unpolarized + Sivers) in SIDIS. The white contours reveal the distortion: the distribution center is shifted left, meaning quarks carry a net $\langle k_x \rangle \neq 0$ in a transversely polarized proton.

**(e)** The **Sivers sign change** in Drell-Yan — the dipole is flipped compared to (c). This is the Collins (2002) prediction that SpinQuest/E1039 and the EIC aim to definitively confirm.

**(f)** Boer-Mulders $\cos(2\phi)$ modulation — a quadrupole pattern arising from quark transverse polarization in an *unpolarized* proton. Different angular structure from the Sivers dipole.

Parameters are Gaussian ansatz inspired by MAP/Pavia global fits — illustrative, not quantitative extractions. The real distributions have non-Gaussian tails and nontrivial $x$- and $Q^2$-dependence from TMD evolution.

