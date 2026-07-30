# Spec: Pythia8 + FastJet parton-jet response study

Instructions for a coding agent to reproduce `demo_pythia_bjet_response.ipynb` from scratch, or to build
an analogous study (different flavor, different observable). This captures not just *what* the notebook
does but *why*, since several choices are non-obvious and were arrived at by trial, measurement, or fixing
a real bug.

## Goal

Generate $b\bar b$ events, cluster anti-$k_t$ jets over a range of radii $R$, match each jet to the outgoing
b-quark that produced it, and characterize the jet $p_T$ response (jet $p_T$ / b-quark $p_T$) as a function
of $R$ and b-quark $p_T$. Produce both per-R diagnostic plots and cross-R summary plots, plus a
charged-particle-only variant.

## Environment

Two layered tools: [henv](https://github.com/matplo/henv) manages the Python virtualenv itself;
[heppyyier](https://github.com/matplo/heppyyier) (installed *inside* that venv) builds and exposes Pythia8 /
FastJet / HepMC3 / etc. to Python via cppyy. Both are one-time setup per machine/env — don't add "run this
before starting the kernel" boilerplate to notebooks; they only need the `heppyyier.load(...)` calls below.

### 1. Install henv (once per machine)

```bash
curl -fsSL https://raw.githubusercontent.com/matplo/henv/main/henv \
  -o ~/.local/bin/henv && chmod +x ~/.local/bin/henv
```

(or the self-installing form: `curl -fsSL .../henv | bash -s -- --install`, which also checks `~/.local/bin`
is on `PATH`).

### 2. Create an env and install packages (once per project/env)

```bash
cd ~/myanalysis        # or wherever the notebook lives
henv .                 # creates ./.venv, prompts to install heppyyier, drops into an activated subshell
# → prompt becomes: (henv:myanalysis) ...

heyy install fastjet pythia8      # inside the subshell; heyy/her/heppyyier are equivalent CLI names
exit                               # back to the parent shell
```

`henv .` picks up an existing `./.venv` on subsequent calls (no reinstall, no prompts) — safe to call
routinely at the start of a session. Other invocation forms: `henv` alone uses a single global env at
`$HOME/.henvs/default`; `henv --name NAME` uses a named global env at `$HOME/.henvs/NAME` (handy for sharing
one env across multiple analysis directories instead of a `.venv` per project).

### 3. Run commands inside the env

Prefer `henv <location> --run <command>` over manually activating — it performs the same initialization as
the interactive subshell (venv activation, `HEPPYYIER_PACKAGES_DIR`, module-file regeneration) without
spawning a subshell, which is what makes it composable from scripts/agents:

```bash
henv . --run python -c "import fastjet; print('ok')"
henv . --run pip list
henv . --run heyy list                              # see what heppyyier packages are installed
```

Full notebook execution:

```bash
henv . --run jupyter nbconvert --to notebook --execute --inplace <nb>.ipynb \
  --ExecutePreprocessor.timeout=600
```

A 9-bin x 2 track-type scan at 2000 events/bin takes ~130s — always execute end-to-end and check for error
outputs before calling a notebook change done; don't just eyeball the diff.

### 4. Inside the notebook

```python
import heppyyier
heppyyier.load('pythia8')
heppyyier.load('fastjet')
```

`heppyyier.load()` is a no-op if the package was already loaded (e.g. via a heppyyier-aware Jupyter kernel
or `module load`), so it's safe to always include.

## Event generation strategy

**Process**: `HardQCD:hardbbbar = on` — genuine $gg,\,q\bar q \to b\bar b$ heavy-flavor pair production at
the hard 2→2 level (not gluon splitting in the shower). This is the standard choice for a "b-jet" study as
opposed to inclusive QCD jets that happen to contain a B-hadron.

**Statistics across a wide $p_T$ range**: the b-quark $p_T$ spectrum falls steeply, so a single Pythia run
would starve the high-$p_T$ end of the target range. Fix: run **one Pythia instance per target $p_T$ bin**,
each with only `PhaseSpace:pTHatMin` set to that bin's lower edge — **do not set `PhaseSpace:pTHatMax`**
(leave it at Pythia's unbounded default). Do not use a bounded `[pTHatMin, pTHatMax]` window per bin either
(an earlier version of this notebook did, with an extra padding margin to catch edge spillover — that's
unnecessary complexity, see below).

Why min-only, no window, works and is *better* than a bounded window: within one run, events concentrate
near that run's own threshold (falling cross section), so a run with `pTHatMin=50` naturally supplies most
of its statistics to the `[50,60)` bin. But it also generates a non-trivial tail above 60, 70, 80... GeV —
verified empirically:

```
pTHatMin=50, no Max, 2000 events → counts per [10,20,...,100) bin:
[0, 0, 0, 0, 2114, 874, 448, 230, 112]
```

Sort every generated event into its target bin by its **actual** generated b-quark $p_T$ (not by which run
produced it), and this overflow becomes bonus statistics for the bins above each run's own threshold — pure
upside, no double-counting concern, since we never merge these runs into a single cross-section-normalized
sample; each run's events are just individually classified by measured $p_T$ and pooled by that classification
only. Comparing the final per-(R, $p_T$-bin) matched-pair counts, min-only-no-max gave equal or better
statistics than the padded-window version it replaced.

Expose the per-bin event count explicitly and early, as an editable dict keyed by that bin's `pTHatMin`:

```python
n_events_per_bin = {lo: 2000 for lo in pt_edges[:-1]}   # override individual entries as needed
```

## Truth object selection

Outgoing b-quarks = particles with `abs(id()) == 5` and `statusAbs() == 23` (Pythia8's "outgoing particle of
the hardest subprocess" status code) — i.e. the born-level kinematics right after the 2→2 hard process,
before parton-shower/hadronization further modifies them. A 2→2 process yields exactly 2 such particles per
event (verified: 2000 events → 4000 b-quarks).

## Jet reconstruction

- FastJet anti-$k_t$, one `JetDefinition` per $R$ value.
- Final-state particles for clustering: `p.isFinal() and p.isVisible()` (excludes neutrinos automatically).
  For a **charged-particle-only** variant (mimics track-based reconstruction), add `and p.isCharged()`.
- Build the "all particles" and "charged particles" PseudoJet lists **from the same generated event** and
  cluster both, rather than running Pythia twice — event generation dominates the runtime, clustering is
  cheap, so this effectively gets a second full analysis for free (confirmed: adding the charged-only
  clustering pass added negligible wall time to the scan).
- Jet acceptance: `|jet.eta()| < eta_cut` (pseudorapidity, 2.5 by default) — this is a detector-style
  acceptance cut and is conventionally on pseudorapidity, unlike the matching metric below.
- `inclusive_jets(jet_pt_min)` with a low threshold (e.g. 2 GeV) so small-$R$ jets that only capture a
  fraction of a low-$p_T$ b-quark aren't pre-filtered away.

## Matching truth to jets — the rapidity gotcha

For each b-quark, find the closest accepted jet by `bp4.delta_R(jet)` (where `bp4` is a `PseudoJet` built
from the b-quark's own px/py/pz/E) and keep the match if `delta_R < R` (the jet's own radius — i.e. the
b-quark direction must fall inside the jet's cone).

**Gotcha**: FastJet's `PseudoJet.delta_R()` is defined using **rapidity** $y$, not pseudorapidity $\eta$.
For a massless constituent the two coincide, but the b-quark has ~4.8 GeV mass, and a clustered jet also has
non-negligible invariant mass, so $y \ne \eta$ measurably near threshold. This was caught by plotting
$\Delta\eta$ vs $\Delta\phi$ for matched pairs and finding a handful of points *outside* the $\Delta R = R$
matching circle — impossible if $\Delta\eta$ had been the actual matching metric. Fix: store and plot
$\Delta y$ (`jet.rap() - bquark_pseudojet.rap()`), not $\Delta\eta$, for any diagnostic that must be
consistent with the matching cut. Store both `eta` (for the acceptance cut) and `rap` (for matching
diagnostics) on every jet/b-quark.

`phi()`/`phi_std()` both return $(-\pi, \pi]$ for Pythia8 `Particle` and FastJet `PseudoJet` respectively;
wrap their difference into $(-\pi, \pi]$ with `(dphi + pi) % (2*pi) - pi` before using it.

## Derived quantities & statistics

- `pt_ratio = jet_pt / b_pt` — the primary observable.
- Per-$(R, p_T\text{ bin})$ **median + 16–84th percentile band** (`profile()` helper) for the per-R plots.
  These bands legitimately exceed 1 at the upper edge — a jet can pick up more $p_T$ than its matched
  b-quark carried, from ISR/FSR or multi-parton-interaction activity landing inside the cone. Don't try to
  "fix" this; show the raw ratio histograms (see plot catalog) so it's visibly explained rather than
  asserted.
- **Most-probable-value (mode) summary**: do **not** use a raw histogram argmax — it's noisy and, for the
  mildly bimodal response distributions typical at small-to-medium $R$ (a b-quark's momentum is either
  mostly "captured" by the cone or mostly "escapes" it), the argmax jumps erratically between two
  comparably-tall bins as $p_T$ changes, producing a chaotic-looking summary curve. Use a Gaussian KDE
  (`scipy.stats.gaussian_kde`) evaluated on a fine grid and take the grid point of maximum density instead —
  much smoother, and any remaining sharp single-step transitions (as the taller of the two humps switches)
  are then a real physics feature, not sampling noise. Verify this by eye: plot the ratio histogram directly
  for a few example bins and confirm the two-hump structure is real.

## Notebook code organization

- One config cell up top: all tunable parameters (`R_values`, `pt_edges`, `eta_cut`, `jet_pt_min`,
  `n_events_per_bin`, `ratio_range`, example-bin switches, a small fixed categorical color palette). No
  numeric literals embedded directly in f-strings, plot titles, or Pythia `readString` calls elsewhere in
  the notebook — everything traces back to a named parameter in this cell.
- One helpers cell: small single-purpose functions (`build_pythia`, `outgoing_bquarks`,
  `final_state_pseudojets`, `wrap_phi`, `bin_index_for_pt`, `finalize_data`, `most_probable`, `profile`).
- `run_scan(track_types=('all','charged'), ...)` as a single reusable function returning a dict of
  finalized-data dicts keyed by track type — not inlined scan code repeated per variant.
- A separate "plotting helpers" cell defining every figure-producing function
  (`plot_perR_ratio`, `plot_summary_mpv`, `plot_pt_correlation`, `plot_matching_debug`,
  `plot_ratio_distribution`), each taking a `data` dict + labels + `savepath=None` and returning the
  `Figure`. Every subsequent "section" of the notebook is then a 1-3 line call cell — this is what makes it
  cheap to add the charged-particle variant (reuse `plot_perR_ratio`/`plot_summary_mpv` with a different
  `data` dict and accent color) without duplicating plotting code.
- An "example $p_T$ bin(s)" switch pattern for expensive-per-bin diagnostic plots (matching debug, ratio
  distributions): a single config list, e.g. `EXAMPLE_PT_BINS = [50.0]`, consumed by a `for pt_value in
  EXAMPLE_PT_BINS: plot_fn(...)` loop. Set it to `list(pt_centers)` to produce one figure per bin instead of
  a single example — no code change needed, just a config edit.
- Fixed small categorical palette assigned by role, not cycled: `COLOR_ALL` (blue), `COLOR_CHARGED` (green),
  `COLOR_ACCENT` (red, for reference lines/medians in debug plots). Sequential quantities (R, when shown as
  a continuous axis in the summary plot) get a `viridis` colormap + colorbar, not a discrete per-line legend.

## Plot catalog

1. **Per-R response** (`plot_perR_ratio`): 2×3 grid, one panel per $R$, median $p_T$ ratio + 16–84% band vs.
   b-quark $p_T$.
2. **Correlation** (`plot_pt_correlation`): 2×3 grid, jet $p_T$ vs. b-quark $p_T$ hexbin density over the
   *full* pooled sample (not sliced to one bin — this is the standard, most-informative form of a response
   correlation plot), with a $y=x$ reference line.
3. **Matching debug** (`plot_matching_debug`): 2×3 grid, $\Delta y$ vs. $\Delta\phi$ scatter for matched
   pairs in one example $p_T$ bin, with the $\Delta R = R$ circle overlaid. Every point falls inside the
   circle by construction (see the rapidity gotcha above) — the plot is a sanity check on the matching
   logic and a visualization of how tightly jets track the quark direction as $R$ grows.
4. **Ratio distribution** (`plot_ratio_distribution`): 2×3 grid, log-scale histogram of the $p_T$ ratio in
   one example $p_T$ bin, median line marked — this is what makes the >1 tail and the bimodal structure
   visible and explainable rather than asserted.
5. **Summary MPV** (`plot_summary_mpv`): single panel, KDE-mode ratio vs. b-quark $p_T$, one line per $R$
   (viridis colormap + colorbar for $R$), plus a printed matched-pair-count table per $(R, p_T\text{ bin})$
   cell so low-statistics regions are visible, not just implied.
6. Repeat (1) and (5) for the charged-particle-only `data` dict, reusing the same functions with
   `color=COLOR_CHARGED`.

## Known pitfalls (fixed during development — don't reintroduce)

- Histogram-argmax "most probable value" → noisy, replaced with KDE mode.
- $\Delta\eta$ used where the matching metric actually uses $\Delta y$ (rapidity) → points outside the
  matching circle; fixed by storing/plotting rapidity consistently with FastJet's `delta_R()`.
- Bounded `[pTHatMin, pTHatMax]` generation window with an ad hoc padding margin → replaced by
  `pTHatMin`-only, unbounded above, which is simpler and gives as-good-or-better statistics via natural
  overflow into higher bins.
- Redundant jet-$E$/b-quark-$E$ ratio dropped entirely (kept only $p_T$ ratio) after review — don't compute
  parallel observables the analysis doesn't use just because they're cheap to add.

## Extending this to a different study

The reusable core (`run_scan`'s per-bin-threshold generation trick, the rapidity-consistent matching, the
KDE-mode summary, the plotting-function/config-cell notebook structure) is not b-jet-specific. To adapt:
swap `HardQCD:hardbbbar` for the relevant process, the parton selection filter in `outgoing_bquarks`
(id/status codes) for the truth object of interest, and reuse everything else unchanged.
