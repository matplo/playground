# ALICE nuclear modification summary

Matplotlib summary and interactive selectors built from the **default SciPaperlib library**, using the locally downloaded HEPData YAML files. The bundle contains **65 measurement curves and 1,315 numeric data points**, covering light hadrons, resonances, charm and beauty probes, J/ψ, charged/full/tagged jets and isolated photons.

## Open the results

- **`alice_raa_summary.png`** — four-panel Matplotlib overview at 5.02 TeV; `alice_raa_summary.svg` is the vector version.
- **`alice_raa_explorer.html`** — double-click to open in a browser. Select presets or individual curves, filter by system/energy/object, change the momentum range, toggle systematic errors and export SVG/CSV/JSON. Entirely self-contained; no installation, server or network needed.
- **`alice_raa_explorer.ipynb`** — executed Jupyter notebook with Matplotlib and ipywidgets controls. Keep the notebook beside `raa_plot.py`, the overview image and `data/`. Run all cells to activate controls; export cells save the current selection.
- **`SOURCES.md`** — table-by-table DOI catalogue and measurement definitions.
- **`data/curves.json`** — normalized plotting data plus original numeric strings, uncertainties, qualifiers, source rows and provenance.
- **`data/points.csv`** — convenient numerical table. Full error components are in JSON/YAML.
- **`data/raw/`** — byte-for-byte copies of selected original HEPData YAML files.
- **`data/validation.json`** — input checksums, row counts, uncertainty labels and omitted nonnumeric placeholders.

The overview is a readable selection from the 65 curves, not an attempt to overlay them all. Other presets include Pb–Pb at 2.76 TeV, Xe–Xe at 5.44 TeV, p–Pb at 8.16 TeV and minimum-bias O–O/p–O results at 5.36/9.62 TeV.

## Python usage

Use an existing Jupyter environment, with dependencies installed in its Python kernel:

```bash
python -m pip install -r requirements.txt
```

Then open `alice_raa_explorer.ipynb` in Jupyter or a notebook-capable editor and run all cells. If Jupyter is not installed, install `jupyterlab` and run `jupyter lab alice_raa_explorer.ipynb`.

Recreate the summary without Jupyter:

```bash
python raa_plot.py
```

For a custom Matplotlib plot:

```python
from raa_plot import plot_selected
fig = plot_selected(['pb_d0', 'pb_b_d0', 'pb_djet'], xlim=(1, 60))
fig.savefig('my_comparison.png', dpi=300, bbox_inches='tight')
fig.savefig('my_comparison.svg', bbox_inches='tight')
```

With the cached dependencies on the originating machine, the build was run using:

```bash
MPLBACKEND=Agg uv run --offline --with matplotlib --with numpy --with pyyaml --with nbformat --with nbclient --with ipykernel --with ipywidgets python raa_plot.py
```

`build_data.py` is the extraction step and uses the original machine's library path. It is not needed for plotting. `build_artifacts.py` regenerates the HTML and executes the notebook using the already bundled data. `selection.py` explicitly identifies every accepted measurement column and preset.

## Selection and interpretation

The overview distinguishes central Pb–Pb **RAA**, centrality-selected p–Pb **QpPb**, and minimum-bias/non-single-diffractive **RpPb**. “Most central” means the most-central pT-differential bin in the selected analysis, not one identical percentile interval: the figure labels show 0–5%, 0–10%, 0–20% and 2–10% as applicable. The p–Pb charged-particle QpPb uses ZNA centrality with the Pb-side hybrid scaling prescription. The oxygen curves are explicitly minimum bias, not central-event data. Charge-conjugate states follow the source definitions even where only one charge is written in a short label.

This is a **curated subset**, not an exhaustive census of all ALICE measurements or a combined ALICE result. No theoretical model curves, RCP, ratios of two RAA values or pT-integrated points are mixed into the pT curves. For example, the inspected Υ table provides pT dependence for 0–90%; it was excluded from the central-pT collection. The absence of a curve here does not imply that ALICE has not measured it.

Different acceptances, reference constructions, energies, centralities and jet definitions are retained. Jets and hadrons at equal plotted pT do not represent equal parent-parton energies. The figure should not be used to extract a mass-ordering significance or an energy-loss value by comparing curves directly.

## Uncertainty and axis policy

1. Statistical errors are vertical bars. Point-dependent systematic errors are shaded boxes. Asymmetric errors and deposited percent errors are handled explicitly.
2. A deposited total systematic is preferred over its components. If no total exists, named point-dependent components are combined in quadrature for visualisation, without constructing a covariance model. In the selected J/ψ QpPb tables, the correlated pT-global components are classified as separate scale errors, following the paper figure conventions.
3. **Separate global scale uncertainties are not drawn.** Deposited normalization, luminosity, nuclear-overlap, branching-ratio and identified pT-global terms remain in JSON/YAML and are visible in the explorer's source details. Global terms given only in prose or qualifiers are retained as metadata, not reconstructed numerically. Some supplied totals may already include global contributions; source definitions apply.
4. When only bin edges are supplied, the arithmetic midpoint is used. Source values and bins are preserved without interpolation or rebinning. Centre-only jet tables get narrow decorative systematic boxes; their widths are not claimed to be experimental bins. A bin beginning at zero is clipped at the positive display limit on logarithmic axes. Display limits may omit low-pT points; all points remain in the data and explorer.
5. **D⁰-tagged jets (HEPData 168854):** the deposited statistical uncertainty decreases to about 1.53×10⁻⁵ at high pT, unusually small relative to the observable. It is preserved verbatim and flagged with †; it has not been validated against an author correction. Check with the source before a quantitative significance calculation.
6. **Full jets (HEPData 93739, Table 31):** the independent-variable header says `R=0.4`; the numerical bins are plotted as jet pT using the table description and paper Figure 6. The original header is retained. The selected full-jet tables impose a leading-track cut of 5 or 7 GeV/c in both pp and Pb–Pb.
7. **Centrality metadata:** for the ML charged jets and D⁰-tagged jets, the 0–10% selection is supplied from the source paper because the deposited qualifiers omit it. For charged particles, some centrality information occurs in column headers instead of qualifiers.
8. **p–Pb charged-particle QpPb scale entries:** common TpA and normalization uncertainties are repeated as bare numbers without percent signs. They are retained exactly and excluded from plotted point errors; no fractional reinterpretation is invented.

## Provenance and validation

The SciPaperlib dataset table index was empty although downloaded YAML files were present. `read_dataset_table` reported “Table not indexed”. Extraction therefore used the existing local YAML files after checking each SHA256 against `get_dataset` metadata. No data were digitized from plots or invented. Paper text search also returned “database is locked”; missing selection definitions were checked against the linked primary papers. All plotted numbers still come exclusively from the default library's existing datasets.

`source_manifest.json` records the consulted dataset manifests and their original absolute-library-relative paths. Original strings are retained using PyYAML's BaseLoader; normalized floats are only a plotting representation. The 88 omitted rows are nonnumeric placeholders in shared species tables, not zeros or discarded outliers. Every omission is logged.

The build validates file checksums, numeric values, bin boundaries, row lengths and error-label classification. All 59 copied YAML files from 34 HEPData records passed checksum validation. The five notebook code cells executed successfully in an IPython in-process kernel (the sandbox blocks Jupyter kernel sockets), including the independent checksum and central-value checks. All nine Matplotlib presets rendered, and the overview and selected-curve plot were visually inspected.

The actual HTML JavaScript passed 25 checks in a minimal DOM test harness: all presets, all 65 datasets, filters, selection changes, axis and uncertainty controls, source cards, invalid ranges and SVG/CSV/JSON exports. **A browser engine/UI was unavailable**, so CSS rendering, native browser downloads and live widget interaction were not tested in a browser. See `qa/python_validation.json`, `qa/explorer_validation.json` and `validate_explorer.cjs`. The upstream table content is not thereby certified error-free (see the D⁰-jet note).

Data credit: **ALICE Collaboration / HEPData**. Retain the table DOIs and the original data licenses recorded in `source_manifest.json` when reusing or sharing the data. Source papers and dataset versions are listed in `SOURCES.md`.
