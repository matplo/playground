# Constituent-Level Quark/Gluon ML Workflow

This document is the durable design and tuning reference for the advanced
quark/gluon-classification notebooks. Keep it synchronized with material
changes to the implementation.

## Status

- [x] Make event generation reproducible and statistics-configurable.
- [x] Prepare a fingerprinted, scalable constituent dataset.
- [x] Train and save a Particle Flow Network (PFN).
- [x] Train and save a compact Particle Transformer.
- [x] Train and save a compact ParticleNet-style graph network.
- [x] Load and compare saved models on a common test sample.
- [x] Interpret model reliance with physics-aware ablations and attributions.
- [x] Report progress for long Python-level generation and ML loops.
- [x] Define physics and ML terminology at a high-school-accessible level.
- [x] Keep distributed notebooks free of stored outputs and execution state.
- [x] Execute and validate the complete quick-mode workflow.

## Teaching style

Introduce each idea in plain language before giving its technical definition or
equation. New physics terms include parton, hard scattering, shower,
hadronization, jet, constituent, transverse momentum, pseudorapidity,
azimuth, acceptance, and truth matching. New ML terms include feature, label,
supervised learning, parameter, hyperparameter, normalization, leakage,
epoch, loss, backpropagation, validation, overfitting, inference, ROC/AUC,
ablation, and attribution.

The explanations should remain technically accurate: quark/gluon truth labels
are simulation-dependent associations, model scores are not automatically
probabilities, and model interpretation measures reliance rather than physical
causation. Equations and real nomenclature are retained, with definitions next
to their first use. Repository notebooks remain clean; students generate
outputs locally by running the cells.

## Workflow

1. `demo_quark_gluon_samples.ipynb` generates the Parquet jet sample.
2. `demo_quark_gluon_constituent_setup.ipynb` validates and prepares it.
3. The PFN, Transformer, and ParticleNet notebooks train independent models.
4. `demo_quark_gluon_model_evaluation.ipynb` reloads and compares them.
5. `demo_quark_gluon_model_interpretation.ipynb` studies model reliance.

The programmatic alternative is `qg_constituent_ml.py`. Prepared data and
model bundles are namespaced by a SHA-256 fingerprint of the source Parquet
file, preventing accidental comparisons across different generated samples.

## Generation and scaling

The generator accepts these environment overrides:

| Variable | Default | Meaning |
|---|---:|---|
| `QG_N_EVENTS` | `20000` | Number of attempted Pythia events |
| `QG_SEED` | `7` | Pythia random seed |
| `QG_OUTPUT_DIR` | `data` | Output directory |

For example, a student can generate ten times the default statistics with
`QG_N_EVENTS=200000`. A generation manifest records all run parameters,
counts, schema version, and the Parquet fingerprint. The current design uses
one reproducible rerun rather than appendable shards. Sharded Parquet plus a
glob-based source manifest is the natural extension beyond samples that fit
comfortably in memory.

Long-running event generation uses a frontend-independent `tqdm` text
progress bar. The same approach is used for constituent preparation and
baseline feature extraction, so increasing the sample size does not leave
students without feedback or require Jupyter widget support.

## Constituent representation

Every stored constituent is used; there is no top-N truncation. Continuous
inputs are `log(z)`, relative eta, wrapped relative phi, and `log(delta-R)`.
PID is represented by six categories: photon, charged hadron, neutral hadron,
electron, muon, and other. Absolute jet pt/eta and truth-matching fields are
excluded to reduce kinematic and label leakage.

Splits are stable hashes of `event_id`, so jets from one event cannot cross
train/validation/test boundaries and adding higher-numbered events does not
reshuffle existing assignments. Normalization is fitted on training
constituents only. Flat `.npy` arrays plus offsets allow memory mapping;
padding occurs dynamically per batch.

## Models

- **PFN:** shared constituent MLP, masked sum pooling, jet-level MLP.
- **Particle Transformer:** constituent attention with pairwise angular bias,
  followed by class-attention pooling; no positional sequence encoding.
- **ParticleNet-style:** masked k-nearest-neighbor graphs and EdgeConv blocks.

Each notebook supports `QG_RUN_MODE=quick|full`. Quick mode is interactive;
full mode uses all training jets and larger/longer configurations. Models
automatically use CUDA and mixed precision when available, with conservative
batch sizes for the node's 8 GB Quadro RTX 4000.

Training uses one continuously updated batch-level progress bar with the
current epoch, loss, validation AUC, and early-stopping counter. Evaluation
and interpretation show separate operation-level bars; short vectorized and
plot-formatting loops intentionally remain unwrapped.

## Environment

Run the visible bootstrap script once. The intended GPU build is:

```bash
henv -x python -m pip install torch==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121
```

`requirements.txt` holds the non-Torch stack, including `tqdm`.
`setup_student_env.sh` creates or updates a student-owned henv, selects a CUDA
or CPU Torch wheel, and runs import/device smoke tests. It registers the
course kernel for a newly selected environment; `--current` instead refreshes
and reuses that henv's existing HEP kernel. Notebooks only validate
dependencies and point back to this setup, avoiding hidden environment
mutation. CPU remains a supported fallback.

## Saved-model contract

Each bundle contains JSON configuration, a weights-only PyTorch state dict,
training history, metrics, and held-out predictions keyed by event and jet.
It also records the dataset/split fingerprints, input schema, normalization,
PID map, PyTorch version, device, timing, and parameter count. Reloaded models
must reproduce their saved predictions within floating-point tolerance.

A model may score a new compatible Parquet sample using its stored
preprocessing. Metrics from different dataset or split fingerprints are
reported separately rather than placed in a misleading direct comparison.

## Interpretation

The correctly posed question is which inputs a trained model relies on—not
which features causally define quark and gluon jets. The interpretation
notebook uses:

- score correlations with familiar jet-shape observables;
- common group ablations of momentum, angular, PID, soft, core, and
  wide-angle information;
- integrated gradients for continuous inputs;
- categorical occlusion for PID inputs;
- per-constituent eta-phi attribution displays.

Attention maps and graph neighborhoods are diagnostics, not standalone
feature importance. Correlated inputs, distribution shift, generator labels,
and unphysical perturbations must be discussed when interpreting results.

## Validation checklist

- Deterministic small generation for a fixed seed.
- Automatic invalidation after source statistics change.
- Complete constituent accounting and artifact round trips.
- Event-disjoint stable splits and train-only normalization.
- Padding and permutation invariance checks for all architectures.
- Quick-mode notebook execution with CUDA and CPU fallback.
- Checkpoint save/reload and identical held-out predictions.
- Labeled and unlabeled inference on another compatible sample.
- Finite metrics, ablations, and attributions without assuming a model rank.

## Decision log

- Use a single larger reproducible rerun instead of shards for the first
  educational version.
- Use broad physics PID categories rather than ordinal PDG IDs.
- Provide PFN, Transformer, and ParticleNet as separate lessons.
- Use an explicit setup script plus requirements file; notebooks validate the selected kernel.
- Use progress bars for operations whose runtime scales materially with the
  event or jet count, including one stable bar during training.
- Save model bundles and add dedicated evaluation and interpretation lessons.

## Implementation record (2026-09-02)

The quick workflow was executed successfully on the Quadro RTX 4000 with
PyTorch 2.5.1+cu121. For dataset fingerprint `1e09b49ecb4d`, held-out ROC
AUCs were 0.807 (PFN), 0.806 (Transformer), and 0.798 (ParticleNet-style).
All saved bundles reloaded with matching predictions. A temporary 30-event
generation test confirmed `QG_N_EVENTS`, `QG_SEED`, the manifest, and
byte-identical Parquet output for the same seed.

Student portability is provided by `setup_student_env.sh`, `requirements.txt`,
and `STUDENT_SETUP.md`. The script supports project-local, named, or already
active henvs; selects a CUDA/CPU Torch wheel; registers a course kernel or
refreshes the active henv's HEP kernel as appropriate; resolves
Pythia8/FastJet through heppyyier; and runs a smoke test. The requirements
file was verified with a networked pip dry-run.
