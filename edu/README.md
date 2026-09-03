# HEP Python educational notebooks

These notebooks progress from Pythia8 event generation and FastJet clustering
to constituent-level quark/gluon classification with tabular baselines,
Particle Flow Networks, Transformers, and graph neural networks.
Physics and machine-learning terminology is introduced in plain language before
the technical details, so the sequence can also serve advanced high-school
students encountering collider physics or AI for the first time.

## Setup

Create a student-owned project environment and launch Jupyter:

```bash
./setup_student_env.sh
henv . -x jupyter lab
```

Select the **Quark/Gluon Constituent ML** kernel. See
[STUDENT_SETUP.md](STUDENT_SETUP.md) for prerequisites, CPU/GPU selection,
using an already-active henv, named environments, verification, scaling event
statistics, and moving saved data or models.

## Suggested order

1. `demo_pythia_fastjet.ipynb`
2. `demo_dijet_event_display.ipynb`
3. `demo_quark_gluon_samples.ipynb`
4. `demo_quark_gluon_classification.ipynb`
5. `demo_quark_gluon_basic_model_visualization.ipynb`
6. `demo_quark_gluon_constituent_setup.ipynb`
7. Train the PFN, Particle Transformer, and ParticleNet models in any order.
8. `demo_quark_gluon_architecture_visualization.ipynb` (can also be opened before training)
9. `demo_quark_gluon_model_evaluation.ipynb` (all seven classifiers on one test split)
10. `demo_quark_gluon_model_interpretation.ipynb`

Generated datasets, prepared arrays, plots, and trained model bundles are
intentionally excluded from Git. The sample and training notebooks recreate
them locally. The detailed implementation and tuning record is in
[QUARK_GLUON_CONSTITUENT_ML_PLAN.md](QUARK_GLUON_CONSTITUENT_ML_PLAN.md).

The main student controls—event count, seed, output path, top-particle count,
input path, and quick/full training mode—are ordinary variables near the top
of the relevant notebooks.
