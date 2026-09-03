# Student setup

## Prerequisites

Students need `henv` on `PATH`, network access during initial installation,
and optionally an NVIDIA driver visible through `nvidia-smi` for GPU training.

```bash
henv --version
```

If `henv` is unavailable, follow the course/site instructions or install it:

```bash
curl -fsSL https://raw.githubusercontent.com/matplo/henv/main/henv \
  -o ~/.local/bin/henv
chmod +x ~/.local/bin/henv
```

Make sure `~/.local/bin` is on `PATH` before continuing.

## Recommended setup: project-local environment

After copying this directory to the student's workspace:

```bash
cd /path/to/students/edu
./setup_student_env.sh
henv . -x jupyter lab
```

In Jupyter, select **Quark/Gluon Constituent ML** as the notebook kernel. The
kernel and packages live in `edu/.venv`; they do not modify the system Python
or another student's environment.

Do not copy an existing `.venv` between accounts or machines. Copy the source
directory and run `./setup_student_env.sh` again in its new location.

## What the setup script does

`setup_student_env.sh`:

1. creates or updates the selected `henv`;
2. upgrades pip inside that environment;
3. installs the scientific/Jupyter stack from `requirements.txt`;
4. installs PyTorch separately from a CUDA or CPU wheel index;
5. registers the course kernel inside the henv using `--sys-prefix`;
6. asks `heppyyier` to resolve Pythia8 and FastJet; and
7. imports the main packages and reports Python, CUDA, and GPU status.

PyTorch is intentionally absent from `requirements.txt`: its package source
depends on whether the student will run on an NVIDIA GPU or CPU.

## GPU and CPU selection

The default, `QG_TORCH_BACKEND=auto`, selects the CUDA 12.1 wheel when
`nvidia-smi` succeeds and otherwise installs the CPU wheel.

```bash
QG_TORCH_BACKEND=cu121 ./setup_student_env.sh
QG_TORCH_BACKEND=cpu ./setup_student_env.sh
```

The notebooks still fall back to CPU if CUDA cannot access a GPU at runtime.

## Named henv alternative

```bash
QG_HENV_NAME=alice-qg ./setup_student_env.sh
henv --name alice-qg -x jupyter lab
```

Kernel labels can also be customized:

```bash
QG_KERNEL_NAME=alice-qg-kernel \
QG_KERNEL_DISPLAY_NAME="Alice quark/gluon ML" \
./setup_student_env.sh
```

## Verify the selected environment

Without opening Jupyter:

```bash
henv . -x python -c \
  "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Inside a notebook, the environment cell prints the PyTorch version, compiled
CUDA version, selected device, and GPU name. If packages are missing, stop
Jupyter, rerun `./setup_student_env.sh`, restart it through the same henv, and
select the course kernel again.

## Notebook order

1. `demo_quark_gluon_samples.ipynb`
2. `demo_quark_gluon_constituent_setup.ipynb`
3. the PFN, Particle Transformer, and ParticleNet notebooks in any order
4. `demo_quark_gluon_model_evaluation.ipynb`
5. `demo_quark_gluon_model_interpretation.ipynb`

Prepared arrays and trained models are automatically associated with the
fingerprint of the generated Parquet sample.

## Generate more statistics

For ten times the default event count:

```bash
QG_N_EVENTS=200000 henv . -x jupyter nbconvert \
  --to notebook --execute --inplace demo_quark_gluon_samples.ipynb
```

For an interactive run:

```bash
export QG_N_EVENTS=200000
henv . -x jupyter lab
```

Downstream notebooks detect the new Parquet fingerprint and build a new
prepared dataset. Older models remain under their original fingerprint.

## Moving data and models

- `data/*.parquet` contains generated samples and may be copied to give
  students common events.
- `data/qg_prepared/` is derived and can be regenerated.
- `artifacts/qg_models/` contains weights plus required JSON configuration;
  copy the whole directory when sharing trained models.
- `.venv/` is machine/account specific and must not be copied.

If only notebooks and scripts are moved, run the setup script and then
generate or supply the Parquet input before starting the ML notebooks.
