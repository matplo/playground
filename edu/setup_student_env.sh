#!/usr/bin/env bash
set -euo pipefail

# Bootstrap a student-owned henv and register its Jupyter kernel.
#
# Usage:
#   ./setup_student_env.sh                 # project-local .venv (recommended)
#   ./setup_student_env.sh /path/to/work   # /path/to/work/.venv
#   QG_HENV_NAME=alice-qg ./setup_student_env.sh  # named global henv
#
# Torch backend:
#   QG_TORCH_BACKEND=auto  (default; cu121 when nvidia-smi works, else cpu)
#   QG_TORCH_BACKEND=cu121
#   QG_TORCH_BACKEND=cpu

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
henv_location="${1:-.}"
kernel_name="${QG_KERNEL_NAME:-qg-constituent-ml}"
kernel_display="${QG_KERNEL_DISPLAY_NAME:-Quark/Gluon Constituent ML}"
torch_backend="${QG_TORCH_BACKEND:-auto}"

if ! command -v henv >/dev/null 2>&1; then
    echo "ERROR: henv is not on PATH. Install it before running this script." >&2
    exit 1
fi

if [[ -n "${QG_HENV_NAME:-}" ]]; then
    henv_cmd=(henv --name "$QG_HENV_NAME")
    env_description="named henv '$QG_HENV_NAME'"
else
    henv_cmd=(henv "$henv_location")
    env_description="henv at '$henv_location'"
fi

if [[ "$torch_backend" == "auto" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
        torch_backend="cu121"
    else
        torch_backend="cpu"
    fi
fi
if [[ "$torch_backend" != "cu121" && "$torch_backend" != "cpu" ]]; then
    echo "ERROR: QG_TORCH_BACKEND must be auto, cu121, or cpu." >&2
    exit 2
fi

echo "Creating/updating $env_description"
"${henv_cmd[@]}" --yes --run python -m pip install --upgrade pip
"${henv_cmd[@]}" --yes --run python -m pip install -r "$script_dir/requirements.txt"
"${henv_cmd[@]}" --yes --run python -m pip install "torch==2.5.1" \
    --index-url "https://download.pytorch.org/whl/$torch_backend"

# --sys-prefix keeps the kernelspec with this henv rather than modifying an
# unrelated system Python or a different student's user-wide kernel registry.
"${henv_cmd[@]}" --yes --run python -m ipykernel install --sys-prefix \
    --name "$kernel_name" --display-name "$kernel_display"

"${henv_cmd[@]}" --yes --run python - <<'PY'
import importlib
packages = ['numpy', 'pandas', 'pyarrow', 'matplotlib', 'sklearn', 'torch', 'ipykernel']
for package in packages:
    importlib.import_module(package)
import heppyyier
# Resolve the HEP packages used by the event-generation notebooks now, so a
# missing site package is discovered during setup rather than halfway through class.
heppyyier.load('pythia8')
heppyyier.load('fastjet')
import pythia8, fastjet
import torch
print("Python and HEP dependencies: OK")
print(f"PyTorch {torch.__version__}; CUDA build {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
PY

echo
echo "Environment ready. Start Jupyter with:"
if [[ -n "${QG_HENV_NAME:-}" ]]; then
    echo "  henv --name $QG_HENV_NAME -x jupyter lab"
else
    echo "  henv $henv_location -x jupyter lab"
fi
echo "Select the '$kernel_display' kernel in each notebook."
