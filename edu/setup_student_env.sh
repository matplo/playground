#!/usr/bin/env bash
set -euo pipefail

# Bootstrap a student-owned henv and register its Jupyter kernel.
#
# Usage:
#   ./setup_student_env.sh                 # project-local .venv (recommended)
#   ./setup_student_env.sh /path/to/work   # /path/to/work/.venv
#   QG_HENV_NAME=alice-qg ./setup_student_env.sh  # named global henv
#   ./setup_student_env.sh --current       # use the already-active henv
#
# Torch backend:
#   QG_TORCH_BACKEND=auto  (default; cu121 when nvidia-smi works, else cpu)
#   QG_TORCH_BACKEND=cu121
#   QG_TORCH_BACKEND=cpu

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
use_current=false
henv_location="."
if [[ "${1:-}" == "--current" || "${1:-}" == "--active" ]]; then
    use_current=true
    shift
fi
if $use_current && (( $# != 0 )); then
    echo "ERROR: --current does not accept a henv location." >&2
    exit 2
fi
if (( $# > 1 )); then
    echo "ERROR: expected at most one henv location." >&2
    exit 2
fi
if (( $# == 1 )); then
    henv_location="$1"
fi
kernel_name="${QG_KERNEL_NAME:-qg-constituent-ml}"
kernel_display="${QG_KERNEL_DISPLAY_NAME:-Quark/Gluon Constituent ML}"
torch_backend="${QG_TORCH_BACKEND:-auto}"

if $use_current && [[ -n "${QG_HENV_NAME:-}" ]]; then
    echo "ERROR: --current and QG_HENV_NAME cannot be used together." >&2
    exit 2
fi

if ! $use_current && ! command -v henv >/dev/null 2>&1; then
    echo "ERROR: henv is not on PATH. Install it before running this script." >&2
    exit 1
fi

if $use_current; then
    if [[ -z "${HENV_PATH:-}" || -z "${VIRTUAL_ENV:-}" ]]; then
        echo "ERROR: --current requires an active henv (HENV_PATH and VIRTUAL_ENV must be set)." >&2
        echo "Activate one first, for example: eval \"\$(henv --print-activate .)\"" >&2
        exit 2
    fi
    if ! python -c 'import os, pathlib, sys; assert pathlib.Path(sys.prefix).resolve() == pathlib.Path(os.environ["HENV_PATH"]).resolve()' 2>/dev/null; then
        echo "ERROR: the current Python does not belong to the active henv at '$HENV_PATH'." >&2
        exit 2
    fi
    env_description="currently active henv '${HENV_ACTIVE:-$HENV_PATH}' at '$HENV_PATH'"
elif [[ -n "${QG_HENV_NAME:-}" ]]; then
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

run_in_env() {
    if $use_current; then
        command "$@"
    else
        "${henv_cmd[@]}" --yes --run "$@"
    fi
}

echo "Installing/updating dependencies in $env_description"
run_in_env python -m pip install --upgrade pip
run_in_env python -m pip install -r "$script_dir/requirements.txt"
run_in_env python -m pip install "torch==2.5.1" \
    --index-url "https://download.pytorch.org/whl/$torch_backend"

run_in_env python - <<'PY'
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

if $use_current; then
    # Refresh the existing HEP kernelspec after resolving Pythia8/FastJet so
    # its PATH and library variables include every installed HEP package.
    if ! run_in_env heyy kernel update; then
        echo "ERROR: could not update the existing HEP kernel for this henv." >&2
        echo "Create it with 'heyy kernel install', then rerun this script." >&2
        exit 3
    fi
else
    # A newly selected henv gets a course-specific kernelspec. --sys-prefix
    # keeps it out of another student's user-wide kernel registry.
    run_in_env python -m ipykernel install --sys-prefix \
        --name "$kernel_name" --display-name "$kernel_display"
fi

echo
echo "Environment ready. Start Jupyter with:"
if $use_current; then
    echo "  jupyter lab"
elif [[ -n "${QG_HENV_NAME:-}" ]]; then
    echo "  henv --name $QG_HENV_NAME -x jupyter lab"
else
    echo "  henv $henv_location -x jupyter lab"
fi
if $use_current; then
    echo "Continue using the existing 'HEP' kernel for this henv."
else
    echo "Select the '$kernel_display' kernel in each notebook."
fi
