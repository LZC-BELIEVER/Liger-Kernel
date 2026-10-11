#!/bin/bash
# msProf application wrapper for Issue #1799 (CANN 9.1.0, grpo322 env).
#
#   msprof --output=<dir> --ai-core=on --aic-metrics=PipeUtilization \
#          --ascendcl=off --application=<this script> -- --B 4 --iters 2
#
# Device, caches and the CANN root can be overridden from the caller's
# environment; defaults follow REPRODUCE.md.
set -euo pipefail

source /workspace3/lcw/.local/Ascend-9.1.0/cann-9.1.0/aarch64-linux/bin/setenv.bash
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-2}"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export ASCEND_UB_CAPACITY_BITS=1572864
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_grpo_20261011_msprof_dev${ASCEND_RT_VISIBLE_DEVICES}}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/torchinductor_grpo_20261011_msprof_dev${ASCEND_RT_VISIBLE_DEVICES}}"

cd /workspace3/lcw/liger-kernel-ref
export PYTHONPATH=src
exec /workspace2/lanzichang/envs/grpo322/bin/python \
  /workspace3/lcw/SLAI-Ascend-OpForge/docs/lzc/1799/evidence/20261011/msprof/profile_grpo_path.py "$@"
