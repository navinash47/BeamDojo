#!/usr/bin/env bash
# Run inside isaac-lab-base. Fill missing Python deps without touching
# Isaac Sim setuptools/packaging (those uninstalls break pip).
set -euo pipefail
PY=/workspace/isaaclab/_isaac_sim/python.sh
PIPZ=/tmp/pip.pyz
if [[ ! -f "$PIPZ" ]]; then
  curl -fsSL -o "$PIPZ" https://bootstrap.pypa.io/pip/pip.pyz
fi
# Wheels only — never use isolated sdist builds (flatdict used to need pkg_resources).
"$PY" "$PIPZ" install --no-build-isolation \
  flatdict \
  'gymnasium==1.2.1' \
  'rsl-rl-lib==3.0.1' \
  prettytable hidapi pyglet einops
"$PY" -c "import flatdict, gymnasium, rsl_rl, torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0)); print('REPAIR_OK')"
