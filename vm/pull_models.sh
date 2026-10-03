#!/usr/bin/env bash
# Pull a single-family model ladder (Qwen2.5-Coder) so results compare model size, not vendor.
#   bash vm/pull_models.sh             # all sizes (~37 GB)
#   bash vm/pull_models.sh 0.5b 7b     # only these sizes
# Approximate download sizes (4-bit): 0.5b 0.4 GB, 1.5b 1.0 GB, 3b 1.9 GB, 7b 4.7 GB,
# 14b 9.0 GB, 32b 20 GB. The 32B fits the A5000's 24 GB on its own.
set -uo pipefail

SIZES=("$@")
[ ${#SIZES[@]} -eq 0 ] && SIZES=(0.5b 1.5b 3b 7b 14b 32b)
echo "Free disk in \$HOME: $(df -h --output=avail "$HOME" | tail -1 | tr -d ' ')"
for size in "${SIZES[@]}"; do
  echo "== pulling qwen2.5-coder:$size"
  ollama pull "qwen2.5-coder:$size" || echo "   FAILED: qwen2.5-coder:$size"
done
echo
echo "=================== PASTE EVERYTHING BELOW THIS LINE ==================="
ollama list | grep -E "NAME|qwen2.5-coder"
echo "Free disk in \$HOME now: $(df -h --output=avail "$HOME" | tail -1 | tr -d ' ')"
echo "========================================================================"
