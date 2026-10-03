#!/usr/bin/env bash
# Pull the benchmark model set. Ordered small-first across families, so every vendor
# arrives early and experiments can start before the large models finish.
#   bash vm/pull_models.sh            # everything (~60 GB)
#   bash vm/pull_models.sh core       # Qwen3.5 size ladder only (~31 GB)
#   bash vm/pull_models.sh xfamily    # Granite, Ministral, Gemma at matched sizes (~28 GB)
#   bash vm/pull_models.sh gemma3:4b  # any explicit tags
#
# Design: one current family at five sizes (the size ladder: isolates model size) plus
# three other vendors at matched ~3-4B and ~8-12B sizes (shows results aren't Qwen-specific).
set -uo pipefail

CORE=(qwen3.5:0.8b qwen3.5:2b qwen3.5:4b qwen3.5:9b qwen3.5:27b)
XFAMILY=(granite4.2:3b ministral-3:3b gemma3:4b granite4.2:8b ministral-3:8b gemma3:12b)
ALL=(qwen3.5:0.8b qwen3.5:2b granite4.2:3b ministral-3:3b gemma3:4b qwen3.5:4b
     granite4.2:8b ministral-3:8b qwen3.5:9b gemma3:12b qwen3.5:27b)

case "${1:-all}" in
  all) MODELS=("${ALL[@]}") ;;
  core) MODELS=("${CORE[@]}") ;;
  xfamily) MODELS=("${XFAMILY[@]}") ;;
  *) MODELS=("$@") ;;
esac

echo "Free disk in \$HOME: $(df -h --output=avail "$HOME" | tail -1 | tr -d ' ')"
for model in "${MODELS[@]}"; do
  echo "== pulling $model"
  # Flaky networks drop long downloads; ollama resumes partial layers, so just retry.
  for attempt in 1 2 3 4 5 6 7 8; do
    ollama pull "$model" && break
    echo "   connection dropped (try $attempt of 8); resuming in 5s"; sleep 5
  done
  ollama list | awk '{print $1}' | grep -qx "$model" || echo "   FAILED: $model (re-run this script to resume)"
done
echo
echo "=================== PASTE EVERYTHING BELOW THIS LINE ==================="
ollama list
echo "Free disk in \$HOME now: $(df -h --output=avail "$HOME" | tail -1 | tr -d ' ')"
echo "========================================================================"
