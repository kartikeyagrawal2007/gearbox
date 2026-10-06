#!/usr/bin/env bash
# A second, CPU-only Ollama on port 11435, for the async experiment's "worker on the CPU"
# placement (vm/async.vm.yaml, tiers ending in -cpu). It reads the same model files as the
# main GPU Ollama, so nothing is downloaded again.
#
#   bash vm/ollama_cpu.sh start    # start it (in the background; survives closing the terminal)
#   bash vm/ollama_cpu.sh check    # prove a model runs there with 0 bytes on the GPU
#   bash vm/ollama_cpu.sh stop
set -uo pipefail

PORT=11435
URL="http://127.0.0.1:$PORT"
PIDFILE="$HOME/ollama-cpu.pid"
LOG="$HOME/ollama-cpu.log"
MODEL="${2:-qwen3.5:4b}"

up() { curl -fs "$URL/api/version" >/dev/null 2>&1; }

case "${1:-start}" in
  start)
    if up; then echo "CPU Ollama already running on :$PORT"; exit 0; fi
    # Hide every GPU from this instance; `check` confirms nothing lands in VRAM.
    CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 GGML_VK_VISIBLE_DEVICES=-1 \
      OLLAMA_HOST="127.0.0.1:$PORT" OLLAMA_NUM_PARALLEL=4 OLLAMA_MAX_LOADED_MODELS=2 OLLAMA_KEEP_ALIVE=30m \
      nohup ollama serve > "$LOG" 2>&1 &
    echo $! > "$PIDFILE"
    for _ in $(seq 1 30); do up && break; sleep 1; done
    if up; then echo "CPU Ollama running on :$PORT (log: $LOG)"; else echo "FAILED to start; see $LOG"; exit 1; fi
    ;;
  check)
    up || { echo "not running; run: bash vm/ollama_cpu.sh start"; exit 1; }
    echo "loading $MODEL on the CPU instance (the first load can take a minute)..."
    curl -fs "$URL/api/generate" -d "{\"model\":\"$MODEL\",\"prompt\":\"Reply with OK.\",\"stream\":false,\"think\":false,\"options\":{\"num_predict\":5}}" >/dev/null \
      || { echo "FAILED: $MODEL did not answer; see $LOG"; exit 1; }
    curl -fs "$URL/api/ps" | python3 -c '
import json, sys
models = json.load(sys.stdin).get("models", [])
for m in models:
    vram_gb = m.get("size_vram", 0) / 2**30
    verdict = "OK   on the CPU (0 bytes in VRAM)" if vram_gb == 0 else "FAIL %.1f GB is on the GPU" % vram_gb
    print("   " + verdict + ": " + m["name"])
sys.exit(0 if models and all(m.get("size_vram", 0) == 0 for m in models) else 1)'
    ;;
  stop)
    if [ -f "$PIDFILE" ] && kill "$(cat "$PIDFILE")" 2>/dev/null; then echo "stopped"; else echo "not running"; fi
    rm -f "$PIDFILE"
    ;;
  *)
    echo "usage: bash vm/ollama_cpu.sh start|check|stop [model]"; exit 2 ;;
esac
