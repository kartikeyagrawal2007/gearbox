#!/usr/bin/env bash
# Set up Gearbox on a Windows + WSL2 (Ubuntu) machine with an NVIDIA GPU.
# Run from the repo root inside WSL:  bash vm/setup_wsl.sh
# Safe to re-run: every step checks before it installs.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUMMARY=()
note() { echo; echo "== $*"; }
ok()   { echo "   OK   $*"; SUMMARY+=("OK   $*"); }
warn() { echo "   WARN $*"; SUMMARY+=("WARN $*"); }
fail() { echo "   FAIL $*"; SUMMARY+=("FAIL $*"); print_summary; exit 1; }
print_summary() {
  echo
  echo "=================== PASTE EVERYTHING BELOW THIS LINE ==================="
  printf '%s\n' "${SUMMARY[@]}"
  echo "========================================================================"
}

note "1/7 Checking this is WSL2 with the GPU visible"
grep -qi microsoft /proc/version && ok "running inside WSL ($(. /etc/os-release && echo "$PRETTY_NAME"))" \
  || warn "not WSL; continuing as plain Linux"
if command -v nvidia-smi >/dev/null 2>&1 && GPU=$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null); then
  ok "GPU: $GPU"
else
  fail "nvidia-smi does not work in WSL. Update the NVIDIA driver on Windows (do NOT install a Linux driver inside WSL), then run 'wsl --shutdown' in PowerShell and retry."
fi
MEM_GB=$(free -g | awk '/^Mem:/ {print $2}')
if [ "${MEM_GB:-0}" -ge 90 ]; then ok "RAM visible to WSL: ${MEM_GB} GB"
else warn "RAM visible to WSL: ${MEM_GB} GB (WSL defaults to half the host's RAM; see vm/README.md to raise it to ~100 GB)"; fi
DISK_GB=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc '0-9')
if [ "${DISK_GB:-0}" -ge 60 ]; then ok "free disk in \$HOME: ${DISK_GB} GB"
else warn "free disk in \$HOME: ${DISK_GB} GB (the full model ladder needs ~40 GB)"; fi
ok "CPU threads: $(nproc)"

note "2/7 Installing system packages (asks for your sudo password)"
sudo apt-get update -qq && sudo apt-get install -y -qq python3-venv python3-pip git curl zstd build-essential >/dev/null \
  && ok "apt packages present" || fail "apt-get install failed (see output above)"
ok "python: $(python3 --version)"

note "3/7 Creating the Python environment"
cd "$REPO" || fail "cannot cd to $REPO"
[ -d .venv ] || python3 -m venv .venv || fail "python3 -m venv failed"
.venv/bin/pip install -q --upgrade pip && .venv/bin/pip install -q -e ".[dev,gpu]" \
  && ok "gearbox installed ($(git log --oneline -1))" || fail "pip install failed (see output above)"

note "4/7 Installing and starting Ollama"
if ! command -v ollama >/dev/null 2>&1; then
  echo "   Ollama not found; running the official installer from ollama.com"
  curl -fsSL https://ollama.com/install.sh | sh || fail "Ollama install failed"
fi
ok "ollama: $(ollama --version 2>/dev/null | tail -1)"
# Benchmarks need several models resident at once and no reloads mid-run.
if [ "$(ps -p 1 -o comm=)" = "systemd" ] && systemctl list-unit-files ollama.service >/dev/null 2>&1; then
  sudo mkdir -p /etc/systemd/system/ollama.service.d
  printf '[Service]\nEnvironment="OLLAMA_NUM_PARALLEL=4"\nEnvironment="OLLAMA_MAX_LOADED_MODELS=3"\nEnvironment="OLLAMA_KEEP_ALIVE=30m"\n' \
    | sudo tee /etc/systemd/system/ollama.service.d/gearbox.conf >/dev/null
  sudo systemctl daemon-reload && sudo systemctl enable --now ollama >/dev/null 2>&1 && sudo systemctl restart ollama
  ok "ollama runs as a systemd service (parallel=4, max loaded=3, keep alive 30m)"
else
  if ! curl -s --max-time 2 localhost:11434/api/version >/dev/null; then
    OLLAMA_NUM_PARALLEL=4 OLLAMA_MAX_LOADED_MODELS=3 OLLAMA_KEEP_ALIVE=30m nohup ollama serve > "$HOME/ollama.log" 2>&1 &
  fi
  warn "no systemd: started 'ollama serve' in the background; re-run this script after a WSL restart"
fi
for _ in $(seq 1 20); do curl -s --max-time 2 localhost:11434/api/version >/dev/null && break; sleep 1; done
curl -s --max-time 2 localhost:11434/api/version >/dev/null && ok "ollama API answers on localhost:11434" \
  || fail "ollama API not reachable on localhost:11434 (is Ollama for Windows also running? quit it and retry)"

note "5/7 Proving GPU inference with a tiny model (qwen2.5-coder:0.5b, ~400 MB)"
ollama pull qwen2.5-coder:0.5b >/dev/null 2>&1 || fail "could not pull qwen2.5-coder:0.5b"
REPLY=$(ollama run qwen2.5-coder:0.5b "Reply with the single word: ready" 2>/dev/null | head -c 80)
PROC=$(ollama ps | awk 'NR==2 {for (i=1;i<=NF;i++) if ($i ~ /GPU|CPU/) {print $(i-1), $i; exit}}')
case "$PROC" in
  *100%*GPU*) ok "model ran on the GPU ($PROC); reply: ${REPLY//$'\n'/ }" ;;
  *) warn "model placement: '${PROC:-unknown}' (expected 100% GPU); reply: ${REPLY//$'\n'/ }" ;;
esac

note "6/7 Running the test suite"
TESTS=$(.venv/bin/python -m pytest -q 2>&1 | tail -1)
case "$TESTS" in *failed*|*error*) warn "tests: $TESTS (run '.venv/bin/python -m pytest -q' for details)" ;; *) ok "tests: $TESTS" ;; esac

note "7/7 Checking measurement and safety features"
ISO=$(.venv/bin/python -c "from gearbox.verify import isolation_mode; print(isolation_mode())" 2>&1)
case "$ISO" in *"no network"*) ok "check isolation: $ISO" ;; *) warn "check isolation: $ISO" ;; esac
ENERGY=$(.venv/bin/python -c "from gearbox.cost.energy import EnergyMeter; m = EnergyMeter(); print(m.available, m.read_mj())" 2>&1 | tail -1)
case "$ENERGY" in True*) ok "GPU energy counter readable via NVML ($ENERGY mJ)" ;;
  *) warn "GPU energy counter not available ($ENERGY); energy figures will come from power sampling instead" ;; esac

print_summary
