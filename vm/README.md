# Running Gearbox on the A5000 VM (Windows + WSL2)

Everything runs inside **WSL2 Ubuntu**. The Windows NVIDIA driver already exposes the GPU to WSL, so never install a Linux NVIDIA driver inside WSL.

## 0. Windows side (PowerShell)

Check WSL and the GPU:

```powershell
wsl --status; wsl -l -v; nvidia-smi
```

- If WSL or Ubuntu is missing, open PowerShell **as Administrator** and run `wsl --install -d Ubuntu-24.04`. Reboot, then open "Ubuntu" from the Start menu once to create your Linux user.
- If **Ollama for Windows** is installed and running, quit it from the system tray. It would clash with the copy inside WSL on port 11434.

**Give WSL more RAM.** By default WSL only sees half of the machine's 128 GB, which is too little for large models on the CPU. First check whether a config already exists:

```powershell
Get-Content "$env:USERPROFILE\.wslconfig" -ErrorAction SilentlyContinue
```

If that prints nothing, create one, then restart WSL:

```powershell
Set-Content -Path "$env:USERPROFILE\.wslconfig" -Value "[wsl2]`nmemory=100GB`nswap=16GB"
wsl --shutdown
```

If it printed something, add `memory=100GB` under its `[wsl2]` section by hand instead.

## 1. Inside WSL (Ubuntu): clone and set up

```text
git clone https://github.com/kartikeyagrawal2007/gearbox.git ~/gearbox && cd ~/gearbox && bash vm/setup_wsl.sh
```

The script checks the GPU, RAM and disk; installs packages (it asks for your sudo password), the Python environment and Ollama; and proves GPU inference with a 0.4 GB model. It then runs the tests and finishes with a summary block to paste back.

**On a network that drops connections**, re-run the script and it resumes the 1.4 GB Ollama download where it stopped. If it keeps failing, download the file in a Windows browser instead: <https://ollama.com/download/ollama-linux-amd64.tar.zst>. Copy it into WSL, then re-run the script; it finds the file and skips the download:

```text
cp /mnt/c/Users/$USER_WINDOWS/Downloads/ollama-linux-amd64.tar.zst ~/ && cd ~/gearbox && bash vm/setup_wsl.sh
```

(Replace `$USER_WINDOWS` with your Windows user name, e.g. `PRO-LAB-2`.)

## 2. Pull the benchmark models

The set is one current family at five sizes plus three other vendors at matched sizes. The size ladder (Qwen3.5: 0.8B, 2B, 4B, 9B, 27B) isolates model size. Granite 4.2 (IBM), Ministral 3 (Mistral) and Gemma 3 (Google) at ~3-4B and ~8-12B show results aren't Qwen-specific. That's about 60 GB, ordered small-first so every family arrives early:

```text
cd ~/gearbox && bash vm/pull_models.sh
```

Use `core` (Qwen3.5 ladder only) or `xfamily` (the other vendors only) to pull a subset. Re-run the same command to resume after a dropped connection.

## Watch it in the browser

Open a second Ubuntu tab and start the dashboard with every benchmark model listed:

```text
cd ~/gearbox && .venv/bin/gearbox --config vm/models.vm.yaml ui
```

Then open <http://127.0.0.1:8790> in Firefox or Edge on the lab PC. `networkingMode=mirrored` shares WSL's localhost with Windows. It shows:
- which models have finished downloading (it refreshes every 15 s)
- the race, with host and worker picked from downloaded models
- a **Benchmark results** table built from everything in `runs/`, refreshed every 10 s

Keep that tab open while you use it.

## 3. Verify the models

Check that each downloaded model answers, with Qwen3.5's thinking switched off:

```text
cd ~/gearbox && .venv/bin/python vm/verify_models.py
```

## 4. First experiment: false-done rate by model

Run it with the hatch on (default) and with `--hatch off`:

```text
cd ~/gearbox && mkdir -p runs && .venv/bin/python bench/false_done.py --config vm/models.vm.yaml --json runs/false_done_all.json
```

## 5. HumanEval+ and MBPP+ across all models

These are the real numbers for the paper: HumanEval+ (164 problems) and MBPP+ (378 problems), both with hidden tests, per model, with the UNSURE hatch on and off. Results appear in the dashboard as each model finishes.

HumanEval+ (roughly 1–3 hours for both runs):

```text
cd ~/gearbox && .venv/bin/python bench/false_done.py --config vm/models.vm.yaml --tasks humaneval+ --json runs/humaneval_hatch_on.json && .venv/bin/python bench/false_done.py --config vm/models.vm.yaml --tasks humaneval+ --hatch off --json runs/humaneval_hatch_off.json
```

MBPP+ (roughly twice as long, since it has more problems):

```text
cd ~/gearbox && .venv/bin/python bench/false_done.py --config vm/models.vm.yaml --tasks mbpp+ --json runs/mbpp_hatch_on.json && .venv/bin/python bench/false_done.py --config vm/models.vm.yaml --tasks mbpp+ --hatch off --json runs/mbpp_hatch_off.json
```

The test data (about 1.3 MB) downloads on first use. If the network blocks it, download `HumanEvalPlus.jsonl.gz` from <https://github.com/evalplus/humanevalplus_release/releases/tag/v0.1.10> and `MbppPlus.jsonl.gz` from <https://github.com/evalplus/mbppplus_release/releases/tag/v0.2.0> in a browser, and copy them to `~/gearbox/bench/data/`.

## 6. Async delegation: does the host finish sooner if it doesn't wait?

`bench/async_bench.py` runs the same agent episode four ways: the host does everything itself, delegates and waits for each subtask, delegates everything and waits, or delegates everything and keeps working. The host is Qwen3.5 27B on the GPU. The worker is Qwen3.5 4B, either on the same GPU or on the CPU. Run the steps in order.

**a. Start the CPU-only Ollama and check that a model really runs there.** The check must print `OK on the CPU`:

```text
cd ~/gearbox && git pull -q && bash vm/ollama_cpu.sh start && bash vm/ollama_cpu.sh check
```

**b. Calibrate (about 8 episodes).** Read the time estimate it prints, and look at the `host GPU` column at the end: it must say `100%`. If the 27B spilled off the GPU when it had to share it with the 4B, the shared-GPU numbers aren't a fair test:

```text
cd ~/gearbox && .venv/bin/python bench/async_bench.py --k 4 --host-tokens 256 --reps 1 --out runs/async_calibration.jsonl
```

**c. The full run, in tmux, so a dropped connection doesn't stop it.** It writes one line per episode and skips finished ones, so if it stops, run the same command again:

```text
tmux new -d -s async "cd ~/gearbox && .venv/bin/python -u bench/async_bench.py --out runs/async.jsonl 2>&1 | tee -a ~/async.log"; sleep 5; tmux ls
```

Check progress any time with `tail -5 ~/async.log`, and print the tables with `.venv/bin/python bench/async_bench.py --summary runs/async.jsonl`. When it's done, stop the CPU Ollama with `bash vm/ollama_cpu.sh stop`.

## Updating later

```text
cd ~/gearbox && git pull && .venv/bin/pip install -q -e ".[dev,gpu]"
```
