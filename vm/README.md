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

## Updating later

```text
cd ~/gearbox && git pull && .venv/bin/pip install -q -e ".[dev,gpu]"
```
