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

```bash
git clone https://github.com/kartikeyagrawal2007/gearbox.git ~/gearbox && cd ~/gearbox && bash vm/setup_wsl.sh
```

The script checks the GPU, RAM and disk; installs packages (it asks for your sudo password), the Python environment and Ollama; and proves GPU inference with a 0.4 GB model. It then runs the tests and finishes with a summary block to paste back.

**On a network that drops connections**, re-run the script and it resumes the 1.4 GB Ollama download where it stopped. If it keeps failing, download the file in a Windows browser instead: <https://ollama.com/download/ollama-linux-amd64.tar.zst>. Copy it into WSL, then re-run the script; it finds the file and skips the download:

```bash
cp /mnt/c/Users/$USER_WINDOWS/Downloads/ollama-linux-amd64.tar.zst ~/ && cd ~/gearbox && bash vm/setup_wsl.sh
```

(Replace `$USER_WINDOWS` with your Windows user name, e.g. `PRO-LAB-2`.)

## 2. Pull the model ladder

Six sizes of one model family, about 37 GB in total:

```bash
cd ~/gearbox && bash vm/pull_models.sh
```

## 3. First experiment: false-done rate vs model size

```bash
cd ~/gearbox && mkdir -p runs && .venv/bin/python bench/false_done.py --config vm/gearbox.vm.yaml --json runs/false_done_ladder.json
```

## Updating later

```bash
cd ~/gearbox && git pull && .venv/bin/pip install -q -e ".[dev,gpu]"
```
