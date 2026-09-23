import os
import sys
import subprocess
from pathlib import Path
import modal

# Ensure UTF-8 output on Windows consoles to prevent UnicodeEncodeError with Modal symbols
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# 1. Configuration & App Setup
# ─────────────────────────────────────────────────────────────────────────────

APP_NAME = "wangp"
app = modal.App(APP_NAME)

# Choose default GPU: "A100" (40GB), "A100-80GB", "A10G" (24GB), "H100", or "A100:2"
# Can be overridden via environment variable MODAL_GPU:
# e.g.: $env:MODAL_GPU="A10G"; modal serve modal_app.py
DEFAULT_GPU = os.environ.get("MODAL_GPU", "A100")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Persistent Storage (Modal Volumes)
# ─────────────────────────────────────────────────────────────────────────────

# Persistent volume for /kaggle/tmp
kaggle_tmp_vol = modal.Volume.from_name("wangp-kaggle-tmp", create_if_missing=True)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Custom Container Image
# ─────────────────────────────────────────────────────────────────────────────

cuda_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install(
        "build-essential",
        "clang",
        "g++",
        "git",
        "ffmpeg",
        "libgl1",
        "libglib2.0-0",
        "cmake",
        "ninja-build",
        "wget",
        "curl",
    )
    .env({
        "DEBIAN_FRONTEND": "noninteractive",
        "PYTHONUNBUFFERED": "1",
        "HF_HOME": "/root/.cache/huggingface",
        "TORCH_ALLOW_TF32_CUBLAS": "1",
        "TORCH_ALLOW_TF32_CUDNN": "1",
        "SDL_AUDIODRIVER": "dummy",
        "NUMBA_THREADING_LAYER": "workqueue",
        "GRADIO_ANALYTICS_ENABLED": "False",
        "TORCH_CUDA_ARCH_LIST": "8.0;8.6;8.9;9.0",
        "FORCE_CUDA": "1",
        "MAX_JOBS": "8",
    })
    .run_commands(
        "pip install --upgrade pip setuptools wheel cython",
        "pip install torch==2.10.0+cu128 torchvision==0.25.0+cu128 torchaudio==2.10.0+cu128 --index-url https://download.pytorch.org/whl/cu128",
    )
    .add_local_file(Path(__file__).parent / "requirements.txt", "/tmp/requirements.txt", copy=True)
    .run_commands(
        "pip install -r /tmp/requirements.txt",
    )
    # Build SageAttention for Ampere/Ada/Hopper compute architectures (falls back gracefully if skipped)
    .run_commands(
        "git clone https://github.com/thu-ml/SageAttention.git /tmp/sageattention && "
        "cd /tmp/sageattention && "
        "python3 -c 'import os; content = open(\"setup.py\").read(); "
        "old = \"compute_capabilities = set()\\ndevice_count = torch.cuda.device_count()\\nfor i in range(device_count):\\n    major, minor = torch.cuda.get_device_capability(i)\\n    if major < 8:\\n        warnings.warn(f\\\"skipping GPU {i} with compute capability {major}.{minor}\\\")\\n        continue\\n    compute_capabilities.add(f\\\"{major}.{minor}\\\")\"; "
        "new = \"compute_capabilities = {\\\"8.0\\\", \\\"8.6\\\", \\\"8.9\\\", \\\"9.0\\\"}\"; "
        "open(\"setup.py\", \"w\").write(content.replace(old, new))' && "
        "pip install --no-build-isolation . || echo '[Warn] SageAttention installation skipped; PyTorch SDPA will be used.'"
    )
    .add_local_dir(
        Path(__file__).parent,
        remote_path="/workspace",
        copy=True,
        ignore=[
            "**/__pycache__/**",
            "*.pyc",
            "**/.git/**",
            "repo.zip",
            "temp_extract/**",
        ],
    )
)

# ─────────────────────────────────────────────────────────────────────────────
# 4. Web Server Endpoint (Gradio UI on Modal)
# ─────────────────────────────────────────────────────────────────────────────

@app.function(
    image=cuda_image,
    gpu=DEFAULT_GPU,
    volumes={
        "/kaggle/tmp": kaggle_tmp_vol,
    },
    timeout=3600,
    max_containers=1,
    scaledown_window=300,
)
@modal.concurrent(max_inputs=100)
@modal.web_server(port=7860, startup_timeout=600)
def ui():
    """Starts the WanGP Gradio server inside the container on port 7860."""
    os.chdir("/workspace")
    os.makedirs("/kaggle/tmp", exist_ok=True)

    # Clean up any leftover lock file from unexpected previous stops
    if os.path.exists("/workspace/startup.lock"):
        try:
            os.remove("/workspace/startup.lock")
        except Exception:
            pass

    # Profile 1 is HighRAM_HighVRAM (ideal for A100 with 24GB+ VRAM)
    # Pass custom args using WANGP_ARGS or WAN2GP_ARGS environment variable if desired
    extra_args = os.environ.get("WANGP_ARGS", os.environ.get("WAN2GP_ARGS", "--profile 1 --advanced"))
    cmd = f"python3 -u wgp.py --listen --server-port 7860 {extra_args}"
    print(f"[WanGP Modal] Launching server: {cmd}")
    subprocess.Popen(cmd, shell=True, cwd="/workspace")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Remote Diagnostics / GPU Verification Function
# ─────────────────────────────────────────────────────────────────────────────

@app.function(
    image=cuda_image,
    gpu=DEFAULT_GPU,
    timeout=120,
)
def check_gpu():
    """Verify PyTorch and GPU availability on Modal."""
    import torch

    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "devices": [],
    }
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info["devices"].append({
                "index": i,
                "name": props.name,
                "vram_gb": round(props.total_memory / (1024**3), 2),
                "major_minor": f"{props.major}.{props.minor}",
            })
    return info


# ─────────────────────────────────────────────────────────────────────────────
# 6. Local CLI Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(check: bool = False):
    """
    Local CLI entrypoint:
      modal run modal_app.py            -> Print usage and instructions
      modal run modal_app.py --check    -> Test remote GPU execution on Modal
      modal serve modal_app.py          -> Launch interactive Web UI tunnel
      modal deploy modal_app.py         -> Permanently deploy the Web UI
    """
    if check:
        print("[WanGP] Testing remote GPU environment on Modal...")
        result = check_gpu.remote()
        print("\n--- Remote GPU Check Result ---")
        print(f"CUDA Available: {result['cuda_available']}")
        print(f"Device Count:   {result['device_count']}")
        for dev in result["devices"]:
            print(f"  GPU [{dev['index']}]: {dev['name']} ({dev['vram_gb']} GB VRAM, SM {dev['major_minor']})")
        print("-------------------------------\n")
        return

    print("=" * 65)
    print("  WanGP on Modal.com")
    print("=" * 65)
    print(f"Configured GPU: {DEFAULT_GPU}")
    print("\nCommands to run:")
    print("  1. Interactive Web UI (Live URL with hot-reloading):")
    print("     modal serve modal_app.py")
    print("\n  2. Test GPU Connection on Modal:")
    print("     modal run modal_app.py --check")
    print("\n  3. Deploy permanently to your Modal account:")
    print("     modal deploy modal_app.py")
    print("\nTo select a different GPU, set $env:MODAL_GPU before running:")
    print("  - Single A10G (24GB VRAM, low cost):  $env:MODAL_GPU='A10G'")
    print("  - Single A100 (40GB VRAM, standard):  $env:MODAL_GPU='A100'")
    print("  - Dual A100 (Multi-GPU):              $env:MODAL_GPU='A100:2'")
    print("  - Single H100 (80GB VRAM, fastest):   $env:MODAL_GPU='H100'")
    print("=" * 65)
