"""Hardware-agnostic device and dtype resolution."""

from __future__ import annotations

import platform
from dataclasses import dataclass
from importlib import metadata
from typing import Any

from scriba.errors import DeviceUnavailableError


@dataclass
class GPUInfo:
    index: int
    name: str
    total_memory_gb: float
    capability: tuple[int, int]
    bf16: bool


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def list_gpus() -> list[GPUInfo]:
    try:
        import torch
    except ImportError:
        return []
    if not torch.cuda.is_available():
        return []
    gpus = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        cap = torch.cuda.get_device_capability(i)
        gpus.append(
            GPUInfo(
                index=i,
                name=props.name,
                total_memory_gb=round(props.total_memory / 1024**3, 1),
                capability=cap,
                bf16=cap[0] >= 8,
            )
        )
    return gpus


def resolve_device(requested: str) -> str:
    """'auto' → 'cuda:0' when a CUDA GPU exists, else 'cpu'. Validates explicit values."""
    req = (requested or "auto").strip().lower()
    try:
        import torch
    except ImportError as exc:
        raise DeviceUnavailableError("PyTorch is not installed") from exc

    if req == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if req == "cpu":
        return "cpu"
    if req.startswith("cuda"):
        if not torch.cuda.is_available():
            raise DeviceUnavailableError(f"Device '{requested}' requested but CUDA is not available")
        index = int(req.split(":", 1)[1]) if ":" in req else 0
        if index >= torch.cuda.device_count():
            raise DeviceUnavailableError(
                f"Device '{requested}' not found ({torch.cuda.device_count()} CUDA device(s) available)"
            )
        return f"cuda:{index}"
    if req == "mps":
        if not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
            raise DeviceUnavailableError("MPS requested but not available")
        return "mps"
    raise DeviceUnavailableError(f"Unknown device '{requested}'")


def resolve_dtype_name(requested: str, device: str) -> str:
    """'auto' → bfloat16 on GPUs that support it, float16 on older GPUs, float32 on CPU."""
    if requested != "auto":
        return requested
    if device.startswith("cuda"):
        import torch

        index = int(device.split(":", 1)[1]) if ":" in device else 0
        return "bfloat16" if torch.cuda.get_device_capability(index)[0] >= 8 else "float16"
    if device == "mps":
        return "float16"
    return "float32"


def torch_dtype(name: str) -> Any:
    import torch

    aliases = {
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
        "float16": torch.float16, "fp16": torch.float16, "half": torch.float16,
        "float32": torch.float32, "fp32": torch.float32,
    }
    return aliases[name.strip().lower()]


def device_display_name(device: str) -> str:
    if device.startswith("cuda"):
        import torch

        index = int(device.split(":", 1)[1]) if ":" in device else 0
        return torch.cuda.get_device_name(index)
    if device == "cpu":
        return f"CPU ({platform.processor() or platform.machine()})"
    return device


def vllm_supported() -> tuple[bool, str]:
    if platform.system() == "Windows":
        return False, "vLLM does not run on native Windows"
    if package_version("vllm") is None:
        return False, "vllm is not installed"
    return True, "available"


def runtime_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": package_version("torch"),
        "transformers": package_version("transformers"),
        "qwen-asr": package_version("qwen-asr"),
        "vllm": package_version("vllm"),
        "pyannote.audio": package_version("pyannote.audio"),
        "cuda": None,
    }
    try:
        import torch

        versions["cuda"] = torch.version.cuda
    except ImportError:
        pass
    return versions
