"""Checkpoint, mixed-precision and visualization helpers."""

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import torch

from .unet import UNetModel

AMP_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "off": None}


def resolve_amp_dtype(name: str, device: torch.device) -> Optional[torch.dtype]:
    """Autocast dtype for ``name`` in {"bf16", "fp16", "off"}; None on CPU or when off.

    Falls back from bf16 to fp16 on GPUs without bf16 support.
    """
    dtype = AMP_DTYPES[name]
    if device.type != "cuda" or dtype is None:
        return None
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        print("bf16 is not supported on this GPU, using fp16 instead.")
        return torch.float16
    return dtype


def enable_tf32() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def latest_checkpoint(run_dir: Path) -> Optional[Path]:
    checkpoints = sorted(Path(run_dir).glob("ckpt_epoch*.pt"))
    return checkpoints[-1] if checkpoints else None


def load_model(ckpt_path: str, device: torch.device) -> Tuple[UNetModel, Dict[str, Any]]:
    """Build the U-Net from a training checkpoint and load its weights.

    The number of classes is read from the label embedding, so this works for
    class-conditional and unconditional checkpoints, saved with or without DDP.

    Returns:
        The model in eval mode and the checkpoint metadata (everything except the weights
        and optimizer states).
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state = ckpt["model"] if "model" in ckpt else ckpt
    if all(k.startswith("module.") for k in state):
        state = {k[len("module.") :]: v for k, v in state.items()}

    num_classes = state["label_emb.weight"].shape[0] - 1 if "label_emb.weight" in state else None
    model = UNetModel(num_classes=num_classes)
    model.load_state_dict(state)
    model.to(device).eval()

    meta = {k: v for k, v in ckpt.items() if k not in ("model", "optimizer", "scaler")} if "model" in ckpt else {}
    meta["num_classes"] = num_classes
    return model, meta


def denormalize(x: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> torch.Tensor:
    """Undo ``transforms.Normalize`` for an image batch [B, 3, H, W] or image [3, H, W]; clamps to [0, 1]."""
    shape = (-1, 1, 1)
    mean = torch.as_tensor(mean, device=x.device, dtype=x.dtype).view(shape)
    std = torch.as_tensor(std, device=x.device, dtype=x.dtype).view(shape)
    return (x * std + mean).clamp(0, 1)


def save_visualization(image: torch.Tensor, heatmap: torch.Tensor, mask: torch.Tensor, path: Path) -> None:
    """Save a row of (image | heatmap | ground truth).

    Args:
        image: [3, H, W] in [0, 1]; heatmap: [H, W]; mask: [H, W] or [1, H, W].
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=150)
    axes[0].imshow(image.permute(1, 2, 0).cpu().numpy())
    axes[1].imshow(heatmap.cpu().numpy(), cmap="jet")
    axes[2].imshow(mask.squeeze().cpu().numpy(), cmap="gray")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
