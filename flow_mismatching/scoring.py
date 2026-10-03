"""Flow Mismatching anomaly scoring.

Given a velocity network v trained by flow matching on normal images, a test image y is
scored by probing v along K affine paths x_t = (1 - t) x_0 + t y, x_0 ~ N(0, I):

    delta_t(i) = agg_k || v(x_t^k, t)_i - (y - x_0^k)_i ||^2     (summed over channels)
    H(i)       = sqrt( 1/T * sum_t w(t) delta_t(i) ),   w(t) = t^r
    S          = lam * max_i H(i) + (1 - lam) * mean of the top 1% of H

where agg is the minimum (default) or the mean over the K paths, and the T time steps are
evenly spaced inside (t0, t1). H is smoothed with a Gaussian kernel before computing S.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.inference_mode()
def anomaly_heatmap(
    model: nn.Module,
    image: torch.Tensor,
    num_paths: int = 10,
    num_steps: int = 10,
    power: float = 2.0,
    aggregation: str = "min",
    class_idx: Optional[int] = None,
    t_range: Tuple[float, float] = (0.0, 1.0),
    amp_dtype: Optional[torch.dtype] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Pixel-wise velocity mismatch heatmap of a single image.

    Args:
        model: velocity network, called as ``model(x, t, extra={"label": ...})``.
        image: normalized image of shape [1, C, H, W].
        num_paths: K, the number of Gaussian noise samples (paths).
        num_steps: T, the number of evenly spaced time steps strictly inside ``t_range``.
        power: r in the time weight w(t) = t^r.
        aggregation: "min" or "mean" over the K paths.
        class_idx: class label for a class-conditional model, None otherwise.
        t_range: (t0, t1) with 0 <= t0 < t1 <= 1; the endpoints themselves are excluded.
        amp_dtype: autocast dtype for the forward passes on CUDA, None for full precision.
            The mismatch itself is always computed in float32.
        generator: random generator for the noise (the global one if None).

    Returns:
        Heatmap of shape [H, W] in float32.
    """
    if image.dim() != 4 or image.shape[0] != 1:
        raise ValueError(f"Expected a single image of shape [1, C, H, W], got {tuple(image.shape)}")
    if aggregation not in ("min", "mean"):
        raise ValueError(f"aggregation must be 'min' or 'mean', got {aggregation!r}")
    t0, t1 = t_range
    if not 0.0 <= t0 < t1 <= 1.0:
        raise ValueError(f"Require 0 <= t0 < t1 <= 1, got t_range={t_range}")
    if num_paths < 1 or num_steps < 1:
        raise ValueError("num_paths and num_steps must be positive")

    model.eval()
    device = image.device
    _, c, h, w = image.shape
    use_amp = amp_dtype is not None and device.type == "cuda"

    # T interior points of an evenly spaced grid on [t0, t1].
    times = torch.linspace(t0, t1, steps=num_steps + 2, device=device, dtype=torch.float32)[1:-1]

    x0 = torch.randn(num_paths, c, h, w, device=device, dtype=image.dtype, generator=generator)
    x1 = image.expand(num_paths, -1, -1, -1)
    target = (x1 - x0).float()  # geometric velocity, constant along each path

    heat = torch.zeros(h, w, device=device, dtype=torch.float32)
    for t in times:
        xt = (1.0 - t) * x0 + t * x1
        t_batch = torch.full((num_paths,), float(t), device=device, dtype=torch.float32)
        extra = {}
        if class_idx is not None:
            extra["label"] = torch.full((num_paths,), int(class_idx), device=device, dtype=torch.long)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            velocity = model(xt, t_batch, extra=extra)

        mismatch = (velocity.float() - target).pow(2).sum(dim=1)  # [K, H, W]
        if aggregation == "min":
            mismatch = mismatch.min(dim=0).values
        else:
            mismatch = mismatch.mean(dim=0)
        heat += t.pow(power) * mismatch

    return (heat / num_steps).sqrt()


def gaussian_kernel(ksize: int, sigma: float) -> torch.Tensor:
    """Normalized 2D Gaussian kernel of shape [ksize, ksize]."""
    ax = torch.arange(ksize, dtype=torch.float32) - ksize // 2
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def gaussian_smooth(heatmap: torch.Tensor, ksize: int = 9, sigma: float = 3.0) -> torch.Tensor:
    """Smooth an [H, W] heatmap with a Gaussian kernel (zero padding, same output size).

    An even ``ksize`` is rounded up to the next odd number. ``ksize <= 1`` or
    ``sigma <= 0`` disables smoothing.
    """
    ksize = int(ksize)
    if ksize % 2 == 0:
        ksize += 1
    if ksize <= 1 or sigma <= 0:
        return heatmap
    if heatmap.dim() != 2:
        raise ValueError(f"Expected a heatmap of shape [H, W], got {tuple(heatmap.shape)}")

    h = heatmap.float()
    kernel = gaussian_kernel(ksize, float(sigma)).to(device=h.device, dtype=h.dtype)
    out = F.conv2d(h[None, None], kernel[None, None], padding=ksize // 2)[0, 0]
    return out.to(dtype=heatmap.dtype)


def image_score(heatmap: torch.Tensor, topk_ratio: float = 0.01, lam: float = 0.5) -> float:
    """Image-level anomaly score: lam * max(H) + (1 - lam) * mean of the top-k pixels of H.

    k = ceil(topk_ratio * number of pixels), i.e. 164 pixels for a 128x128 heatmap.
    Returns 0.0 if the heatmap contains non-finite values.
    """
    flat = heatmap.float().flatten()
    k = min(flat.numel(), max(1, math.ceil(topk_ratio * flat.numel())))
    max_value = float(flat.max())
    topk_mean = float(torch.topk(flat, k).values.mean())
    if not (math.isfinite(max_value) and math.isfinite(topk_mean)):
        return 0.0
    return lam * max_value + (1 - lam) * topk_mean
