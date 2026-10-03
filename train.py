"""Train the flow matching velocity network on the normal training images of MVTec-AD or VisA.

A single class-conditional model is trained on all classes of a dataset.

    # one GPU
    python train.py --dataset mvtec --data_root data/mvtec_anomaly_detection

    # several GPUs (DistributedDataParallel)
    torchrun --nproc_per_node 4 train.py --dataset mvtec --data_root data/mvtec_anomaly_detection
"""

import argparse
import contextlib
import importlib.util
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from flow_mismatching.data import NORM_STATS, TrainDataset, discover_classes
from flow_mismatching.optim import StableAdamW
from flow_mismatching.unet import UNetModel
from flow_mismatching.utils import denormalize, enable_tf32, latest_checkpoint, resolve_amp_dtype


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=sorted(NORM_STATS), help="dataset (sets the normalization)")
    parser.add_argument("--data_root", required=True, help="dataset root in the MVTec-AD folder layout")
    parser.add_argument("--output_dir", default=None, help="checkpoints and samples (default: runs/<dataset>)")
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--class_cond", action=argparse.BooleanOptionalAction, default=True,
                        help="condition the model on the class label")

    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=32, help="batch size per GPU")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.99))
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--update_clip", type=float, default=1.0, help="StableAdamW update-RMS clip; <= 0 disables")
    parser.add_argument("--grad_clip", type=float, default=0.0, help="gradient-norm clip; <= 0 disables")
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--amp", choices=["bf16", "fp16", "off"], default="bf16", help="mixed precision")
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True, help="allow TF32 on Ampere+")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--save_every", type=int, default=500, help="save a checkpoint every N epochs")
    parser.add_argument("--resume", action="store_true", help="resume from the latest checkpoint in output_dir")
    parser.add_argument("--num_samples", type=int, default=36, help="images generated at each checkpoint; 0 disables")
    parser.add_argument("--ode_method", default="heun3", help="ODE solver for the samples (a torchdiffeq method)")
    parser.add_argument("--ode_steps", type=int, default=100, help="ODE steps for the samples")
    return parser.parse_args()


def setup_device():
    """Return (device, rank, world_size); initializes the process group when launched with torchrun."""
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank), dist.get_rank(), dist.get_world_size()
    return torch.device("cuda" if torch.cuda.is_available() else "cpu"), 0, 1


@torch.inference_mode()
def save_samples(model, path, num_samples, img_size, num_classes, mean, std, ode_method, ode_steps, device, amp_dtype):
    """Generate images by integrating the learned ODE from noise (t = 0) to data (t = 1)."""
    from flow_matching.solver import ODESolver
    from torchvision.utils import save_image

    model.eval()
    labels = torch.arange(num_samples, device=device) % num_classes if num_classes else None

    def velocity(x, t):
        t = torch.as_tensor(t, device=x.device, dtype=x.dtype).reshape(-1).expand(x.shape[0])
        return model(x, t, extra={"label": labels} if labels is not None else {})

    x0 = torch.randn(num_samples, 3, img_size, img_size, device=device)
    with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
        x1 = ODESolver(velocity_model=velocity).sample(
            x_init=x0,
            step_size=1.0 / ode_steps,
            method=ode_method,
            time_grid=torch.tensor([0.0, 1.0], device=device),
        )
    save_image(denormalize(x1.float(), mean, std), path, nrow=int(math.sqrt(num_samples)))


def main():
    args = parse_args()
    device, rank, world_size = setup_device()
    distributed = dist.is_initialized()
    is_main = rank == 0
    torch.manual_seed(args.seed + rank)

    if args.num_samples > 0 and importlib.util.find_spec("flow_matching") is None:
        raise SystemExit("Generating samples needs `pip install flow_matching` (or pass --num_samples 0).")

    if args.tf32 and device.type == "cuda":
        enable_tf32()
    amp_dtype = resolve_amp_dtype(args.amp, device)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)

    # Data: normal images of all classes.
    mean, std = NORM_STATS[args.dataset]
    class_names = discover_classes(args.data_root, "train")
    dataset = TrainDataset(args.data_root, class_names, args.img_size, mean, std)
    num_classes = len(class_names) if args.class_cond else None
    sampler = DistributedSampler(dataset, shuffle=True, seed=args.seed) if distributed else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    # Model and optimizer.
    net = UNetModel(num_classes=num_classes).to(device)
    model = DDP(net, device_ids=[device.index]) if distributed else net
    optimizer = StableAdamW(
        net.parameters(),
        lr=args.lr,
        betas=tuple(args.betas),
        eps=args.eps,
        weight_decay=args.weight_decay,
        amsgrad=True,
        update_clip=args.update_clip,
    )

    out_dir = Path(args.output_dir or Path("runs") / args.dataset)
    out_dir.mkdir(parents=True, exist_ok=True)
    start_epoch = 0
    ckpt_path = latest_checkpoint(out_dir) if args.resume else None
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        if args.class_cond and ckpt.get("class_names") != class_names:
            raise ValueError(f"Classes in {ckpt_path} ({ckpt.get('class_names')}) differ from {class_names}")
        net.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if scaler.is_enabled() and ckpt.get("scaler"):
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt["epoch"])

    if is_main:
        num_params = sum(p.numel() for p in net.parameters()) / 1e6
        print(f"{len(dataset)} training images from {len(class_names)} classes: {', '.join(class_names)}")
        print(f"U-Net: {num_params:.1f}M parameters, {'class-conditional' if num_classes else 'unconditional'}")
        print(f"{world_size} process(es), batch size {args.batch_size} per process, AMP: {amp_dtype or 'off'}")
        if ckpt_path is not None:
            print(f"Resuming from {ckpt_path} at epoch {start_epoch}")

    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, args.epochs):
        model.train()
        if sampler is not None:
            sampler.set_epoch(epoch)
        loss_sum = torch.zeros((), device=device)
        progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}", leave=False, disable=not is_main)
        for step, (x1, labels) in enumerate(progress):
            x1 = x1.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Conditional flow matching on the affine path x_t = (1 - t) x_0 + t x_1,
            # whose velocity is x_1 - x_0.
            t = torch.rand(x1.shape[0], device=device)
            x0 = torch.randn_like(x1)
            t_ = t.view(-1, 1, 1, 1)
            xt = (1.0 - t_) * x0 + t_ * x1
            extra = {"label": labels} if args.class_cond else {}

            # Gradients are synchronized across processes only on update steps.
            update = (step + 1) % args.accumulation_steps == 0 or step + 1 == len(loader)
            with contextlib.nullcontext() if update or not distributed else model.no_sync():
                with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                    loss = F.mse_loss(model(xt, t, extra=extra), x1 - x0)
                scaler.scale(loss / args.accumulation_steps).backward()
            loss_sum += loss.detach().float()

            if update:
                scaler.unscale_(optimizer)
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        epoch_loss = loss_sum / max(len(loader), 1)
        if distributed:
            dist.all_reduce(epoch_loss)
            epoch_loss /= world_size
        if is_main:
            print(f"Epoch {epoch + 1}/{args.epochs} | loss {epoch_loss.item():.4f}")

        if is_main and ((epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs):
            ckpt_path = out_dir / f"ckpt_epoch{epoch + 1:04d}.pt"
            torch.save(
                {
                    "model": net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict() if scaler.is_enabled() else None,
                    "epoch": epoch + 1,
                    "dataset": args.dataset,
                    "img_size": args.img_size,
                    "norm_mean": list(mean),
                    "norm_std": list(std),
                    "class_cond": args.class_cond,
                    "num_classes": num_classes,
                    "class_names": class_names if args.class_cond else None,
                    "args": vars(args),
                },
                ckpt_path,
            )
            print(f"Saved {ckpt_path}")
            if args.num_samples > 0:
                sample_path = out_dir / f"samples_epoch{epoch + 1:04d}.png"
                save_samples(
                    net, sample_path, args.num_samples, args.img_size, num_classes, mean, std,
                    args.ode_method, args.ode_steps, device, amp_dtype,
                )
                print(f"Saved {sample_path}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
