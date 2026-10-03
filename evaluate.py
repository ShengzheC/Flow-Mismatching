"""Evaluate Flow Mismatching on MVTec-AD or VisA.

    python evaluate.py --dataset mvtec --data_root data/mvtec_anomaly_detection \
        --ckpt runs/mvtec/ckpt_epoch2000.pt

Writes to --output_dir:
    per_image.csv           image-level score and pixel metrics of every test image
    metrics_pooled.csv      per-class metrics, pixels pooled over each class, and their mean
    metrics_per_image.csv   per-class metrics, pixel metrics averaged over anomalous images
    config.json             the arguments of this run
    vis/<class>/*.png       image | heatmap | ground truth (with --save_vis)
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from flow_mismatching.data import NORM_STATS, TestDataset, discover_classes
from flow_mismatching.metrics import PIXEL_METRICS, class_summary, pixel_metrics
from flow_mismatching.scoring import anomaly_heatmap, gaussian_smooth, image_score
from flow_mismatching.utils import denormalize, enable_tf32, load_model, resolve_amp_dtype, save_visualization

METRIC_COLUMNS = [
    "image_auroc", "image_ap", "image_f1_max", "pixel_auroc", "pixel_ap", "pixel_f1_max", "pixel_aupro",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=sorted(NORM_STATS))
    parser.add_argument("--data_root", required=True, help="dataset root in the MVTec-AD folder layout")
    parser.add_argument("--ckpt", required=True, help="checkpoint saved by train.py")
    parser.add_argument("--output_dir", default=None, help="default: results/<dataset>")
    parser.add_argument("--classes", nargs="+", default=None, help="evaluate only these classes")
    parser.add_argument("--img_size", type=int, default=128)

    fm = parser.add_argument_group("Flow Mismatching")
    fm.add_argument("--num_paths", type=int, default=10, help="K, noise paths per image")
    fm.add_argument("--num_steps", type=int, default=10, help="T, time steps per path")
    fm.add_argument("--t0", type=float, default=0.0, help="time steps lie strictly inside (t0, t1)")
    fm.add_argument("--t1", type=float, default=1.0)
    fm.add_argument("--power", type=float, default=2.0, help="r in the time weight w(t) = t^r")
    fm.add_argument("--aggregation", choices=["min", "mean"], default="min", help="aggregation over the paths")
    fm.add_argument("--gauss_ksize", type=int, default=9, help="Gaussian smoothing kernel size; 1 disables it")
    fm.add_argument("--gauss_sigma", type=float, default=3.0, help="Gaussian smoothing sigma")
    fm.add_argument("--topk_ratio", type=float, default=0.01, help="fraction of top pixels in the image score")
    fm.add_argument("--lam", type=float, default=0.5, help="weight of the max pixel in the image score")

    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", choices=["bf16", "fp16", "off"], default="bf16", help="mixed precision")
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True, help="allow TF32 on Ampere+")
    parser.add_argument("--seed", type=int, default=0, help="seed of the noise paths, reset for every class")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--save_vis", action="store_true", help="save image | heatmap | ground truth figures")
    return parser.parse_args()


def evaluate_class(model, class_name, class_idx, args, mean, std, device, amp_dtype, out_dir):
    """Score every test image of one class. Returns (per-image rows, class summary)."""
    dataset = TestDataset(args.data_root, class_name, args.img_size, mean, std)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    rows, labels, scores, masks, heatmaps, per_image = [], [], [], [], [], []
    for image, mask, label, name in tqdm(loader, desc=class_name, leave=False):
        image = image.to(device)
        heatmap = anomaly_heatmap(
            model,
            image,
            num_paths=args.num_paths,
            num_steps=args.num_steps,
            power=args.power,
            aggregation=args.aggregation,
            class_idx=class_idx,
            t_range=(args.t0, args.t1),
            amp_dtype=amp_dtype,
            generator=generator,
        )
        heatmap = gaussian_smooth(heatmap, args.gauss_ksize, args.gauss_sigma)
        score = image_score(heatmap, args.topk_ratio, args.lam)

        heatmap_np = heatmap.cpu().numpy()
        mask_np = mask[0, 0].numpy().astype(np.uint8)
        metrics = pixel_metrics(mask_np, heatmap_np)

        labels.append(int(label))
        scores.append(score)
        masks.append(mask_np)
        heatmaps.append(heatmap_np)
        per_image.append(metrics)
        rows.append({"class": class_name, "name": name[0], "label": int(label), "score": score,
                     **{f"pixel_{k}": metrics[k] for k in PIXEL_METRICS}})

        if args.save_vis:
            save_visualization(
                denormalize(image[0], mean, std), heatmap, mask[0], out_dir / "vis" / class_name / f"{name[0]}.png"
            )

    summary = class_summary(labels, scores, np.stack(masks), np.stack(heatmaps), per_image)
    return rows, summary


def write_csv(path, columns, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: f"{v:.6f}" if isinstance(v, float) else v for k, v in row.items()})


def print_table(title, rows):
    header = f"{'class':<14}" + "".join(f"{c.replace('_f1_max', '_f1'):>13}" for c in METRIC_COLUMNS)
    print(f"\n{title}\n{header}\n{'-' * len(header)}")
    for row in rows:
        print(f"{row['class']:<14}" + "".join(f"{100 * row[c]:>13.1f}" for c in METRIC_COLUMNS))


def main():
    args = parse_args()
    device = torch.device(args.device)
    if args.tf32 and device.type == "cuda":
        enable_tf32()
    amp_dtype = resolve_amp_dtype(args.amp, device)
    out_dir = Path(args.output_dir or Path("results") / args.dataset)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, meta = load_model(args.ckpt, device)
    mean = meta.get("norm_mean") or NORM_STATS[args.dataset][0]
    std = meta.get("norm_std") or NORM_STATS[args.dataset][1]
    train_size = meta.get("img_size") or meta.get("args", {}).get("img_size")
    if train_size is not None and int(train_size) != args.img_size:
        print(f"Warning: the model was trained at {train_size}x{train_size}, evaluating at {args.img_size}.")

    # Class labels must follow the order used in training, which is stored in the checkpoint.
    available = discover_classes(args.data_root, "test")
    num_classes = meta["num_classes"]
    if num_classes is None:
        class_to_idx = None
        classes = available
    else:
        class_names = meta.get("class_names") or available
        if len(class_names) != num_classes:
            raise ValueError(f"The checkpoint has {num_classes} classes but {len(class_names)} class names were found")
        class_to_idx = {c: i for i, c in enumerate(class_names)}
        classes = [c for c in class_names if c in available]
    if args.classes:
        unknown = sorted(set(args.classes) - set(classes))
        if unknown:
            raise ValueError(f"Unknown classes {unknown}; available: {classes}")
        classes = args.classes

    print(f"Loaded {args.ckpt}" + (f" (epoch {meta['epoch']})" if "epoch" in meta else ""))
    print(f"Evaluating {len(classes)} classes on {device}: {', '.join(classes)}")
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    all_rows, summaries = [], {}
    for class_name in classes:
        class_idx = class_to_idx[class_name] if class_to_idx else None
        rows, summary = evaluate_class(model, class_name, class_idx, args, mean, std, device, amp_dtype, out_dir)
        all_rows += rows
        summaries[class_name] = summary
        s = summary["pooled"]
        print(f"{class_name:<14} image AUROC {100 * s['image_auroc']:5.1f} | pixel AUROC "
              f"{100 * s['pixel_auroc']:5.1f} | pixel AP {100 * s['pixel_ap']:5.1f} | AUPRO {100 * s['pixel_aupro']:5.1f}")

    write_csv(out_dir / "per_image.csv", ["class", "name", "label", "score"] + [f"pixel_{k}" for k in PIXEL_METRICS],
              all_rows)
    for protocol, title in (("pooled", "pixels pooled per class"), ("per_image", "pixel metrics averaged over anomalous images")):
        rows = [{"class": c, **s[protocol]} for c, s in summaries.items()]
        rows.append({"class": "mean", **{k: float(np.mean([r[k] for r in rows])) for k in METRIC_COLUMNS}})
        write_csv(out_dir / f"metrics_{protocol}.csv", ["class"] + METRIC_COLUMNS, rows)
        print_table(f"Results (%), {title}:", rows)
    print(f"\nSaved results to {out_dir}")


if __name__ == "__main__":
    main()
