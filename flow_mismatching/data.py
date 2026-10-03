"""Datasets in the MVTec-AD folder layout (VisA is converted to the same layout).

    <data_root>/<class>/train/good/*.png
    <data_root>/<class>/test/<good | defect type>/*.png
    <data_root>/<class>/ground_truth/<defect type>/<image stem>_mask.png   (MVTec-AD)
    <data_root>/<class>/ground_truth/<defect type>/<image stem>.png        (VisA)
"""

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# Per-channel mean and std of the normal training images.
NORM_STATS = {
    "mvtec": ((0.431892, 0.401233, 0.391279), (0.254873, 0.251537, 0.248335)),
    "visa": ((0.320139, 0.389447, 0.311646), (0.278132, 0.229037, 0.216387)),
}


def image_transform(img_size: int, mean: Sequence[float], std: Sequence[float]) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def list_images(folder: Path) -> List[Path]:
    return [p for p in sorted(folder.rglob("*")) if p.suffix.lower() in IMG_EXTENSIONS]


def discover_classes(data_root: str, split: str) -> List[str]:
    """Sorted names of the class folders under data_root that contain a ``split`` folder."""
    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    classes = sorted(p.name for p in root.iterdir() if (p / split).is_dir())
    if not classes:
        raise RuntimeError(f"No class folders with a '{split}' sub-folder found under {root}")
    return classes


class TrainDataset(Dataset):
    """Normal training images of all classes. Returns (image, class index)."""

    def __init__(
        self,
        data_root: str,
        class_names: Sequence[str],
        img_size: int,
        mean: Sequence[float],
        std: Sequence[float],
    ):
        self.class_names = list(class_names)
        self.files: List[Path] = []
        self.labels: List[int] = []
        for idx, name in enumerate(self.class_names):
            files = list_images(Path(data_root) / name / "train" / "good")
            self.files += files
            self.labels += [idx] * len(files)
        if not self.files:
            raise RuntimeError(f"No training images found under {data_root}/<class>/train/good")
        self.transform = image_transform(img_size, mean, std)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image = Image.open(self.files[idx]).convert("RGB")
        return self.transform(image), self.labels[idx]


def find_mask(mask_dir: Path, stem: str) -> Optional[Path]:
    """Ground-truth mask of an image: <stem>_mask.* (MVTec-AD) or <stem>.* (VisA)."""
    for pattern in (f"{stem}_mask.*", f"{stem}.*", f"{stem}*"):
        candidates = sorted(mask_dir.glob(pattern))
        if candidates:
            return candidates[0]
    return None


class TestDataset(Dataset):
    """Test images of one class. Returns (image, mask [1, H, W], label, name).

    label is 0 for images in test/good and 1 otherwise; name is "<defect type>_<image stem>".
    """

    def __init__(
        self,
        data_root: str,
        class_name: str,
        img_size: int,
        mean: Sequence[float],
        std: Sequence[float],
    ):
        class_dir = Path(data_root) / class_name
        test_dir = class_dir / "test"
        if not test_dir.is_dir():
            raise FileNotFoundError(f"Missing test folder: {test_dir}")

        self.items: List[Tuple[Path, Optional[Path], int, str]] = []
        for defect_dir in sorted(p for p in test_dir.iterdir() if p.is_dir()):
            defect = defect_dir.name
            for img_path in list_images(defect_dir):
                if defect == "good":
                    mask_path, label = None, 0
                else:
                    mask_path, label = find_mask(class_dir / "ground_truth" / defect, img_path.stem), 1
                    if mask_path is None:
                        raise FileNotFoundError(f"No ground-truth mask for {img_path}")
                self.items.append((img_path, mask_path, label, f"{defect}_{img_path.stem}"))
        if not self.items:
            raise RuntimeError(f"No test images found under {test_dir}")

        self.img_size = img_size
        self.transform = image_transform(img_size, mean, std)
        self.mask_resize = transforms.Resize(
            (img_size, img_size), interpolation=transforms.InterpolationMode.NEAREST
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        img_path, mask_path, label, name = self.items[idx]
        image = self.transform(Image.open(img_path).convert("RGB"))
        if mask_path is None:
            mask = torch.zeros(1, self.img_size, self.img_size)
        else:
            mask = TF.to_tensor(self.mask_resize(Image.open(mask_path).convert("L")))
            mask = (mask > 0.5).float()
        return image, mask, label, name
