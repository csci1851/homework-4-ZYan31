"""
Data loader for Homework 4: CNNs vs FCNs

Assumes the dataset is already present under:
homework4/data/chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}/*.jpeg

Example structure:
data/chest_xray/
  train/
    NORMAL/
    PNEUMONIA/
  val/
    NORMAL/
    PNEUMONIA/
  test/
    NORMAL/
    PNEUMONIA/
"""

from typing import Tuple, List
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


class ChestXRayDataset(Dataset):
    def __init__(self, split_dir: Path, transform=None, isPneu=True):
        """
        Args:
            split_dir: Path like data/chest_xray/train
            transform: torchvision transform pipeline
        """
        self.split_dir = split_dir
        self.transform = transform
        self.class_to_idx = {
            "NORMAL": 0,
            "PNEUMONIA": 1,
        }
        if not isPneu:
            self.class_to_idx = {
                "NORMAL": 0,
                "CARDIOMEGALY": 1,
            }

        self.samples: List[Tuple[Path, int]] = []
        self._index_images()

    def _index_images(self) -> None:
        if not self.split_dir.exists():
            raise FileNotFoundError(f"Split folder not found: {self.split_dir}")

        for class_name, class_idx in self.class_to_idx.items():
            class_dir = self.split_dir / class_name
            if not class_dir.exists():
                raise FileNotFoundError(
                    f"Expected class folder not found: {class_dir}\n"
                    f"Expected classes: {list(self.class_to_idx.keys())}"
                )

            # Most common extensions in this dataset
            exts = ["*.jpeg", "*.jpg", "*.png"]
            for ext in exts:
                for img_path in class_dir.glob(ext):
                    self.samples.append((img_path, class_idx))

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found in {self.split_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        # X-rays are grayscale
        image = Image.open(img_path).convert("L")

        if self.transform:
            image = self.transform(image)

        return image, label


class HW4DataLoader:
    def __init__(self, filename):
        self.homework_dir = Path(__file__).resolve().parent.parent
        self.data_dir = self.homework_dir / "data"
        self.dataset_dir = self.data_dir / filename

    def _check_dataset_exists(self) -> None:
        if not self.dataset_dir.exists():
            raise FileNotFoundError(
                f"Expected dataset folder not found: {self.dataset_dir}\n"
                "Make sure the data is placed in:\n"
                "  homework4/data/chest_xray/\n"
            )

    def _augment_data_fcn(self):
        # FCN expects a flat vector
        return T.Compose(
            [
                T.Resize((128, 128)),
                T.ToTensor(),
                T.Lambda(lambda x: x.view(-1)),
            ]
        )

    def _augment_data_cnn(self):
        # CNN expects (C,H,W)
        return T.Compose(
            [
                T.Resize((128, 128)),
                T.RandomHorizontalFlip(),
                T.RandomRotation(10),
                T.ToTensor(),
                T.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

    def get_chest_xray_data(
        self,
        split: str = "train",
        for_cnn: bool = True,
        batch_size: int = 32,
        num_workers: int = 0,
        isPneu = True
    ) -> DataLoader:
        """
        Load the Chest X-Ray dataset.
        """
        self._check_dataset_exists()

        split_dir = self.dataset_dir / split
        #print(split_dir)
        if not split_dir.exists():
            raise FileNotFoundError(f"Expected split folder not found: {split_dir}")

        transform = self._augment_data_cnn() if for_cnn else self._augment_data_fcn()
        
        dataset = None
        if isPneu:
            dataset = ChestXRayDataset(split_dir=split_dir, transform=transform)
        else:
            dataset = ChestXRayDataset(split_dir=split_dir, transform=transform, isPneu = False)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
        )
