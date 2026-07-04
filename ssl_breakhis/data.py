from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from torch.utils.data import Dataset


SUBTYPE_TO_INDEX = {
    "adenosis": 0,
    "fibroadenoma": 1,
    "phyllodes_tumor": 2,
    "tubular_adenoma": 3,
    "ductal_carcinoma": 4,
    "lobular_carcinoma": 5,
    "mucinous_carcinoma": 6,
    "papillary_carcinoma": 7,
}

BINARY_TO_INDEX = {"benign": 0, "malignant": 1}
INDEX_TO_SUBTYPE = {v: k for k, v in SUBTYPE_TO_INDEX.items()}
INDEX_TO_BINARY = {v: k for k, v in BINARY_TO_INDEX.items()}


@dataclass(frozen=True)
class BreakHisSample:
    path: str
    subtype: str
    subtype_label: int
    binary_label: int
    patient_id: str
    magnification: str


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def parse_breakhis_path(path: str | Path, root_dir: str | Path) -> BreakHisSample:
    path = Path(path)
    rel = path.relative_to(root_dir)
    parts = rel.parts
    if len(parts) < 6 or parts[1] != "SOB":
        raise ValueError(f"Unexpected BreakHis path layout: {path}")

    binary_class = parts[0]
    subtype = parts[2]
    patient_id = parts[3]
    magnification = parts[4].upper()

    if binary_class not in BINARY_TO_INDEX:
        raise ValueError(f"Unknown binary class '{binary_class}' in {path}")
    if subtype not in SUBTYPE_TO_INDEX:
        raise ValueError(f"Unknown subtype '{subtype}' in {path}")

    return BreakHisSample(
        path=str(path),
        subtype=subtype,
        subtype_label=SUBTYPE_TO_INDEX[subtype],
        binary_label=BINARY_TO_INDEX[binary_class],
        patient_id=patient_id,
        magnification=magnification,
    )


def scan_breakhis(
    root_dir: str | Path,
    magnifications: Optional[Iterable[str]] = None,
) -> list[BreakHisSample]:
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"BreakHis root does not exist: {root}")

    wanted_mags = None
    if magnifications:
        wanted_mags = {m.upper().replace("X", "") + "X" for m in magnifications}

    samples: list[BreakHisSample] = []
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            path = Path(dirpath) / filename
            try:
                sample = parse_breakhis_path(path, root)
            except ValueError:
                continue
            if wanted_mags is None or sample.magnification in wanted_mags:
                samples.append(sample)

    samples.sort(key=lambda sample: sample.path)
    if not samples:
        raise RuntimeError(f"No BreakHis images found under {root}")
    return samples


def _patient_labels(samples: list[BreakHisSample], task: str) -> tuple[list[str], list[int]]:
    labels_by_patient: dict[str, int] = {}
    for sample in samples:
        label = sample.subtype_label if task == "subtype" else sample.binary_label
        existing = labels_by_patient.setdefault(sample.patient_id, label)
        if existing != label:
            raise ValueError(f"Patient {sample.patient_id} appears with multiple labels.")
    patients = sorted(labels_by_patient)
    labels = [labels_by_patient[patient] for patient in patients]
    return patients, labels


def _stratified_patient_split(
    patients: list[str],
    labels: list[int],
    test_size: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    counts = np.bincount(np.asarray(labels))
    can_stratify = len(set(labels)) > 1 and counts[counts > 0].min() >= 2
    if can_stratify:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(splitter.split(patients, labels))
        return [patients[i] for i in train_idx], [patients[i] for i in test_idx]
    return train_test_split(patients, test_size=test_size, random_state=seed)


def patient_level_split(
    samples: list[BreakHisSample],
    task: str = "subtype",
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> tuple[list[BreakHisSample], list[BreakHisSample], list[BreakHisSample]]:
    if task not in {"subtype", "binary"}:
        raise ValueError("task must be 'subtype' or 'binary'")
    total = train_frac + val_frac + test_frac
    if not np.isclose(total, 1.0):
        raise ValueError("train_frac + val_frac + test_frac must equal 1.0")

    patients, labels = _patient_labels(samples, task)
    train_patients, heldout_patients = _stratified_patient_split(
        patients, labels, test_size=val_frac + test_frac, seed=seed
    )
    heldout_labels = [labels[patients.index(patient)] for patient in heldout_patients]
    relative_test_size = test_frac / (val_frac + test_frac)
    val_patients, test_patients = _stratified_patient_split(
        heldout_patients, heldout_labels, test_size=relative_test_size, seed=seed + 1
    )

    split_by_patient = {
        "train": set(train_patients),
        "val": set(val_patients),
        "test": set(test_patients),
    }
    train = [sample for sample in samples if sample.patient_id in split_by_patient["train"]]
    val = [sample for sample in samples if sample.patient_id in split_by_patient["val"]]
    test = [sample for sample in samples if sample.patient_id in split_by_patient["test"]]
    return train, val, test


class BreakHisDataset(Dataset):
    def __init__(self, samples: list[BreakHisSample], transform=None, task: str = "subtype"):
        if task not in {"subtype", "binary", "ssl"}:
            raise ValueError("task must be 'subtype', 'binary', or 'ssl'")
        self.samples = samples
        self.transform = transform
        self.task = task

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = Image.open(sample.path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        if self.task == "ssl":
            return image, sample.patient_id
        label = sample.subtype_label if self.task == "subtype" else sample.binary_label
        return image, torch.tensor(label, dtype=torch.long)


def class_weights(samples: list[BreakHisSample], task: str, device: torch.device) -> torch.Tensor:
    labels = [sample.subtype_label if task == "subtype" else sample.binary_label for sample in samples]
    num_classes = 8 if task == "subtype" else 2
    counts = np.bincount(np.asarray(labels), minlength=num_classes).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)

