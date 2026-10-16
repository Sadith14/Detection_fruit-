"""Carga y preprocesamiento compartidos por el entrenamiento y la evaluación."""
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from model import CLASSES, letterbox

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def find_images(root, require_all_classes, allow_empty=False):
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"No existe la raíz de datos: {root}")
    records = []
    missing = []
    for label, class_name in enumerate(CLASSES):
        class_dir = root / class_name
        paths = sorted(
            path for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ) if class_dir.is_dir() else []
        if not paths:
            missing.append(class_name)
        records.extend((path, label) for path in paths)
    if not records and allow_empty:
        return []
    if not records:
        raise ValueError(f"No se encontraron imágenes JPG, JPEG o PNG en {root}")
    if require_all_classes and missing:
        raise ValueError(f"Faltan imágenes de estas clases en {root}: {', '.join(missing)}")
    return records


class ImageDataset(Dataset):
    def __init__(self, roots, require_all_classes, allow_empty_roots=False):
        if isinstance(roots, (str, Path)):
            roots = [roots]
        self.records = []
        for root in roots:
            self.records.extend(
                find_images(root, require_all_classes=False, allow_empty=allow_empty_roots)
            )
        present_labels = {label for _, label in self.records}
        if not self.records:
            raise ValueError("No hay imágenes para procesar.")
        if require_all_classes:
            missing = [CLASSES[index] for index in range(len(CLASSES)) if index not in present_labels]
            if missing:
                raise ValueError(f"Faltan imágenes de estas clases: {', '.join(missing)}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        path, label = self.records[index]
        with Image.open(path) as image:
            array = np.asarray(letterbox(image), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        return tensor, label


def normalize_batch(images, mean, std):
    mean_tensor = torch.tensor(mean, dtype=images.dtype).view(1, 3, 1, 1)
    std_tensor = torch.tensor(std, dtype=images.dtype).view(1, 3, 1, 1)
    return (images - mean_tensor) / std_tensor
