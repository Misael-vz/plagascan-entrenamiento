#!/usr/bin/env python3
"""
PlagaScan — augment_dataset.py

Data augmentation sobre el set de TRAIN únicamente (val se deja intacto,
100% imágenes reales, para que las métricas de evaluación sean confiables).

Aplica combinaciones de:
  - Rotación (aleatoria, rango configurable)
  - Espejeo horizontal
  - Ajuste de brillo/contraste

Uso:
    python augment_dataset.py --root ~/plagascan/dataset --multiplier 4

Con multiplier=4, cada imagen real de train genera 3 versiones aumentadas
adicionales (total 4x: 1 original + 3 aumentadas), guardadas junto a las
originales en la misma carpeta train/{clase}/.
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

random.seed(42)
np.random.seed(42)


def rotate(img, max_angle=25):
    h, w = img.shape[:2]
    angle = random.uniform(-max_angle, max_angle)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def flip_horizontal(img):
    return cv2.flip(img, 1)


def adjust_brightness_contrast(img, brightness_range=(-30, 30), contrast_range=(0.85, 1.15)):
    brightness = random.uniform(*brightness_range)
    contrast = random.uniform(*contrast_range)
    out = img.astype(np.float32) * contrast + brightness
    return np.clip(out, 0, 255).astype(np.uint8)


def augment_once(img):
    """Aplica una combinación aleatoria de las transformaciones."""
    out = img.copy()
    if random.random() < 0.7:
        out = rotate(out)
    if random.random() < 0.5:
        out = flip_horizontal(out)
    out = adjust_brightness_contrast(out)
    return out


def augment_class_dir(class_dir: Path, multiplier: int):
    original_files = sorted([p for p in class_dir.glob("*.jpg")
                              if "_aug" not in p.stem])
    print(f"  {class_dir.name}: {len(original_files)} imágenes originales -> "
          f"generando {multiplier - 1} versiones aumentadas cada una")

    generated = 0
    for f in original_files:
        img = cv2.imread(str(f))
        if img is None:
            continue
        for k in range(multiplier - 1):
            aug_img = augment_once(img)
            out_path = class_dir / f"{f.stem}_aug{k}.jpg"
            cv2.imwrite(str(out_path), aug_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            generated += 1

    print(f"  {class_dir.name}: {generated} imágenes aumentadas generadas "
          f"(total final: {len(original_files) + generated})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="~/plagascan/dataset")
    parser.add_argument("--multiplier", type=int, default=4,
                         help="Factor total (1=original, 4=original+3 aumentadas)")
    args = parser.parse_args()

    dataset_root = Path(args.root).expanduser()
    train_dir = dataset_root / "train"

    if not train_dir.exists():
        print(f"[!] No existe {train_dir}. Corre build_dataset.py primero.")
        return

    print(f"Aplicando augmentation (multiplier={args.multiplier}) SOLO sobre train/")
    print("val/ se deja intacto (debe contener únicamente imágenes reales).\n")

    for class_dir in sorted(train_dir.iterdir()):
        if class_dir.is_dir():
            augment_class_dir(class_dir, args.multiplier)

    print("\nListo. Verifica conteos finales con:")
    print(f"  find {train_dir} -name '*.jpg' | sed 's#.*/##; s#_[a-z]*[0-9]*\\.jpg##' | wc -l")
    for class_dir in sorted(train_dir.iterdir()):
        if class_dir.is_dir():
            n = len(list(class_dir.glob("*.jpg")))
            print(f"  {class_dir.name}: {n} imágenes totales en train/")


if __name__ == "__main__":
    main()
