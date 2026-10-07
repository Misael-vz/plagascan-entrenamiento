#!/usr/bin/env python3
"""
PlagaScan — preprocess_opencv.py

Módulo 2, Paso 1: Preprocesamiento OpenCV antes de la clasificación.

Según la arquitectura híbrida del proyecto (documento U2_P3, sección
"Módulo de Procesamiento OpenCV"): esta etapa recibe una foto cruda de
cámara y la deja lista para el modelo MobileNetV2/TFLite, mediante:

  1. Conversión RGB -> HSV
  2. Segmentación de la hoja (separarla del fondo: tierra, piso, manos, etc.)
  3. Recorte a la región de la hoja (bounding box del contorno más grande)
  4. Redimensionado a 224x224 con el MISMO método de padding reflejado
     usado en build_dataset.py (letterbox_resize) — es importante que
     coincida con cómo se entrenó el modelo, o el preprocesamiento en
     producción no será consistente con el entrenamiento.

Uso:
    python preprocess_opencv.py --image ruta/a/foto.jpg
    python preprocess_opencv.py --batch ~/plagascan/dataset/val/plaga --n 6

Salida:
    Genera un mosaico de control (original | máscara | segmentada) para
    verificar visualmente que la segmentación funciona antes de conectar
    este módulo con el clasificador.
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

TARGET_SIZE = 224

# Rango HSV para vegetación (hoja verde). Amplio a propósito: cubre desde
# verde oscuro hasta verde-amarillento, para no perder hojas con
# amarillamiento (síntoma relevante para el proyecto) ni colonias de plaga
# sobre la hoja.
LEAF_HUE_MIN, LEAF_HUE_MAX = 25, 95
LEAF_SAT_MIN = 30
LEAF_VAL_MIN = 30


def letterbox_resize(img, size=TARGET_SIZE):
    """Idéntico al usado en build_dataset.py — debe coincidir con el
    preprocesamiento de entrenamiento para que el modelo funcione bien
    en producción."""
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    top = (size - nh) // 2
    bottom = size - nh - top
    left = (size - nw) // 2
    right = size - nw - left
    return cv2.copyMakeBorder(resized, top, bottom, left, right, borderType=cv2.BORDER_REFLECT)


def segment_leaf(img_bgr):
    """Segmenta la hoja del fondo y recorta al contorno más grande.

    Devuelve: (imagen_recortada_bgr, mascara_binaria, encontrado_bool)
    Si no se encuentra un contorno razonable, devuelve la imagen original
    sin recortar (fallback seguro: mejor procesar de más que perder la
    imagen completa por una segmentación fallida)."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    lower = np.array([LEAF_HUE_MIN, LEAF_SAT_MIN, LEAF_VAL_MIN])
    upper = np.array([LEAF_HUE_MAX, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    # Limpieza morfológica: cerrar huecos pequeños (nervaduras, manchas
    # oscuras de plaga dentro de la hoja) y quitar ruido pequeño de fondo
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img_bgr, mask, False

    largest = max(contours, key=cv2.contourArea)
    area_ratio = cv2.contourArea(largest) / (img_bgr.shape[0] * img_bgr.shape[1])

    if area_ratio < 0.03:  # contorno demasiado pequeño, probablemente ruido
        return img_bgr, mask, False

    x, y, w, h = cv2.boundingRect(largest)
    # Margen de seguridad del 8% alrededor del contorno, para no cortar
    # bordes de la hoja ni colonias de plaga cerca del borde
    margin_x = int(w * 0.08)
    margin_y = int(h * 0.08)
    x0 = max(0, x - margin_x)
    y0 = max(0, y - margin_y)
    x1 = min(img_bgr.shape[1], x + w + margin_x)
    y1 = min(img_bgr.shape[0], y + h + margin_y)

    cropped = img_bgr[y0:y1, x0:x1]
    return cropped, mask, True


def preprocess_single(img_path: Path):
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    cropped, mask, found = segment_leaf(img)
    final = letterbox_resize(cropped)
    return img, mask, cropped, final, found


def make_control_grid(samples, out_path):
    """samples: lista de (nombre, original, mask, cropped, final, found)"""
    rows = []
    for name, original, mask, cropped, final, found in samples:
        thumb_size = 180
        orig_t = cv2.resize(original, (thumb_size, thumb_size))
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_t = cv2.resize(mask_bgr, (thumb_size, thumb_size))
        final_t = cv2.resize(final, (thumb_size, thumb_size))

        label = "OK" if found else "FALLBACK (sin recorte)"
        color = (0, 180, 0) if found else (0, 0, 220)
        for t in (orig_t, mask_t, final_t):
            cv2.rectangle(t, (0, 0), (thumb_size, 20), (0, 0, 0), -1)
        cv2.putText(orig_t, "original", (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(mask_t, "mascara", (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(final_t, label, (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        row = np.hstack([orig_t, mask_t, final_t])
        name_bar = np.zeros((22, row.shape[1], 3), dtype=np.uint8)
        cv2.putText(name_bar, name, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        rows.append(np.vstack([name_bar, row]))

    grid = np.vstack(rows)
    cv2.imwrite(str(out_path), grid)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, help="Ruta a una sola imagen")
    parser.add_argument("--batch", type=str, help="Carpeta con varias imágenes")
    parser.add_argument("--n", type=int, default=6, help="Cuántas imágenes tomar de --batch")
    parser.add_argument("--out", type=str, default="~/plagascan/model/preprocess_check.png")
    args = parser.parse_args()

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    samples = []

    if args.image:
        img_path = Path(args.image).expanduser()
        result = preprocess_single(img_path)
        if result is None:
            print(f"[!] No se pudo leer: {img_path}")
            return
        original, mask, cropped, final, found = result
        samples.append((img_path.name, original, mask, cropped, final, found))

    elif args.batch:
        batch_dir = Path(args.batch).expanduser()
        files = list(batch_dir.glob("*.jpg")) + list(batch_dir.glob("*.jpeg"))
        random.seed(3)
        chosen = random.sample(files, min(args.n, len(files)))
        for f in chosen:
            result = preprocess_single(f)
            if result is None:
                continue
            original, mask, cropped, final, found = result
            samples.append((f.name, original, mask, cropped, final, found))
    else:
        print("Debes pasar --image o --batch")
        return

    if not samples:
        print("[!] No se procesó ninguna imagen.")
        return

    n_found = sum(1 for s in samples if s[5])
    print(f"Procesadas: {len(samples)} | Segmentación exitosa: {n_found} | Fallback: {len(samples) - n_found}")

    make_control_grid(samples, out_path)
    print(f"Guardado: {out_path}")
    print("\nRevisa: columna 'mascara' debe cubrir la hoja (blanco) sin incluir")
    print("demasiado fondo (negro debe ser tierra/piso/otros elementos).")


if __name__ == "__main__":
    main()
