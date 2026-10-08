#!/usr/bin/env python3
"""
construir_dataset_v6.py -- Dataset v6: dataset_seg (v5) + fotos NUEVAS de tu planta.

Objetivo: que el modelo reconozca TU frijol (hoy lo clasifica como "no_frijol")
sin dar falsas alarmas con las cosas que no son frijol.

No toca dataset/, dataset_seg/, model*/ ni la app. Crea:
  ~/plagascan/dataset_v6/   copia de dataset_seg + fotos nuevas (solo en train)
  ~/plagascan/prueba_oct/   fotos nuevas APARTADAS para probar (el modelo no las ve)
      frijol/  -> se espera "sano"
      otros/   -> se espera "no_frijol"

Fotos nuevas = las de Recoleccion (PlagaScan Dev) tomadas desde el 3 de octubre,
que ya estan en ~/plagascan/campo_oct/files/ground_truth_captures/{frijol,otros}.
Las de septiembre NO se usan para entrenar: siguen siendo prueba independiente.

Como se separan: en orden de captura, en bloques de 4 fotos seguidas; de cada
4 bloques, 3 van a entrenamiento y 1 a prueba. Asi las fotos casi iguales
(tomadas una tras otra) no quedan repartidas entre entrenamiento y prueba, y
la prueba incluye de todo (planta completa, hoja, enves, detalle).

Las fotos de entrenamiento pasan por el MISMO flujo que la app (segmentar,
recortar, 224x224) y se generan 4 versiones de cada una, igual que antes.

Uso (dentro de ~/plagascan, con el venv activo):
    python ~/Descargas/construir_dataset_v6.py
    python ~/Descargas/construir_dataset_v6.py --sin-otros   (no recomendado)
"""
import csv
import math
import shutil
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

PS = Path.home() / "plagascan"
BASE = PS / "dataset_seg"
DESTINO = PS / "dataset_v6"
PRUEBA = PS / "prueba_oct"
CAPTURAS = PS / "campo_oct/files/ground_truth_captures"
DESDE_MS = int(datetime(2026, 10, 3).timestamp() * 1000)
MIN_OTROS = 20

HUE_MIN, HUE_MAX, SAT_MIN, VAL_MIN = 25, 95, 30, 30
MIN_AREA_RATIO, MARGIN_RATIO, TARGET, APP_MAX_SIDE, VARIANTES = 0.15, 0.08, 224, 1280, 3


def kt_round(x):
    return int(math.floor(x + 0.5))


def letterbox(img):
    h, w = img.shape[:2]
    s = TARGET / max(h, w)
    nh, nw = kt_round(h * s), kt_round(w * s)
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    top, left = (TARGET - nh) // 2, (TARGET - nw) // 2
    return cv2.copyMakeBorder(r, top, TARGET - nh - top, left, TARGET - nw - left, cv2.BORDER_REFLECT)


def flujo_app(img_bgr):
    h, w = img_bgr.shape[:2]
    if max(h, w) > APP_MAX_SIDE:
        s = APP_MAX_SIDE / max(h, w)
        img_bgr = cv2.resize(img_bgr, (kt_round(w * s), kt_round(h * s)), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([HUE_MIN, SAT_MIN, VAL_MIN]), np.array([HUE_MAX, 255, 255]))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total = img_bgr.shape[0] * img_bgr.shape[1]
    if not contours:
        return letterbox(img_bgr), False
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) / total < MIN_AREA_RATIO:
        return letterbox(img_bgr), False
    x, y, bw, bh = cv2.boundingRect(largest)
    mx, my = int(bw * MARGIN_RATIO), int(bh * MARGIN_RATIO)
    x0, y0 = max(0, x - mx), max(0, y - my)
    x1, y1 = min(img_bgr.shape[1], x + bw + mx), min(img_bgr.shape[0], y + bh + my)
    return letterbox(img_bgr[y0:y1, x0:x1]), True


def variante(img, rng):
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), float(rng.uniform(-25, 25)), 1.0)
    out = cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REFLECT)
    if rng.random() < 0.5:
        out = cv2.flip(out, 1)
    f = out.astype(np.float32)
    media = f.mean()
    f = (f - media) * float(rng.uniform(0.85, 1.15)) + media + float(rng.uniform(-30, 30))
    return np.clip(f, 0, 255).astype(np.uint8)


def ms_de(p):
    """Momento de la captura: del nombre gt_<milisegundos>.jpg; si no, de la fecha del archivo."""
    try:
        return int(p.stem.split("_")[1])
    except (IndexError, ValueError):
        return int(p.stat().st_mtime * 1000)


def nuevas(carpeta):
    return sorted((p for p in (CAPTURAS / carpeta).glob("*.jpg") if ms_de(p) >= DESDE_MS), key=ms_de)


def separar(fotos):
    """Bloques de 4 fotos seguidas; 1 de cada 4 bloques va a prueba."""
    train, test = [], []
    for i, p in enumerate(fotos):
        (test if (i // 4) % 4 == 3 else train).append(p)
    return train, test


def main():
    sin_otros = "--sin-otros" in sys.argv
    for ruta, que in [(BASE, "dataset_seg (corre antes construir_dataset_seg.py)"), (CAPTURAS, "campo_oct (trae las fotos del teléfono)")]:
        if not ruta.exists():
            sys.exit(f"NO SE HIZO NADA: falta {que}: {ruta}")
    for ruta in (DESTINO, PRUEBA):
        if ruta.exists():
            sys.exit(f"NO SE HIZO NADA: ya existe {ruta}. Para rehacerlo: rm -r {DESTINO} {PRUEBA}")

    frijol, otros = nuevas("frijol"), nuevas("otros")
    print(f"Fotos nuevas (desde el 3 de octubre): frijol={len(frijol)}  otros={len(otros)}")
    if len(frijol) < 20:
        sys.exit("NO SE HIZO NADA: hay menos de 20 fotos nuevas de frijol.")
    if len(otros) < MIN_OTROS and not sin_otros:
        sys.exit(f"NO SE HIZO NADA: solo hay {len(otros)} fotos nuevas de 'otros' (mínimo {MIN_OTROS}).\n"
                 "  Sin ellas el modelo puede dar falsas alarmas. Tómalas en Recolección > Otros,\n"
                 "  vuelve a traer las fotos del teléfono y corre este script otra vez.\n"
                 "  (Para continuar de todos modos: agrega --sin-otros)")

    shutil.copytree(BASE, DESTINO)
    rng = np.random.default_rng(6)
    filas = []
    resumen = []
    for nombre, fotos, clase, fuente in [("frijol", frijol, "sano", "campo_oct"), ("otros", otros, "no_frijol", "campo_oct_otros")]:
        train, test = separar(fotos)
        (PRUEBA / nombre).mkdir(parents=True, exist_ok=True)
        for p in test:
            shutil.copy2(p, PRUEBA / nombre / p.name)
        sin_hoja = 0
        carpeta = DESTINO / "train" / clase
        for p in train:
            img = cv2.imread(str(p))
            if img is None:
                continue
            lista, hoja = flujo_app(img)
            sin_hoja += not hoja
            for v, im in enumerate([lista] + [variante(lista, rng) for _ in range(VARIANTES)]):
                archivo = f"{clase}_{fuente}_{p.stem}_v{v}.jpg"
                cv2.imwrite(str(carpeta / archivo), im, [cv2.IMWRITE_JPEG_QUALITY, 92])
                filas.append(dict(archivo=f"train/{clase}/{archivo}", clase=clase, split="train", fuente=fuente,
                                  licencia="propia", origen_local=str(p), hoja_detectada=hoja, area_ratio="", variante=v))
        resumen.append(f"  {nombre}: {len(train)} a entrenamiento ({len(train) * 4} archivos; "
                       f"{sin_hoja} sin hoja clara) y {len(test)} apartadas para prueba")

    meta = DESTINO / "metadata.csv"
    with open(meta, newline="", encoding="utf-8") as f:
        campos = csv.DictReader(f).fieldnames
    with open(meta, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=campos).writerows(filas)

    print("\n".join(resumen))
    print("\nArchivos por clase en dataset_v6:")
    for split in ("train", "val"):
        for clase in ("no_frijol", "plaga", "sano"):
            print(f"  {split}/{clase}: {len(list((DESTINO / split / clase).glob('*.jpg')))}")
    print(f"\nListo: {DESTINO} y {PRUEBA}  (nada más se modificó)")


if __name__ == "__main__":
    main()
