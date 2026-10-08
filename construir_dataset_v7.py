#!/usr/bin/env python3
"""
construir_dataset_v7.py -- Dataset v7: dataset_v6 + hojas de limon SANAS como
ejemplos de "no_frijol".

Por que: la prueba del 5 de octubre mostro que el modelo v6 marca como plaga
292 de 330 hojas de limon SANAS. Como casi toda la clase "plaga" son hojas de
limon con pulgon (LeLePhid), el modelo aprendio "hoja de limon = plaga" en vez
de fijarse en el pulgon. Al darle la misma cantidad de limon sano (no_frijol)
que de limon con pulgon (plaga), la hoja deja de servirle de pista y tiene
que mirar el insecto.

No toca dataset/, dataset_seg/, dataset_v6/, model*/ ni la app. Crea:
  ~/plagascan/dataset_v7/     copia de dataset_v6 + 120 hojas de limon sanas
                              en train/no_frijol (4 versiones de cada una)
  ~/plagascan/prueba_limon/   fotos APARTADAS que el modelo no ve al entrenar
      limon_sano/    210 hojas de limon sanas        -> se espera "no_frijol"
      limon_pulgon/  185 hojas de limon con pulgon   -> se espera "plaga"
                     (las de LeLePhid que nunca se usaron)

Como se separan las sanas: en orden de nombre, en bloques de 5 fotos seguidas;
de cada 11 bloques, 4 van a entrenamiento y 7 a prueba (fotos seguidas suelen
ser de la misma hoja, asi no quedan repartidas entre los dos lados).

Fuente: LeLePhid (Mendeley Data), licencia CC BY 4.0.

Uso (dentro de ~/plagascan, con el venv activo):
    python ~/Descargas/construir_dataset_v7.py
"""
import csv
import math
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

PS = Path.home() / "plagascan"
BASE = PS / "dataset_v6"
DESTINO = PS / "dataset_v7"
PRUEBA = PS / "prueba_limon"
RAW = PS / "dataset/raw"
EXT = (".jpg", ".jpeg", ".png")

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


def main():
    for ruta, que in [(BASE, "dataset_v6"), (RAW, "dataset/raw")]:
        if not ruta.exists():
            sys.exit(f"NO SE HIZO NADA: falta {que}: {ruta}")
    for ruta in (DESTINO, PRUEBA):
        if ruta.exists():
            sys.exit(f"NO SE HIZO NADA: ya existe {ruta}. Para rehacerlo: rm -r {DESTINO} {PRUEBA}")

    lele = [p for p in RAW.rglob("*") if p.is_file() and p.suffix.lower() in EXT and "lelephid" in str(p).lower()]
    sanas = sorted((p for p in lele if p.name.lower().startswith("healthy")), key=lambda p: p.name)
    pulgon = sorted((p for p in lele if p.name.lower().startswith("aphid")), key=lambda p: p.name)
    usadas = set()
    with open(PS / "dataset/metadata.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            o = Path(r["origen_local"])
            usadas.add((o if o.is_absolute() else PS / o).resolve())
    if any(p.resolve() in usadas for p in sanas):
        sys.exit("NO SE HIZO NADA: algunas hojas de limón sanas ya están en el dataset; avísame antes de seguir.")
    pulgon_nuevo = [p for p in pulgon if p.resolve() not in usadas]
    print(f"LeLePhid: {len(sanas)} hojas sanas; {len(pulgon)} con pulgón ({len(pulgon_nuevo)} sin usar)")
    if len(sanas) < 100:
        sys.exit("NO SE HIZO NADA: hay menos de 100 hojas de limón sanas.")

    train = [p for i, p in enumerate(sanas) if (i // 5) % 11 < 4]
    test = [p for i, p in enumerate(sanas) if (i // 5) % 11 >= 4]

    shutil.copytree(BASE, DESTINO)
    for carpeta, fotos in (("limon_sano", test), ("limon_pulgon", pulgon_nuevo)):
        (PRUEBA / carpeta).mkdir(parents=True, exist_ok=True)
        for p in fotos:
            shutil.copy2(p, PRUEBA / carpeta / p.name)

    rng = np.random.default_rng(7)
    filas, sin_hoja = [], 0
    carpeta = DESTINO / "train" / "no_frijol"
    for p in train:
        img = cv2.imread(str(p))
        if img is None:
            continue
        lista, hoja = flujo_app(img)
        sin_hoja += not hoja
        for v, im in enumerate([lista] + [variante(lista, rng) for _ in range(VARIANTES)]):
            archivo = f"no_frijol_lelephid_sano_{p.stem}_v{v}.jpg"
            cv2.imwrite(str(carpeta / archivo), im, [cv2.IMWRITE_JPEG_QUALITY, 92])
            filas.append(dict(archivo=f"train/no_frijol/{archivo}", clase="no_frijol", split="train",
                              fuente="lelephid_sano", licencia="CC-BY-4.0", origen_local=str(p),
                              hoja_detectada=hoja, area_ratio="", variante=v))
    meta = DESTINO / "metadata.csv"
    with open(meta, newline="", encoding="utf-8") as f:
        campos = csv.DictReader(f).fieldnames
    with open(meta, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=campos, extrasaction="ignore").writerows(filas)

    print(f"  limón sano: {len(train)} a entrenamiento ({len(filas)} archivos; {sin_hoja} sin hoja clara) "
          f"y {len(test)} apartadas para prueba")
    print(f"  limón con pulgón sin usar: {len(pulgon_nuevo)} apartadas para prueba")
    print("\nArchivos por clase en dataset_v7:")
    for split in ("train", "val"):
        for clase in ("no_frijol", "plaga", "sano"):
            print(f"  {split}/{clase}: {len(list((DESTINO / split / clase).glob('*.jpg')))}")
    n_lele = len(list((DESTINO / "train/plaga").glob("*lelephid*")))
    print(f"\nEn train: limón con pulgón (plaga) = {n_lele} archivos; limón sano (no_frijol) = {len(filas)} archivos")
    print(f"Listo: {DESTINO} y {PRUEBA}  (nada más se modificó)")


if __name__ == "__main__":
    main()
