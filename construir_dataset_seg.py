#!/usr/bin/env python3
"""
construir_dataset_seg.py -- Fase A del modelo: dataset procesado IGUAL que la app.

Problema: el modelo se entreno con la foto completa (letterbox), pero la app
primero SEGMENTA la hoja y recorta. Por eso baja de 97.9 % a 92.4 % con el
flujo real de la app.

Este script NO toca dataset/, model/ ni la app. Crea una carpeta NUEVA:
    ~/plagascan/dataset_seg/{train,val}/{no_frijol,plaga,sano}/
a partir de las MISMAS fotos originales (columna origen_local de
dataset/metadata.csv) y con la MISMA division train/val, pero procesadas con
el flujo de la app (copia exacta de LeafSegmenter.kt / eval_tflite_app.py):
    reducir a 1280 px -> mascara HSV -> limpieza morfologica -> contorno mayor
    -> si ocupa >= 15 %: recorte con margen 8 % ; si no: foto completa
    -> letterbox 224 x 224 con borde reflejado
Ademas agrega a train/sano las 48 fotos de campo propias (carpetas "entrenar";
las de "prueba_honesta" NO se usan, siguen siendo la prueba independiente).

En train se generan 4 versiones por foto (original + 3 variaciones: rotacion
+-25 grados, espejo, brillo +-30, contraste 0.85-1.15, semilla 42), igual que
el dataset anterior. Val queda 100 % real (sin variaciones).

Tambien informa cuantas fotos de cada clase RECHAZARIA la app por no ver
una hoja clara (dato util: esas fotos nunca llegan al modelo en el telefono).

Uso (dentro de ~/plagascan, con el venv activo):
    python ~/Descargas/construir_dataset_seg.py
"""
import csv
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

PS = Path.home() / "plagascan"
ORIGEN = PS / "dataset"
DESTINO = PS / "dataset_seg"
CAMPO = ["evidencia_campo_frijol/entrenar", "evidencia_campo_v2/entrenar"]

# Constantes de LeafSegmenter.kt / CameraScreen.kt
HUE_MIN, HUE_MAX, SAT_MIN, VAL_MIN = 25, 95, 30, 30
MIN_AREA_RATIO = 0.15
MARGIN_RATIO = 0.08
TARGET = 224
APP_MAX_SIDE = 1280
VARIANTES = 3  # + la original = 4 por foto en train


def kt_round(x):
    return int(math.floor(x + 0.5))


def letterbox(img):
    h, w = img.shape[:2]
    s = TARGET / max(h, w)
    nh, nw = kt_round(h * s), kt_round(w * s)
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    top = (TARGET - nh) // 2
    left = (TARGET - nw) // 2
    return cv2.copyMakeBorder(r, top, TARGET - nh - top, left, TARGET - nw - left, cv2.BORDER_REFLECT)


def flujo_app(img_bgr):
    """(imagen 224 lista, encontro_hoja, area_ratio). Si no hay hoja usa la foto completa."""
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
        return letterbox(img_bgr), False, 0.0
    largest = max(contours, key=cv2.contourArea)
    ratio = cv2.contourArea(largest) / total
    if ratio < MIN_AREA_RATIO:
        return letterbox(img_bgr), False, ratio
    x, y, bw, bh = cv2.boundingRect(largest)
    mx, my = int(bw * MARGIN_RATIO), int(bh * MARGIN_RATIO)
    x0, y0 = max(0, x - mx), max(0, y - my)
    x1, y1 = min(img_bgr.shape[1], x + bw + mx), min(img_bgr.shape[0], y + bh + my)
    return letterbox(img_bgr[y0:y1, x0:x1]), True, ratio


def variante(img, rng):
    """Rotacion +-25, espejo, brillo +-30, contraste 0.85-1.15 (bordes reflejados, no negros)."""
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), float(rng.uniform(-25, 25)), 1.0)
    out = cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REFLECT)
    if rng.random() < 0.5:
        out = cv2.flip(out, 1)
    f = out.astype(np.float32)
    media = f.mean()
    f = (f - media) * float(rng.uniform(0.85, 1.15)) + media + float(rng.uniform(-30, 30))
    return np.clip(f, 0, 255).astype(np.uint8)


def resolver(p):
    p = Path(p)
    return p if p.is_absolute() else PS / p


def main():
    if DESTINO.exists():
        sys.exit(f"NO SE HIZO NADA: ya existe {DESTINO}. Si quieres rehacerlo, bórralo antes:\n"
                 f"  rm -r {DESTINO}")
    meta = ORIGEN / "metadata.csv"
    if not meta.exists():
        sys.exit(f"NO SE HIZO NADA: no existe {meta}")

    items = []
    with open(meta, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("clase") not in ("no_frijol", "plaga", "sano") or row.get("split") not in ("train", "val"):
                continue
            items.append(dict(clase=row["clase"], split=row["split"], fuente=row["fuente"],
                              licencia=row["licencia"], origen=resolver(row["origen_local"])))
    for carpeta in CAMPO:
        for p in sorted((PS / carpeta).glob("*.jpg")):
            items.append(dict(clase="sano", split="train", fuente="campo_propio",
                              licencia="propia", origen=p))

    faltan = [it for it in items if not it["origen"].exists()]
    if faltan:
        print(f"[!] {len(faltan)} fotos originales no se encontraron; se omiten. Ejemplo: {faltan[0]['origen']}")
    items = [it for it in items if it["origen"].exists()]

    rng = np.random.default_rng(42)
    filas = []
    conteo = Counter()
    rechazos = defaultdict(lambda: [0, 0])  # (split,clase) -> [sin_hoja, total]
    for n, it in enumerate(items):
        img = cv2.imread(str(it["origen"]))
        if img is None:
            continue
        lista, hoja, ratio = flujo_app(img)
        clave = (it["split"], it["clase"])
        rechazos[clave][1] += 1
        if not hoja:
            rechazos[clave][0] += 1
        carpeta = DESTINO / it["split"] / it["clase"]
        carpeta.mkdir(parents=True, exist_ok=True)
        base = f"{it['clase']}_{it['fuente']}_{n:04d}"
        versiones = [lista] + ([variante(lista, rng) for _ in range(VARIANTES)] if it["split"] == "train" else [])
        for v, im in enumerate(versiones):
            nombre = f"{base}_v{v}.jpg" if it["split"] == "train" else f"{base}.jpg"
            cv2.imwrite(str(carpeta / nombre), im, [cv2.IMWRITE_JPEG_QUALITY, 92])
            conteo[clave] += 1
            filas.append(dict(archivo=f"{it['split']}/{it['clase']}/{nombre}", clase=it["clase"],
                              split=it["split"], fuente=it["fuente"], licencia=it["licencia"],
                              origen_local=str(it["origen"]), hoja_detectada=hoja,
                              area_ratio=round(ratio, 4), variante=v))
        if (n + 1) % 100 == 0:
            print(f"  {n + 1}/{len(items)} fotos procesadas...")

    with open(DESTINO / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)

    lineas = ["# dataset_seg -- procesado con el flujo de la app", "",
              "## Archivos generados (train incluye 4 versiones por foto)"]
    for k in sorted(conteo):
        lineas.append(f"  {k[0]}/{k[1]}: {conteo[k]}")
    lineas += ["", "## Fotos en las que la app NO veria una hoja clara (usan la foto completa aqui)"]
    for k in sorted(rechazos):
        sin, tot = rechazos[k]
        lineas.append(f"  {k[0]}/{k[1]}: {sin} de {tot} ({100 * sin / max(1, tot):.0f} %)")
    texto = "\n".join(lineas)
    (DESTINO / "resumen.txt").write_text(texto, encoding="utf-8")
    print(texto)
    print(f"\nListo. Carpeta nueva: {DESTINO}  (dataset/ y model/ no se tocaron)")


if __name__ == "__main__":
    main()
