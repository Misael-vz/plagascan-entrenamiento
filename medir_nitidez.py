#!/usr/bin/env python3
"""
medir_nitidez.py -- Experimento de SOLO LECTURA: ¿conviene que la app rechace
las fotos donde la hoja salio borrosa?

Motivo: la unica hoja de frijol sana que el modelo v7 marco en rojo
(gt_1788923767683.jpg) es una foto de noche donde la camara enfoco la pared y
la hoja quedo fuera de foco.

Que mide: la nitidez DENTRO de la hoja (varianza del Laplaciano, con OpenCV,
sobre el recorte de la hoja llevado a 512 px). Numero bajo = hoja borrosa.
Se mide dentro de la hoja y no en toda la foto porque en esa foto el fondo si
estaba nitido.

Para cada umbral posible reporta cuantas fotos se rechazarian en cada grupo y
cuantos errores del modelo quedarian de cada lado. El filtro sirve solo si
los errores se concentran en las fotos borrosas y casi no rechaza fotos buenas.

No toca la app, el modelo ni los datasets.

Uso (dentro de ~/plagascan, con el venv activo):
    python ~/Descargas/medir_nitidez.py 2>/dev/null
"""
import csv
import os
import sys
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2
import numpy as np

PS = Path.home() / "plagascan"
sys.path.insert(0, str(PS))
import eval_tflite_app as ev  # constantes y modelo iguales a los de la app

UMBRALES = (200, 400, 600, 800, 1000, 1500)
DESDE_OCT = int(datetime(2026, 10, 3).timestamp() * 1000)
EXT = (".jpg", ".jpeg", ".png")


def analizar(it, img):
    """(nitidez dentro de la hoja o None si no hay hoja, clase que dice el modelo)."""
    h, w = img.shape[:2]
    if max(h, w) > ev.APP_MAX_SIDE:
        s = ev.APP_MAX_SIDE / max(h, w)
        img = cv2.resize(img, (ev.kt_round(w * s), ev.kt_round(h * s)), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([ev.HUE_MIN, ev.SAT_MIN, ev.VAL_MIN]), np.array([ev.HUE_MAX, 255, 255]))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return None, "sin_hoja"
    mayor = max(contornos, key=cv2.contourArea)
    if cv2.contourArea(mayor) / (img.shape[0] * img.shape[1]) < ev.MIN_AREA_RATIO:
        return None, "sin_hoja"
    hoja = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(hoja, [mayor], -1, 255, -1)
    x, y, bw, bh = cv2.boundingRect(mayor)
    mx, my = int(bw * ev.MARGIN_RATIO), int(bh * ev.MARGIN_RATIO)
    x0, y0 = max(0, x - mx), max(0, y - my)
    x1, y1 = min(img.shape[1], x + bw + mx), min(img.shape[0], y + bh + my)
    recorte, m = img[y0:y1, x0:x1], hoja[y0:y1, x0:x1]

    s = 512 / max(recorte.shape[:2])
    tam = (max(1, int(recorte.shape[1] * s + 0.5)), max(1, int(recorte.shape[0] * s + 0.5)))
    gris = cv2.cvtColor(cv2.resize(recorte, tam, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gris, cv2.CV_64F, ksize=3)
    dentro = cv2.erode((cv2.resize(m, tam, interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    nitidez = float(lap[dentro].var()) if int(dentro.sum()) > 100 else float(lap.var())

    p = ev.predecir(it, ev.letterbox(recorte, ev.kt_round))
    return nitidez, ev.CLASSES[int(np.argmax(p))]


def ms_de(p):
    try:
        return int(p.stem.split("_")[1])
    except (IndexError, ValueError):
        return int(p.stat().st_mtime * 1000)


def reunir():
    """[(grupo, clase esperada, ruta)]"""
    items, vistos = [], set()

    def agregar(grupo, esperado, ruta):
        ruta = Path(ruta)
        if ruta.exists() and ruta.suffix.lower() in EXT and ruta not in vistos:
            vistos.add(ruta)
            items.append((grupo, esperado, ruta))

    entrenar = [PS / "evidencia_campo_frijol/entrenar", PS / "evidencia_campo_v2/entrenar"]
    usadas = {p.name for c in entrenar for p in c.glob("*.jpg")}
    for carpeta in ("evidencia_campo_frijol/prueba_honesta", "evidencia_campo_v2/prueba_honesta",
                    "campo_nuevo/files/ground_truth_captures/frijol", "prueba_oct/frijol"):
        for p in sorted((PS / carpeta).glob("*")):
            if p.name not in usadas:
                agregar("frijol propio, NO visto al entrenar", "sano", p)
    apartadas = {p.name for p in (PS / "prueba_oct/frijol").glob("*")}
    for c in entrenar:
        for p in sorted(c.glob("*")):
            agregar("frijol propio, visto al entrenar", "sano", p)
    for p in sorted((PS / "campo_oct/files/ground_truth_captures/frijol").glob("*")):
        if ms_de(p) >= DESDE_OCT and p.name not in apartadas:
            agregar("frijol propio, visto al entrenar", "sano", p)
    with open(PS / "dataset/metadata.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] == "val" and r["clase"] == "sano":
                agregar("val/sano (Makerere)", "sano", ev.resolver(r["origen_local"]))
            elif r["split"] == "val" and r["clase"] == "plaga":
                limon = "lelephid" in r.get("fuente", "").lower()
                agregar("val/plaga (limón)" if limon else "val/plaga (morfo amarillo)", "plaga", ev.resolver(r["origen_local"]))
    return items


def main():
    it = ev.cargar_interprete(ev.APP_MODEL)
    print(f"Modelo de la app: {ev.APP_MODEL} ({ev.APP_MODEL.stat().st_size // 1024} KB)")
    items = reunir()
    print(f"Fotos a revisar: {len(items)}")
    filas, grupos = [], defaultdict(list)
    for grupo, esperado, ruta in items:
        img = cv2.imread(str(ruta))
        if img is None:
            continue
        nit, clase = analizar(it, img)
        fila = dict(grupo=grupo, archivo=ruta.name, esperado=esperado, modelo=clase,
                    nitidez="" if nit is None else round(nit, 1), ancho=img.shape[1], alto=img.shape[0])
        filas.append(fila)
        grupos[grupo].append(fila)

    L = [f"# Nitidez de la hoja y errores del modelo -- {datetime.now():%Y-%m-%d %H:%M}",
         "nitidez = varianza del Laplaciano dentro de la hoja (recorte a 512 px). Bajo = borrosa.", ""]
    L.append(f"{'grupo':38s} {'n':>4s} {'con hoja':>8s} {'mín':>7s} {'p10':>7s} {'mediana':>8s} {'errores':>8s}   tamaño típico")
    for g, fs in grupos.items():
        con = [f for f in fs if f["nitidez"] != ""]
        v = np.array([f["nitidez"] for f in con]) if con else np.array([0.0])
        err = sum(f["modelo"] != f["esperado"] for f in con)
        tam = f"{int(np.median([f['ancho'] for f in fs]))}x{int(np.median([f['alto'] for f in fs]))}"
        L.append(f"{g:38s} {len(fs):4d} {len(con):8d} {v.min():7.0f} {np.percentile(v, 10):7.0f} {np.median(v):8.0f} {err:8d}   {tam}")

    L += ["", "## Si la app rechazara las fotos con nitidez menor al umbral",
          "   rechazadas = fotos que el usuario tendría que repetir; entre paréntesis, cuántas de ellas eran errores del modelo.",
          "   'errores que quedan' = errores del modelo en las fotos que sí pasarían.", ""]
    L.append(f"{'grupo':38s} " + " ".join(f"{'<' + str(u):>12s}" for u in UMBRALES))
    for g, fs in grupos.items():
        con = [f for f in fs if f["nitidez"] != ""]
        celdas = []
        for u in UMBRALES:
            bajo = [f for f in con if f["nitidez"] < u]
            celdas.append(f"{len(bajo):3d} ({sum(f['modelo'] != f['esperado'] for f in bajo)})".rjust(12))
        L.append(f"{g:38s} " + " ".join(celdas))
        quedan = [f"{sum(f['modelo'] != f['esperado'] for f in con if f['nitidez'] >= u):3d}".rjust(12) for u in UMBRALES]
        L.append(f"{'   errores que quedan':38s} " + " ".join(quedan))

    propias = sorted((f for f in filas if f["grupo"].startswith("frijol propio") and f["nitidez"] != ""), key=lambda f: f["nitidez"])
    L += ["", "## Las 15 fotos de frijol propio con la hoja más borrosa"]
    for f in propias[:15]:
        marca = "   <-- ERROR" if f["modelo"] != f["esperado"] else ""
        L.append(f"  nitidez {f['nitidez']:8.1f} | {f['archivo']} | modelo dice {f['modelo']}{marca}")
    L += ["", "## Errores del modelo en frijol propio y su nitidez"]
    L += [f"  nitidez {f['nitidez']:8.1f} | {f['archivo']} | modelo dice {f['modelo']} | {f['grupo']}"
          for f in propias if f["modelo"] != f["esperado"]] or ["  ninguno"]
    texto = "\n".join(L)
    salida = PS / "eval_app" / f"nitidez_{datetime.now():%Y%m%d_%H%M%S}"
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.with_suffix(".txt").write_text(texto, encoding="utf-8")
    with open(salida.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)
    print(texto)
    print(f"\nGuardado en: {salida.with_suffix('.txt')} (detalle por foto en el .csv)")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("ERROR (pégame este texto):")
        print(traceback.format_exc())
        sys.exit(1)
