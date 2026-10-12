#!/usr/bin/env python3
"""
probar_limon.py -- Prueba de SOLO LECTURA: ¿el modelo detecta el pulgon o solo
reconoce la hoja de limon?

Casi toda la clase "plaga" son hojas de limon con pulgon (LeLePhid) y toda la
clase "sano" son hojas de frijol. Un modelo puede sacar 35 de 38 aprendiendo
"hoja de limon = plaga" sin fijarse en el insecto. LeLePhid tambien trae hojas
de limon SANAS (healthy.*) que ya estan en el disco y sirven para comprobarlo:

  - Limon SANO: lo correcto es "no_frijol". Si muchas salen "plaga", el modelo
    esta reconociendo la hoja de limon y no el pulgon.
  - Limon CON pulgon que NO se uso al entrenar: deberia salir "plaga".

No toca la app, el modelo ni los datasets.

Uso (dentro de ~/plagascan, con el venv activo):
    python ~/Descargas/probar_limon.py 2>/dev/null
"""
import csv
import os
import sys
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2
import numpy as np

PS = Path.home() / "plagascan"
sys.path.insert(0, str(PS))
import eval_tflite_app as ev  # mismo flujo y mismo modelo que la app

UMBRAL = 0.80
CATS = ["plaga ROJO", "plaga AMARILLO", "sano VERDE", "sano AMARILLO", "no_frijol", "sin_hoja"]
EXT = (".jpg", ".jpeg", ".png")


def categoria(it, ruta):
    img = cv2.imread(str(ruta))
    if img is None:
        return None
    img224, _ = ev.pipeline_app(img)
    if img224 is None:
        return "sin_hoja"
    p = ev.predecir(it, img224)
    clase, conf = ev.CLASSES[int(np.argmax(p))], float(p.max())
    if clase == "no_frijol":
        return clase
    return f"{clase} {('ROJO' if clase == 'plaga' else 'VERDE') if conf >= UMBRAL else 'AMARILLO'}"


def main():
    raw = PS / "dataset/raw"
    lele = [p for p in raw.rglob("*") if p.is_file() and p.suffix.lower() in EXT and "lelephid" in str(p).lower()]
    sanas = sorted(p for p in lele if p.name.lower().startswith("healthy"))
    pulgon = sorted(p for p in lele if p.name.lower().startswith("aphid"))
    if not sanas and not pulgon:
        print(f"No encontré las fotos de LeLePhid dentro de {raw}.")
        print("Pégame la salida de:  ls ~/plagascan/dataset/raw")
        sys.exit(1)

    usadas, clase_de = set(), {}
    with open(PS / "dataset/metadata.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            ruta = ev.resolver(r["origen_local"]).resolve()
            usadas.add(ruta)
            clase_de[ruta] = r["clase"]
    sanas_usadas = Counter(clase_de[p.resolve()] for p in sanas if p.resolve() in usadas)
    pulgon_nuevo = [p for p in pulgon if p.resolve() not in usadas]

    it = ev.cargar_interprete(ev.APP_MODEL)
    L = [f"# ¿Pulgón o hoja de limón? -- {datetime.now():%Y-%m-%d %H:%M}",
         f"Modelo de la app: {ev.APP_MODEL} ({ev.APP_MODEL.stat().st_size // 1024} KB)",
         f"LeLePhid en disco: {len(sanas)} hojas sanas, {len(pulgon)} con pulgón "
         f"({len(pulgon) - len(pulgon_nuevo)} usadas en el dataset, {len(pulgon_nuevo)} sin usar)",
         f"Hojas de limón sanas que ya están en el dataset: {dict(sanas_usadas) or 'ninguna'}", "",
         f"{'grupo':40s} {'n':>4s} | " + " ".join(f"{c:>14s}" for c in CATS)]
    for nombre, fotos in (("limón SANO (se espera no_frijol)", sanas), ("limón CON pulgón, sin usar (plaga)", pulgon_nuevo)):
        c = Counter()
        for k, p in enumerate(fotos, 1):
            cat = categoria(it, p)
            if cat:
                c[cat] += 1
            if k % 100 == 0:
                print(f"  {nombre}: {k}/{len(fotos)}...")
        L.append(f"{nombre:40s} {sum(c.values()):4d} | " + " ".join(f"{c[x]:14d}" for x in CATS))
    texto = "\n".join(L)
    salida = PS / "eval_app" / f"limon_{datetime.now():%Y%m%d_%H%M%S}.txt"
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text(texto, encoding="utf-8")
    print(texto)
    print(f"\nGuardado en: {salida}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("ERROR (pégame este texto):")
        print(traceback.format_exc())
        sys.exit(1)
