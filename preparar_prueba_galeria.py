#!/usr/bin/env python3
"""
preparar_prueba_galeria.py -- Prepara la prueba EN EL TELEFONO con fotos reales
de validacion (las que el modelo NO vio al entrenar), para analizarlas desde
la galeria de la app.

Por que estas fotos y no imagenes de Google: cada una tiene fuente y licencia
registradas en dataset/metadata.csv, asi que la prueba es trazable.

No toca la app, el modelo ni los datasets. Crea:
  ~/plagascan/prueba_galeria/
      PlagaScan_plaga/plaga_01.jpg ...   (todas las de val/plaga)
      PlagaScan_sano/sano_01.jpg ...     (todas las de val/sano)
      esperado_pc.csv   resultado de cada foto en la computadora, con el
                        modelo de la app y el flujo de la app (para comparar
                        despues contra lo que diga el telefono) + fuente y licencia

Las fotos se vuelven a guardar como JPEG (sin datos EXIF) para que la
computadora y el telefono lean exactamente el mismo archivo.

Uso (dentro de ~/plagascan, con el venv activo):
    python ~/Descargas/preparar_prueba_galeria.py 2>/dev/null
"""
import csv
import os
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2
import numpy as np

PS = Path.home() / "plagascan"
sys.path.insert(0, str(PS))
import eval_tflite_app as ev  # mismo flujo y mismo modelo que la app

DESTINO = PS / "prueba_galeria"
UMBRAL = 0.80


def main():
    if DESTINO.exists():
        print(f"NO SE HIZO NADA: ya existe {DESTINO}. Para rehacerlo: rm -r {DESTINO}")
        sys.exit(1)
    with open(PS / "dataset/metadata.csv", newline="", encoding="utf-8") as f:
        filas = [r for r in csv.DictReader(f) if r["split"] == "val" and r["clase"] in ("plaga", "sano")]
    filas.sort(key=lambda r: r["archivo"])
    it = ev.cargar_interprete(ev.APP_MODEL)
    print(f"Modelo de la app: {ev.APP_MODEL} ({ev.APP_MODEL.stat().st_size // 1024} KB)")

    base = time.time() - 3600
    salida, conteo, n = [], {"plaga": Counter(), "sano": Counter()}, Counter()
    for r in filas:
        origen = ev.resolver(r["origen_local"])
        img = cv2.imread(str(origen))
        if img is None:
            print(f"  [!] no se pudo leer {origen}; se omite")
            continue
        clase = r["clase"]
        n[clase] += 1
        carpeta = DESTINO / f"PlagaScan_{clase}"
        carpeta.mkdir(parents=True, exist_ok=True)
        destino = carpeta / f"{clase}_{n[clase]:02d}.jpg"
        cv2.imwrite(str(destino), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        marca = base + sum(n.values()) * 60  # fechas distintas: orden estable en la galeria
        os.utime(destino, (marca, marca))

        img224, ratio = ev.pipeline_app(cv2.imread(str(destino)))  # el MISMO archivo que vera el telefono
        if img224 is None:
            resultado, conf, probs = "sin_hoja", 0.0, ""
        else:
            p = ev.predecir(it, img224)
            resultado, conf = ev.CLASSES[int(np.argmax(p))], float(p.max())
            probs = " ".join(f"{v:.4f}" for v in p)
        semaforo = "" if resultado in ("sin_hoja", "no_frijol") else (
            ("ROJO" if resultado == "plaga" else "VERDE") if conf >= UMBRAL else "AMARILLO")
        conteo[clase][f"{resultado} {semaforo}".strip()] += 1
        salida.append(dict(archivo=f"{carpeta.name}/{destino.name}", clase_real=clase, resultado_pc=resultado,
                           confianza_pc=round(conf, 4), semaforo_pc=semaforo, probs_no_frijol_plaga_sano=probs,
                           area_ratio=round(ratio, 4), fuente=r.get("fuente", ""), licencia=r.get("licencia", ""),
                           origen_local=r["origen_local"]))

    with open(DESTINO / "esperado_pc.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(salida[0].keys()))
        w.writeheader()
        w.writerows(salida)

    print(f"\nFotos preparadas en {DESTINO}:")
    for clase in ("plaga", "sano"):
        print(f"  PlagaScan_{clase}: {n[clase]} fotos  -> en la computadora: "
              + ", ".join(f"{k}: {v}" for k, v in sorted(conteo[clase].items())))
    print("\nLicencias de las fotos:", dict(Counter(s["licencia"] for s in salida)))
    print(f"Resultado esperado por foto: {DESTINO / 'esperado_pc.csv'}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("ERROR (pégame este texto):")
        print(traceback.format_exc())
        sys.exit(1)
