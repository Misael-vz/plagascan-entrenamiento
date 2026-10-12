#!/usr/bin/env python3
"""
calibrar_semaforo.py -- Experimento de SOLO LECTURA: ¿con que confianza debe
ponerse el semaforo en rojo y en verde?

Hoy la app usa 0.80 para los dos colores. Ese valor se eligio sin datos (asi
quedo documentado como limitacion en el Reporte Final). Aqui se mide, con el
modelo que tiene la app y SOLO con fotos que el modelo no vio al entrenar:

  ROJO  (plaga con confianza >= umbral): cuantas alarmas rojas son correctas y
        cuantas son falsas (frijol sano u otra cosa que no es plaga).
  VERDE (sano con confianza >= umbral): cuantas hojas sanas salen en verde y
        cuantas fotos que NO son frijol sano se cuelan como verde.

Lo que no llega al umbral se queda en amarillo ("revisa de nuevo"), asi que
subir un umbral cambia errores graves por avisos amarillos.

No toca la app, el modelo ni los datasets.

Uso (dentro de ~/plagascan, con el venv activo):
    python ~/Descargas/calibrar_semaforo.py 2>/dev/null
"""
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2
import numpy as np

PS = Path.home() / "plagascan"
sys.path.insert(0, str(PS))
import eval_tflite_app as ev  # mismo flujo y mismo modelo que la app

UMBRALES = (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.97, 0.99)
EXT = (".jpg", ".jpeg", ".png")
EXTRAS = [("prueba_oct/frijol", "sano", "frijol propio"), ("prueba_oct/otros", "no_frijol", "otros"),
          ("campo_nuevo/files/ground_truth_captures/frijol", "sano", "frijol propio"),
          ("campo_nuevo/files/ground_truth_captures/otros", "no_frijol", "otros"),
          ("prueba_limon/limon_sano", "no_frijol", "limón sano"),
          ("prueba_limon/limon_pulgon", "plaga", "limón con pulgón")]


def reunir():
    """[(tipo, clase esperada, ruta)] -- solo fotos que el modelo no vio al entrenar."""
    items, vistos = [], set()
    usadas = {p.name for c in ("evidencia_campo_frijol/entrenar", "evidencia_campo_v2/entrenar")
              for p in (PS / c).glob("*.jpg")}
    for it in ev.reunir_items():
        g = it["grupo"]
        if "/entrenar" in g or g == "assets_app":
            continue
        if g == "val/plaga":
            tipo = "limón con pulgón" if "lelephid" in it.get("nota", "").lower() else "morfo amarillo real"
        elif g == "val/sano":
            tipo = "frijol sano (Makerere)"
        elif it["esperado"] == "sano":
            tipo = "frijol propio"
        else:
            tipo = "otros"
        items.append((tipo, it["esperado"], Path(it["ruta_app"])))
    for carpeta, clase, tipo in EXTRAS:
        for p in sorted(q for q in (PS / carpeta).glob("*") if q.suffix.lower() in EXT):
            if p.name not in usadas:
                items.append((tipo, clase, p))
    unicos = []
    for t, c, p in items:
        if p not in vistos:
            vistos.add(p)
            unicos.append((t, c, p))
    return unicos


def main():
    it = ev.cargar_interprete(ev.APP_MODEL)
    print(f"Modelo de la app: {ev.APP_MODEL} ({ev.APP_MODEL.stat().st_size // 1024} KB)")
    items = reunir()
    print(f"Fotos a revisar: {len(items)}")
    res = []  # (tipo, esperado, clase, conf, archivo)
    sin_hoja = 0
    for tipo, esperado, ruta in items:
        img = cv2.imread(str(ruta))
        if img is None:
            continue
        img224, _ = ev.pipeline_app(img)
        if img224 is None:
            sin_hoja += 1
            continue
        p = ev.predecir(it, img224)
        res.append((tipo, esperado, ev.CLASSES[int(np.argmax(p))], float(p.max()), ruta.name))

    n = {c: sum(r[1] == c for r in res) for c in ("plaga", "sano", "no_frijol")}
    L = [f"# Calibración del semáforo -- {datetime.now():%Y-%m-%d %H:%M}",
         f"Fotos con hoja detectada: {len(res)} (plaga {n['plaga']}, frijol sano {n['sano']}, no frijol {n['no_frijol']}); "
         f"sin hoja: {sin_hoja}", ""]

    L += ["## ROJO: 'Pulgón amarillo detectado' cuando plaga >= umbral",
          f"{'umbral':>7s} | {'rojos correctos':>15s} {'(limón)':>8s} {'(amarillo real)':>15s} | "
          f"{'rojos FALSOS':>12s} {'(en frijol sano)':>16s} {'(en no frijol)':>14s} | {'plaga que queda en amarillo':>27s}"]
    plagas = [r for r in res if r[2] == "plaga"]
    for u in UMBRALES:
        rojos = [r for r in plagas if r[3] >= u]
        ok = [r for r in rojos if r[1] == "plaga"]
        mal = [r for r in rojos if r[1] != "plaga"]
        amar = sum(1 for r in plagas if r[1] == "plaga" and r[3] < u)
        L.append(f"{u:7.2f} | {len(ok):15d} {sum(r[0] == 'limón con pulgón' for r in ok):8d} "
                 f"{sum(r[0] == 'morfo amarillo real' for r in ok):15d} | {len(mal):12d} "
                 f"{sum(r[1] == 'sano' for r in mal):16d} {sum(r[1] == 'no_frijol' for r in mal):14d} | {amar:27d}")

    L += ["", "## VERDE: 'Planta sana' cuando sano >= umbral",
          f"{'umbral':>7s} | {'verdes correctos':>16s} {'(frijol propio)':>15s} | {'verdes FALSOS':>13s} "
          f"{'(era plaga)':>11s} {'(no era frijol)':>15s} | {'sano que queda en amarillo':>26s}"]
    sanos = [r for r in res if r[2] == "sano"]
    for u in UMBRALES:
        verdes = [r for r in sanos if r[3] >= u]
        ok = [r for r in verdes if r[1] == "sano"]
        mal = [r for r in verdes if r[1] != "sano"]
        amar = sum(1 for r in sanos if r[1] == "sano" and r[3] < u)
        L.append(f"{u:7.2f} | {len(ok):16d} {sum(r[0] == 'frijol propio' for r in ok):15d} | {len(mal):13d} "
                 f"{sum(r[1] == 'plaga' for r in mal):11d} {sum(r[1] == 'no_frijol' for r in mal):15d} | {amar:26d}")

    L += ["", "## Todas las predicciones 'plaga' equivocadas (de mayor a menor confianza)"]
    L += [f"  {r[3]:.3f} | {r[0]} | {r[4]}" for r in sorted((r for r in plagas if r[1] != "plaga"), key=lambda r: -r[3])] or ["  ninguna"]
    L += ["", "## Todas las predicciones 'sano' equivocadas (de mayor a menor confianza)"]
    L += [f"  {r[3]:.3f} | {r[0]} (era {r[1]}) | {r[4]}" for r in sorted((r for r in sanos if r[1] != "sano"), key=lambda r: -r[3])] or ["  ninguna"]
    texto = "\n".join(L)
    salida = PS / "eval_app" / f"semaforo_{datetime.now():%Y%m%d_%H%M%S}.txt"
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
