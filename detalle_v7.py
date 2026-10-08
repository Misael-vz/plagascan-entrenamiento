#!/usr/bin/env python3
"""
detalle_v7.py -- Revision a detalle del modelo v7 ANTES de meterlo a la app.
Compara el modelo que la app tiene hoy (v6) contra el v7, e incluye las fotos
apartadas de limon (prueba_limon/) y separa la plaga de validacion por fuente
(limon / morfo amarillo real).

Solo lectura para la app: no toca ~/AndroidStudioProjects ni ningun dataset.

Hace dos cosas:

1) Convierte el modelo v7 al formato que SI acepta la app (float32, sin
   cuantizar), igual que tu reconvertir_sin_cuantizar.py:
       model_v7/plagascan_mobilenetv2.keras
    -> model_v7/plagascan_mobilenetv2_float32.tflite
   (el .tflite que deja train_model.py va cuantizado y el telefono no lo
   carga: operador FULLY_CONNECTED v12 incompatible con TFLite 2.14).

2) Compara el modelo ACTUAL de la app contra ese archivo float32 -- el mismo
   que se instalaria -- con el flujo de la app, y muestra QUE VERIA EL USUARIO
   en cada foto:
       sin_hoja   la app no encuentra hoja y no analiza
       no_frijol  "no parece frijol"
       sano(V)    sano con confianza >= 0.80  -> semaforo verde
       sano(A)    sano con confianza <  0.80  -> semaforo amarillo
       plaga(A)   plaga con confianza < 0.80  -> semaforo amarillo
       plaga(R)   plaga con confianza >= 0.80 -> semaforo rojo
   En las fotos que NO son frijol cuenta como correcto "sin_hoja" y
   "no_frijol" (comparar_modelos.py contaba "sin_hoja" como error).

Uso (dentro de ~/plagascan, con el venv activo):
    python ~/Descargas/detalle_v7.py 2>/dev/null
"""
import contextlib
import io
import os
import sys
import traceback
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2
import numpy as np

PS = Path.home() / "plagascan"
sys.path.insert(0, str(PS))
import eval_tflite_app as ev  # mismo flujo que la app

KERAS = PS / "model_v7/plagascan_mobilenetv2.keras"
FLOAT32 = PS / "model_v7/plagascan_mobilenetv2_float32.tflite"
UMBRAL = 0.80
COLUMNAS = ["sin_hoja", "no_frijol", "sano(V)", "sano(A)", "plaga(A)", "plaga(R)"]
CORRECTO = {"no_frijol": {"sin_hoja", "no_frijol"}, "sano": {"sano(V)", "sano(A)"}, "plaga": {"plaga(A)", "plaga(R)"}}
EXTRAS = [("prueba_oct/frijol", "sano"), ("prueba_oct/otros", "no_frijol"),
          ("campo_nuevo/files/ground_truth_captures/frijol", "sano"),
          ("campo_nuevo/files/ground_truth_captures/otros", "no_frijol"),
          ("prueba_limon/limon_sano", "no_frijol"), ("prueba_limon/limon_pulgon", "plaga")]


def convertir():
    if FLOAT32.exists():
        print(f"[1] Ya existe {FLOAT32.name} ({FLOAT32.stat().st_size / 1024:.0f} KB); no se vuelve a convertir.")
        return
    if not KERAS.exists():
        print(f"NO SE HIZO NADA: no existe {KERAS}")
        sys.exit(1)
    import tensorflow as tf
    with contextlib.redirect_stdout(io.StringIO()):  # calla el listado de capas de la conversion
        modelo = tf.keras.models.load_model(KERAS)
        conv = tf.lite.TFLiteConverter.from_keras_model(modelo)  # sin optimizations = float32
        datos = conv.convert()
    FLOAT32.write_bytes(datos)
    print(f"[1] Convertido a float32: {FLOAT32} ({FLOAT32.stat().st_size / 1024:.0f} KB)")


def resultado(it, img224):
    if img224 is None:
        return "sin_hoja", 0.0
    p = ev.predecir(it, img224)
    clase, conf = ev.CLASSES[int(np.argmax(p))], float(p.max())
    if clase == "no_frijol":
        return clase, conf
    return f"{clase}({'V' if clase == 'sano' else 'R'})" if conf >= UMBRAL else f"{clase}(A)", conf


def main():
    convertir()
    modelos = {"actual": ev.APP_MODEL, "v7": FLOAT32}
    its = {k: ev.cargar_interprete(v) for k, v in modelos.items()}
    for k, it in its.items():
        d = it.get_input_details()[0]
        print(f"    {k:7s} entrada {d['shape'].tolist()} {d['dtype'].__name__}  ({modelos[k].stat().st_size / 1024:.0f} KB)")

    # Fotos: validacion + pruebas de campo que el modelo NO vio al entrenar
    usadas = {p.name for c in ("evidencia_campo_frijol/entrenar", "evidencia_campo_v2/entrenar")
              for p in (PS / c).glob("*.jpg")}
    items = [it for it in ev.reunir_items() if "/entrenar" not in it["grupo"] and it["grupo"] != "assets_app"]
    for it in items:  # la plaga de validacion, separada por fuente
        if it["grupo"] == "val/plaga":
            it["grupo"] = "val/plaga (limón)" if "lelephid" in it.get("nota", "").lower() else "val/plaga (morfo amarillo)"
    for carpeta, clase in EXTRAS:
        for p in sorted(q for q in (PS / carpeta).glob("*") if q.suffix.lower() in (".jpg", ".jpeg", ".png")):
            if p.name not in usadas:
                items.append(dict(grupo=carpeta.replace("files/ground_truth_captures/", ""), esperado=clase,
                                  archivo=p.name, ruta_app=p))

    conteo = {k: defaultdict(Counter) for k in modelos}
    esperado_de, errores, unicos = {}, [], {k: {} for k in modelos}
    for item in items:
        img = cv2.imread(str(item["ruta_app"]))
        if img is None:
            continue
        img224, _ = ev.pipeline_app(img)
        g, esp = item["grupo"], item["esperado"]
        esperado_de[g] = esp
        for k, it in its.items():
            r, conf = resultado(it, img224)
            conteo[k][g][r] += 1
            unicos[k][(esp, Path(item["archivo"]).name)] = r
            if k == "v7" and r not in CORRECTO[esp]:
                errores.append(f"  {g} | {Path(item['archivo']).name} | esperado={esp} | v7 dice {r} {conf:.2f}")

    L = [f"# Detalle del modelo v7 (float32) contra el de la app (v6), con el flujo de la app -- {datetime.now():%Y-%m-%d %H:%M}",
         f"actual: {modelos['actual']}", f"v7:     {modelos['v7']}", "",
         f"{'grupo':44s} {'esperado':9s} {'modelo':7s} " + " ".join(f"{c:>9s}" for c in COLUMNAS) + "   correcto"]
    for g, esp in esperado_de.items():
        for k in modelos:
            c = conteo[k][g]
            n = sum(c.values())
            ok = sum(c[x] for x in CORRECTO[esp])
            L.append(f"{g if k == 'actual' else '':44s} {esp if k == 'actual' else '':9s} {k:7s} "
                     + " ".join(f"{c[x]:9d}" for x in COLUMNAS) + f"   {ok:3d}/{n:<3d}")
    L += ["", "## Resumen: validación + campo juntos (cada foto contada una sola vez)"]
    for k in modelos:
        u = unicos[k]
        fr = [r for (e, _), r in u.items() if e == "sano"]
        ot = [r for (e, _), r in u.items() if e == "no_frijol"]
        pl = [r for (e, _), r in u.items() if e == "plaga"]
        L += [f"  {k}:",
              f"    frijol sano reconocido:        {sum(r in CORRECTO['sano'] for r in fr)}/{len(fr)}"
              f"   (falsa plaga: {sum(r.startswith('plaga') for r in fr)}, de ellas en rojo: {fr.count('plaga(R)')})",
              f"    no-frijol bien rechazado:      {sum(r in CORRECTO['no_frijol'] for r in ot)}/{len(ot)}"
              f"   (dijo sano: {sum(r.startswith('sano') for r in ot)}; dijo plaga: {sum(r.startswith('plaga') for r in ot)}, en rojo: {ot.count('plaga(R)')})",
              f"    plaga detectada:               {sum(r in CORRECTO['plaga'] for r in pl)}/{len(pl)}"
              f"   (en rojo: {pl.count('plaga(R)')}; dijo sano: {sum(r.startswith('sano') for r in pl)})"]
    L += ["", f"## Errores del v7, foto por foto ({len(errores)}; se muestran hasta 60)"] + errores[:60]
    texto = "\n".join(L)
    salida = PS / "eval_app" / f"detalle_v7_{datetime.now():%Y%m%d_%H%M%S}.txt"
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text(texto, encoding="utf-8")
    print(texto)
    print(f"\nGuardado en: {salida}")


if __name__ == "__main__":
    try:
        main()
    except Exception:  # que el error se vea aunque se oculten los avisos de TensorFlow
        print("ERROR (pégame este texto):")
        print(traceback.format_exc())
        sys.exit(1)
