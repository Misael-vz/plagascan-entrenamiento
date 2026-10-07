#!/usr/bin/env python3
"""Identifica exactamente que imagenes del set de validacion fueron
mal clasificadas por el modelo de 3 clases, y por que clase real eran."""
import numpy as np
import tensorflow as tf
from pathlib import Path

IMG_SIZE = 224
CLASS_NAMES = ["no_frijol", "plaga", "sano"]

model = tf.keras.models.load_model("model/plagascan_mobilenetv2.keras")

val_dir = Path("dataset/val")
errores = []
total_por_clase = {}

for clase_idx, clase in enumerate(CLASS_NAMES):
    carpeta = val_dir / clase
    archivos = sorted(carpeta.glob("*.jpg"))
    total_por_clase[clase] = len(archivos)
    for img_path in archivos:
        img = tf.keras.utils.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
        arr = np.expand_dims(tf.keras.utils.img_to_array(img), axis=0)
        pred = model.predict(arr, verbose=0)[0]
        pred_idx = np.argmax(pred)
        if pred_idx != clase_idx:
            errores.append({
                "archivo": img_path.name,
                "real": clase,
                "predicho": CLASS_NAMES[pred_idx],
                "confianza_error": pred[pred_idx],
                "confianza_real": pred[clase_idx],
            })

print(f"\n=== {len(errores)} ERRORES DE {sum(total_por_clase.values())} IMAGENES TOTALES ===\n")
for e in errores:
    print(f"  {e['archivo']}")
    print(f"    Real: {e['real']} | Predicho: {e['predicho']} ({e['confianza_error']:.1%}) "
          f"| Confianza en la real: {e['confianza_real']:.1%}")
