#!/usr/bin/env python3
"""Prueba de regresion: confirma que el modelo de 3 clases sigue
clasificando correctamente las imagenes de frijol que ya se usaban
como prueba en la app Android (test_sano.jpg, test_plaga.jpg)."""
import numpy as np
import tensorflow as tf
from pathlib import Path

IMG_SIZE = 224
CLASS_NAMES = ["no_frijol", "plaga", "sano"]

model = tf.keras.models.load_model("model/plagascan_mobilenetv2.keras")

# Busca los assets originales dentro del proyecto Android
candidatos = [
    Path("~/AndroidStudioProjects/proyecto/app/src/main/assets/test_sano.jpg").expanduser(),
    Path("~/AndroidStudioProjects/proyecto/app/src/main/assets/test_plaga.jpg").expanduser(),
]

for img_path in candidatos:
    if not img_path.exists():
        print(f"[!] No encontrado: {img_path}")
        continue
    img = tf.keras.utils.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    arr = np.expand_dims(tf.keras.utils.img_to_array(img), axis=0)
    pred = model.predict(arr, verbose=0)[0]
    idx = np.argmax(pred)
    print(f"\n{img_path.name}: {CLASS_NAMES[idx]} ({pred[idx]*100:.2f}%)")
    for c, p in zip(CLASS_NAMES, pred):
        print(f"    {c}: {p*100:.2f}%")
