#!/usr/bin/env python3
"""Prueba directa del nuevo modelo de 3 clases sobre la foto de la planta
ornamental que causo el falso positivo con el modelo binario anterior."""
import numpy as np
import tensorflow as tf
from pathlib import Path

IMG_SIZE = 224
CLASS_NAMES = ["no_frijol", "plaga", "sano"]  # orden alfabetico confirmado en el log

model = tf.keras.models.load_model("model/plagascan_mobilenetv2.keras")

img_path = Path("test_fuera_dominio/planta_ornamental.jpg")
img = tf.keras.utils.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
arr = tf.keras.utils.img_to_array(img)
arr = np.expand_dims(arr, axis=0)

pred = model.predict(arr, verbose=0)[0]
print("\n=== RESULTADO ===")
for clase, prob in zip(CLASS_NAMES, pred):
    marca = " <-- PREDICCION" if prob == pred.max() else ""
    print(f"  {clase}: {prob:.4f} ({prob*100:.2f}%){marca}")
