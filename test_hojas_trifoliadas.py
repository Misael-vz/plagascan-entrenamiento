import numpy as np
import tensorflow as tf
from pathlib import Path

IMG_SIZE = 224
CLASS_NAMES = ["no_frijol", "plaga", "sano"]

def probar(modelo_path, nombre_modelo, fotos):
    model = tf.keras.models.load_model(modelo_path)
    print(f"\n=== {nombre_modelo} ===")
    for foto in fotos:
        img = tf.keras.utils.load_img(foto, target_size=(IMG_SIZE, IMG_SIZE))
        arr = np.expand_dims(tf.keras.utils.img_to_array(img), axis=0)
        pred = model.predict(arr, verbose=0)[0]
        idx = np.argmax(pred)
        print(f"  {foto.name}: {CLASS_NAMES[idx]} ({pred[idx]*100:.1f}%)")

fotos = [
    Path.home() / "Descargas/pruevas de campo/IMG_20260904_141440492.jpg",
    Path.home() / "Descargas/pruevas de campo/IMG_20260904_141459286.jpg",
    Path.home() / "Descargas/pruevas de campo/IMG_20260904_141503878.jpg",
]

probar("model/plagascan_mobilenetv2.keras", "MODELO ACTUAL (v3, en PlagaScan)", fotos)
probar("model_experimento/plagascan_mobilenetv2.keras", "MODELO EXPERIMENTAL (en PlagaScan Dev)", fotos)
