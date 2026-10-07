import numpy as np
import tensorflow as tf
from pathlib import Path

IMG_SIZE = 224
CLASS_NAMES = ["no_frijol", "plaga", "sano"]

model = tf.keras.models.load_model("model_v4/plagascan_mobilenetv2.keras")

carpetas = [
    ("Hojas primarias (sesion 1)", Path("evidencia_campo_frijol/prueba_honesta")),
    ("Hojas nuevas (sesion 2)", Path("evidencia_campo_v2/prueba_honesta")),
]

total_aciertos = 0
total_fotos = 0

for nombre, carpeta in carpetas:
    print(f"\n=== {nombre} ===")
    for img_path in sorted(carpeta.glob("*.jpg")):
        img = tf.keras.utils.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
        arr = np.expand_dims(tf.keras.utils.img_to_array(img), axis=0)
        pred = model.predict(arr, verbose=0)[0]
        idx = np.argmax(pred)
        correcto = CLASS_NAMES[idx] != "no_frijol"
        total_fotos += 1
        if correcto:
            total_aciertos += 1
        marca = "CORRECTO" if correcto else "FALLO"
        print(f"  {img_path.name}: {CLASS_NAMES[idx]} ({pred[idx]*100:.1f}%) -- {marca}")

print(f"\n=== RESULTADO TOTAL: {total_aciertos}/{total_fotos} reconocidas como frijol ===")
