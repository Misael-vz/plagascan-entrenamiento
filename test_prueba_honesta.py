import numpy as np
import tensorflow as tf
from pathlib import Path

IMG_SIZE = 224
CLASS_NAMES = ["no_frijol", "plaga", "sano"]

model = tf.keras.models.load_model("model_experimento/plagascan_mobilenetv2.keras")

print("\n=== PRUEBA HONESTA: 4 fotos NUNCA vistas por el modelo ===\n")
aciertos = 0
for img_path in sorted(Path("evidencia_campo_frijol/prueba_honesta").glob("*.jpg")):
    img = tf.keras.utils.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    arr = np.expand_dims(tf.keras.utils.img_to_array(img), axis=0)
    pred = model.predict(arr, verbose=0)[0]
    idx = np.argmax(pred)
    correcto = "CORRECTO" if CLASS_NAMES[idx] != "no_frijol" else "SIGUE FALLANDO"
    if CLASS_NAMES[idx] != "no_frijol":
        aciertos += 1
    print(f"{img_path.name}: {CLASS_NAMES[idx]} ({pred[idx]*100:.1f}%) -- {correcto}")

print(f"\nResultado: {aciertos}/4 reconocidas como frijol (antes del experimento: 0/4 esperado)")
