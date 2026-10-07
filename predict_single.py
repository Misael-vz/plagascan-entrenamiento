#!/usr/bin/env python3
"""
PlagaScan — predict_single.py

Prueba de sanidad final: corre el modelo entrenado sobre una imagen suelta,
ajena por completo al dataset de entrenamiento/validación. Es la prueba más
honesta de qué tan bien generaliza el modelo antes de integrarlo a la app.

Uso:
    python predict_single.py --image ~/plagascan/pruebas_campo/prueba1.jpeg
"""

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

IMG_SIZE = 224


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--model", type=str, default="~/plagascan/model/plagascan_mobilenetv2.keras")
    args = parser.parse_args()

    img_path = Path(args.image).expanduser()
    model_path = Path(args.model).expanduser()

    if not img_path.exists():
        print(f"[!] No se encontró la imagen: {img_path}")
        return

    model = tf.keras.models.load_model(model_path)

    img = tf.keras.utils.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    array = tf.keras.utils.img_to_array(img)
    array = np.expand_dims(array, axis=0)

    pred_score = float(model.predict(array, verbose=0)[0][0])

    # orden de clases confirmado en el log de entrenamiento: ['plaga', 'sano']
    # label 0 = plaga, label 1 = sano -> score alto = sano, score bajo = plaga
    pred_class = "sano" if pred_score >= 0.5 else "plaga"
    confianza = pred_score if pred_class == "sano" else 1 - pred_score

    print(f"\nImagen: {img_path.name}")
    print(f"Predicción: {pred_class}")
    print(f"Confianza: {confianza * 100:.1f}%")
    print(f"(score crudo del modelo: {pred_score:.4f}, donde >=0.5 = sano, <0.5 = plaga)")

    if confianza < 0.65:
        print("\n[!] Confianza baja (<65%) — el modelo duda. Es una señal legítima")
        print("    de que esta imagen se aleja bastante de lo que vio en entrenamiento.")


if __name__ == "__main__":
    main()
