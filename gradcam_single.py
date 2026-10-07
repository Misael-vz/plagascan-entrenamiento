#!/usr/bin/env python3
"""
PlagaScan — gradcam_single.py

Variante de gradcam_check.py para diagnosticar UNA imagen externa,
fuera del dataset de validacion -- pensado para casos reales reportados
por el usuario (por ejemplo, una planta que no es frijol y que el
modelo clasifico incorrectamente), sin tocar el script original ya
validado.

Uso:
    python gradcam_single.py --image ~/plagascan/test_fuera_dominio/planta_ornamental.jpg
"""

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from gradcam_check import get_img_array, find_last_conv_layer, make_gradcam_heatmap, overlay_heatmap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--model", type=str, default="~/plagascan/model/plagascan_mobilenetv2.keras")
    parser.add_argument("--out", type=str, default="~/plagascan/model/gradcam_single_out.png")
    args = parser.parse_args()

    img_path = Path(args.image).expanduser()
    model_path = Path(args.model).expanduser()
    out_path = Path(args.out).expanduser()

    model = tf.keras.models.load_model(model_path)
    base_model, last_conv_layer_name = find_last_conv_layer(model)
    base_model_name = base_model.name

    img_array, img = get_img_array(img_path)
    heatmap, pred_score = make_gradcam_heatmap(img_array, model, base_model_name, last_conv_layer_name)
    overlaid = overlay_heatmap(img, heatmap)

    pred_class = "sano" if pred_score >= 0.5 else "plaga"
    confianza = pred_score if pred_class == "sano" else 1 - pred_score

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(overlaid)
    ax.set_title(f"pred: {pred_class} ({confianza:.2%}) -- imagen fuera de dominio", fontsize=11)
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)

    print(f"Prediccion: {pred_class} con {confianza:.2%} de confianza")
    print(f"Guardado: {out_path}")
    print("\nRevisa: si el area caliente cae sobre las zonas amarillas de la hoja")
    print("(en vez de sobre textura/forma especifica de daño), confirma que el")
    print("modelo esta usando 'presencia de amarillo' como atajo de clasificacion.")


if __name__ == "__main__":
    main()
