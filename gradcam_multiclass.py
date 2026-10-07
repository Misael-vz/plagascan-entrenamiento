#!/usr/bin/env python3
"""Grad-CAM adaptado para el modelo de 3 clases (softmax). A diferencia
de gradcam_check.py (binario), aqui hay que especificar sobre cual de
las 3 clases calcular el mapa de calor -- se usa la clase PREDICHA por
el modelo, para ver en que se fijo al tomar esa decision especifica."""
import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

IMG_SIZE = 224
CLASS_NAMES = ["no_frijol", "plaga", "sano"]


def get_img_array(img_path):
    img = tf.keras.utils.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    array = tf.keras.utils.img_to_array(img)
    return np.expand_dims(array, axis=0), img


def find_last_conv_layer(model):
    base = None
    for layer in model.layers:
        if "mobilenetv2" in layer.name:
            base = layer
            break
    for layer in reversed(base.layers):
        if len(layer.output.shape) == 4:
            return base, layer.name
    raise ValueError("No se encontro capa convolucional.")


def make_gradcam_heatmap(img_array, model, base_model_name, last_conv_layer_name):
    base_model = model.get_layer(base_model_name)
    grad_model = tf.keras.models.Model(
        base_model.inputs, [base_model.get_layer(last_conv_layer_name).output, base_model.output]
    )
    preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
    preprocessed = preprocess(img_array)

    with tf.GradientTape() as tape:
        conv_outputs, base_outputs = grad_model(preprocessed)
        tape.watch(conv_outputs)
        gap = tf.keras.layers.GlobalAveragePooling2D()(base_outputs)
        preds = None
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Dense):
                preds = layer(gap)
        pred_idx = tf.argmax(preds[0])
        class_channel = preds[:, pred_idx]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), preds.numpy()[0], int(pred_idx.numpy())


def overlay_heatmap(img, heatmap, alpha=0.45):
    heatmap_resized = np.uint8(255 * heatmap)
    jet = plt.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_resized]
    jet_heatmap = tf.keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.size[0], img.size[1]))
    jet_heatmap = tf.keras.utils.img_to_array(jet_heatmap)
    img_array = tf.keras.utils.img_to_array(img)
    superimposed = jet_heatmap * alpha + img_array
    return tf.keras.utils.array_to_img(superimposed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--model", type=str, default="model/plagascan_mobilenetv2.keras")
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model)
    base_model, last_conv = find_last_conv_layer(model)

    img_array, img = get_img_array(args.image)
    heatmap, preds, pred_idx = make_gradcam_heatmap(img_array, model, base_model.name, last_conv)
    overlaid = overlay_heatmap(img, heatmap)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(overlaid)
    conf = preds[pred_idx]
    ax.set_title(f"{Path(args.image).name} -> {CLASS_NAMES[pred_idx]} ({conf:.1%})", fontsize=11)
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Prediccion: {CLASS_NAMES[pred_idx]} ({conf:.2%})")
    print(f"Guardado: {args.out}")


if __name__ == "__main__":
    main()
