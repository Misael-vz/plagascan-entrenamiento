#!/usr/bin/env python3
"""
PlagaScan — gradcam_check.py

Verificación de sanidad: genera mapas de calor Grad-CAM sobre una muestra
de imágenes de val/ para confirmar en qué parte de la imagen se está
fijando el modelo al clasificar. Si el modelo mira la hoja/insecto, es
señal de que aprendió el patrón correcto. Si mira fondo/bordes/esquinas
de forma consistente, es señal de que aprendió a distinguir la FUENTE de
la imagen en vez del contenido real — un riesgo real dado que sano y
plaga vienen de fuentes con estilos fotográficos muy distintos.

Uso:
    python gradcam_check.py --root ~/plagascan/dataset --model ~/plagascan/model/plagascan_mobilenetv2.keras --n 8
"""

import argparse
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm

IMG_SIZE = 224


def get_img_array(img_path):
    img = tf.keras.utils.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    array = tf.keras.utils.img_to_array(img)
    return np.expand_dims(array, axis=0), img


def find_last_conv_layer(model):
    """Busca la última capa convolucional dentro del submodelo MobileNetV2."""
    base = None
    for layer in model.layers:
        if "mobilenetv2" in layer.name:
            base = layer
            break
    if base is None:
        raise ValueError("No se encontró el submodelo MobileNetV2 dentro del modelo.")
    for layer in reversed(base.layers):
        if len(layer.output.shape) == 4:  # capa con salida espacial (conv)
            return base, layer.name
    raise ValueError("No se encontró ninguna capa convolucional.")


def make_gradcam_heatmap(img_array, model, base_model_name, last_conv_layer_name):
    base_model = model.get_layer(base_model_name)
    grad_model = tf.keras.models.Model(
        base_model.inputs, [base_model.get_layer(last_conv_layer_name).output, base_model.output]
    )

    # Reconstruir el preprocesamiento manual (mismo que en build_model())
    preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
    preprocessed = preprocess(img_array)

    with tf.GradientTape() as tape:
        conv_outputs, base_outputs = grad_model(preprocessed)
        tape.watch(conv_outputs)
        # Pasar base_outputs por el resto del modelo (pooling + dropout + dense)
        x = base_outputs
        for layer in model.layers:
            if layer.name == base_model_name:
                continue
            if "input" in layer.name or "true_divide" in layer.name or "subtract" in layer.name:
                continue
        # Reconstrucción manual del head: GAP -> Dropout(inactivo en inferencia) -> Dense
        gap = tf.keras.layers.GlobalAveragePooling2D()(base_outputs)
        preds = None
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Dense):
                preds = layer(gap)
        class_channel = preds[:, 0]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), float(preds.numpy()[0][0])


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
    parser.add_argument("--root", type=str, default="~/plagascan/dataset")
    parser.add_argument("--model", type=str, default="~/plagascan/model/plagascan_mobilenetv2.keras")
    parser.add_argument("--out", type=str, default="~/plagascan/model/gradcam_check.png")
    parser.add_argument("--n", type=int, default=8, help="Número total de imágenes a revisar (mitad por clase)")
    args = parser.parse_args()

    dataset_root = Path(args.root).expanduser()
    model_path = Path(args.model).expanduser()
    out_path = Path(args.out).expanduser()

    model = tf.keras.models.load_model(model_path)
    base_model, last_conv_layer_name = find_last_conv_layer(model)
    base_model_name = base_model.name
    print(f"Usando última capa conv: {last_conv_layer_name} dentro de {base_model_name}")

    random.seed(7)
    val_dir = dataset_root / "val"
    per_class = args.n // 2
    samples = []
    for class_name in ["sano", "plaga"]:
        files = list((val_dir / class_name).glob("*.jpg"))
        chosen = random.sample(files, min(per_class, len(files)))
        samples.extend([(f, class_name) for f in chosen])

    n_cols = 4
    n_rows = (len(samples) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten() if len(samples) > 1 else [axes]

    for ax, (img_path, true_class) in zip(axes, samples):
        img_array, img = get_img_array(img_path)
        heatmap, pred_score = make_gradcam_heatmap(img_array, model, base_model_name, last_conv_layer_name)
        overlaid = overlay_heatmap(img, heatmap)

        # label 0 = plaga, 1 = sano (alfabético, confirmado en el log de entrenamiento)
        pred_class = "sano" if pred_score >= 0.5 else "plaga"
        confianza = pred_score if pred_class == "sano" else 1 - pred_score

        ax.imshow(overlaid)
        ax.set_title(f"real: {true_class} | pred: {pred_class} ({confianza:.2f})", fontsize=10)
        ax.axis("off")

    for ax in axes[len(samples):]:
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"\nGuardado: {out_path}")
    print("\nRevisa manualmente: el área roja/caliente debe caer sobre la hoja o el")
    print("insecto/colonia, NO consistentemente sobre fondo, bordes, manos o esquinas.")


if __name__ == "__main__":
    main()
