#!/usr/bin/env python3
"""
PlagaScan -- demo_gradcam.py  (EXPERIMENTO, rama `experimento`)

Lamina de demostracion: "en que se fija el modelo". Para cada imagen muestra
  1) lo que recibe el modelo (la hoja segmentada y llevada a 224x224, con el
     MISMO procedimiento que LeafSegmenter.kt en el telefono),
  2) el mapa de calor Grad-CAM de la clase que eligio el modelo,
  3) las probabilidades de las 3 clases.

Grupos:
  - "plaga (base de datos)" y "sano (base de datos)": imagenes de val/ del
    dataset_v7 (no se usaron para entrenar).
  - "imitaciones": fotos propias con senuelos (papelitos, puntos, etc.).

Solo LEE el modelo y las imagenes; no modifica nada.

Uso:
    source venv/bin/activate
    python demo_gradcam.py --extra senuelos/ --n 4 --out demo_gradcam.png

Basado en gradcam_multiclass.py (misma reconstruccion del Grad-CAM).
"""
import argparse
import csv
import random
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

IMG_SIZE = 224
CLASS_NAMES = ["no_frijol", "plaga", "sano"]
COLORES_CLASE = ["#8a8a8a", "#c0392b", "#2e7d4f"]

# --- Segmentacion: mismos valores que LeafSegmenter.kt (telefono) ---
HUE_MIN, HUE_MAX, SAT_MIN, VAL_MIN = 25, 95, 30, 30
MIN_AREA_RATIO = 0.15
MARGIN_RATIO = 0.08


def letterbox(img, size=IMG_SIZE):
    h, w = img.shape[:2]
    s = size / max(h, w)
    nh, nw = int(round(h * s)), int(round(w * s))
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    top, left = (size - nh) // 2, (size - nw) // 2
    return cv2.copyMakeBorder(r, top, size - nh - top, left, size - nw - left, cv2.BORDER_REFLECT)


def entrada_como_telefono(path):
    """Devuelve (rgb_224, hoja_encontrada). Si la imagen ya es de 224 px (como
    las del dataset), se usa tal cual."""
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise ValueError(f"No se pudo leer {path}")
    if max(bgr.shape[:2]) <= IMG_SIZE:
        return cv2.cvtColor(letterbox(bgr), cv2.COLOR_BGR2RGB), True
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (HUE_MIN, SAT_MIN, VAL_MIN), (HUE_MAX, 255, 255))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    encontrada = False
    recorte = bgr
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) / (bgr.shape[0] * bgr.shape[1]) >= MIN_AREA_RATIO:
            x, y, w, h = cv2.boundingRect(c)
            mx, my = int(w * MARGIN_RATIO), int(h * MARGIN_RATIO)
            recorte = bgr[max(0, y - my):min(bgr.shape[0], y + h + my), max(0, x - mx):min(bgr.shape[1], x + w + mx)]
            encontrada = True
    return cv2.cvtColor(letterbox(recorte), cv2.COLOR_BGR2RGB), encontrada


def find_last_conv_layer(model):
    base = next(l for l in model.layers if "mobilenetv2" in l.name)
    for layer in reversed(base.layers):
        if len(layer.output.shape) == 4:
            return base, layer.name
    raise ValueError("No se encontro capa convolucional.")


def gradcam(rgb, model, base, last_conv):
    x = np.expand_dims(rgb.astype(np.float32), 0)
    grad_model = tf.keras.models.Model(base.inputs, [base.get_layer(last_conv).output, base.output])
    pre = tf.keras.applications.mobilenet_v2.preprocess_input(x.copy())
    with tf.GradientTape() as tape:
        conv_out, base_out = grad_model(pre)
        tape.watch(conv_out)
        gap = tf.keras.layers.GlobalAveragePooling2D()(base_out)
        preds = None
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Dense):
                preds = layer(gap)
        idx = int(tf.argmax(preds[0]))
        canal = preds[:, idx]
    grads = tape.gradient(canal, conv_out)
    pesos = tf.reduce_mean(grads, axis=(0, 1, 2))
    hm = tf.squeeze(conv_out[0] @ pesos[..., tf.newaxis])
    hm = tf.maximum(hm, 0) / (tf.reduce_max(hm) + 1e-8)
    # Probabilidades del modelo COMPLETO (las que veria la app), para verificar
    # que la reconstruccion del Grad-CAM coincide con el modelo real.
    probs_modelo = model.predict(x, verbose=0)[0]
    return hm.numpy(), preds.numpy()[0], probs_modelo


def superponer(rgb, hm, alpha=0.45):
    hm = cv2.resize(hm, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_CUBIC)
    color = (plt.get_cmap("jet")(np.clip(hm, 0, 1))[..., :3] * 255).astype(np.float32)
    return np.clip(rgb.astype(np.float32) * (1 - alpha) + color * alpha, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model_v7/plagascan_mobilenetv2.keras")
    ap.add_argument("--dataset", default="dataset_v7/val")
    ap.add_argument("--n", type=int, default=4, help="imagenes por clase del dataset (plaga y sano)")
    ap.add_argument("--extra", default=None, help="carpeta con fotos propias (imitaciones)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="demo_gradcam.png")
    a = ap.parse_args()

    model = tf.keras.models.load_model(a.model)
    base, last_conv = find_last_conv_layer(model)
    ds = Path(a.dataset).expanduser()
    random.seed(a.seed)
    items = []
    for clase in ("plaga", "sano"):
        fs = sorted(p for p in (ds / clase).glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        for p in random.sample(fs, min(a.n, len(fs))):
            items.append((f"{clase} (base de datos)", p))
    if a.extra:
        for p in sorted(Path(a.extra).expanduser().glob("*")):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                items.append(("imitación (foto propia)", p))

    filas, n = [], len(items)
    fig, axes = plt.subplots(n, 3, figsize=(9.6, 3.1 * n), gridspec_kw={"width_ratios": [1, 1, 1.15]})
    axes = np.atleast_2d(axes)
    for (grupo, p), (ax1, ax2, ax3) in zip(items, axes):
        rgb, encontrada = entrada_como_telefono(p)
        hm, probs_cam, probs = gradcam(rgb, model, base, last_conv)
        idx = int(np.argmax(probs))
        if int(np.argmax(probs_cam)) != idx or np.abs(probs_cam - probs).max() > 0.02:
            print(f"  [!] {p.name}: el Grad-CAM no coincide con el modelo completo "
                  f"({np.round(probs_cam, 3)} vs {np.round(probs, 3)}); revisar antes de usar la figura")
        ax1.imshow(rgb); ax1.set_title(f"{grupo}\n{p.name[:28]}", fontsize=8); ax1.axis("off")
        ax2.imshow(superponer(rgb, hm)); ax2.axis("off")
        ax2.set_title(f"Dónde se fijó para decir\n«{CLASS_NAMES[idx]}»", fontsize=8)
        ax3.barh(CLASS_NAMES, probs, color=COLORES_CLASE)
        ax3.set_xlim(0, 1); ax3.invert_yaxis(); ax3.tick_params(labelsize=8)
        for i, v in enumerate(probs):
            ax3.text(min(v + 0.02, 0.8), i, f"{v:.0%}", va="center", fontsize=8)
        if not encontrada:
            ax3.set_title("(la app no encontraría la hoja: sin recorte)", fontsize=7, color="#a04000")
        filas.append([grupo, str(p), encontrada] + [f"{v:.4f}" for v in probs] + [CLASS_NAMES[idx]])
        print(f"{grupo:26s} {p.name:34s} -> {CLASS_NAMES[idx]:9s} " + "  ".join(f"{c}={v:.2f}" for c, v in zip(CLASS_NAMES, probs)))

    fig.suptitle("PlagaScan · en qué se fija el modelo (Grad-CAM, model_v7)", fontsize=11)
    fig.tight_layout()
    fig.savefig(a.out, dpi=140)
    with open(Path(a.out).with_suffix(".csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["grupo", "archivo", "hoja_encontrada"] + [f"p_{c}" for c in CLASS_NAMES] + ["prediccion"])
        w.writerows(filas)
    print(f"\nGuardado: {a.out} y {Path(a.out).with_suffix('.csv')}")


if __name__ == "__main__":
    main()
