#!/usr/bin/env python3
"""
PlagaScan — infer_pipeline.py

Módulo 2, Paso 2 (cierre): Pipeline de inferencia completo, de punta a
punta, tal como correría en producción dentro de la app:

    foto cruda -> OpenCV (segmentación + recorte + letterbox) -> MobileNetV2 -> resultado

Reutiliza segment_leaf() y letterbox_resize() de preprocess_opencv.py, y el
modelo .keras entrenado en el Módulo 1.

Uso:
    python infer_pipeline.py --image ruta/a/foto.jpg
    python infer_pipeline.py --batch ~/plagascan/dataset/val/plaga --n 6

Salida:
    Imprime el resultado semáforo (verde/amarillo/rojo) por imagen, tal
    como lo describe RF02/RF03/RF07 del proyecto, y genera un mosaico de
    control con: original -> recortada por OpenCV -> predicción.
"""

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent))
from preprocess_opencv import segment_leaf, letterbox_resize, TARGET_SIZE  # noqa: E402

# Umbrales de confianza para el semáforo, alineados con el documento de
# requerimientos del proyecto (RF02: alerta temprana con niveles de riesgo)
CONFIDENCE_HIGH = 0.80   # confianza alta -> semáforo definido (verde o rojo)
CONFIDENCE_LOW = 0.60    # por debajo de esto -> amarillo, "posible plaga" / revisar


def classify(model, img_bgr):
    """Corre el pipeline completo sobre una imagen ya cargada (BGR, cv2)."""
    cropped, mask, found = segment_leaf(img_bgr)
    final_bgr = letterbox_resize(cropped)

    # El modelo espera RGB (Keras/TF convención), cv2 carga en BGR
    final_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    batch = np.expand_dims(final_rgb, axis=0)

    score = float(model.predict(batch, verbose=0)[0][0])
    # orden de clases confirmado en entrenamiento: ['plaga', 'sano']
    # score >= 0.5 -> sano ; score < 0.5 -> plaga
    pred_class = "sano" if score >= 0.5 else "plaga"
    confianza = score if pred_class == "sano" else 1 - score

    # Lógica de semáforo (RF02/RF03/RF07)
    if pred_class == "sano" and confianza >= CONFIDENCE_HIGH:
        semaforo = "VERDE (bajo riesgo)"
    elif pred_class == "plaga" and confianza >= CONFIDENCE_HIGH:
        semaforo = "ROJO (alta probabilidad de plaga)"
    else:
        semaforo = "AMARILLO (posible plaga / confianza baja, revisar en campo)"

    return {
        "pred_class": pred_class,
        "confianza": confianza,
        "semaforo": semaforo,
        "found_segmentation": found,
        "cropped": cropped,
        "final": final_bgr,
    }


def make_result_grid(results, out_path):
    rows = []
    for name, original, result in results:
        thumb = 190
        orig_t = cv2.resize(original, (thumb, thumb))
        final_t = cv2.resize(result["final"], (thumb, thumb))

        color_map = {"VERDE": (0, 160, 0), "AMARILLO": (0, 200, 200), "ROJO": (0, 0, 200)}
        bar_color = next((c for k, c in color_map.items() if k in result["semaforo"]), (100, 100, 100))

        info = np.zeros((thumb, 260, 3), dtype=np.uint8)
        cv2.putText(info, f"pred: {result['pred_class']}", (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(info, f"confianza: {result['confianza']*100:.1f}%", (8, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.rectangle(info, (8, 80), (250, 115), bar_color, -1)
        semaforo_short = result["semaforo"].split(" ")[0]
        cv2.putText(info, semaforo_short, (15, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        seg_note = "segmentado OK" if result["found_segmentation"] else "fallback (sin recorte)"
        cv2.putText(info, seg_note, (8, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

        for t, label in [(orig_t, "original"), (final_t, "input al modelo")]:
            cv2.rectangle(t, (0, 0), (thumb, 20), (0, 0, 0), -1)
            cv2.putText(t, label, (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        row = np.hstack([orig_t, final_t, info])
        name_bar = np.zeros((22, row.shape[1], 3), dtype=np.uint8)
        cv2.putText(name_bar, name, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        rows.append(np.vstack([name_bar, row]))

    grid = np.vstack(rows)
    cv2.imwrite(str(out_path), grid)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str)
    parser.add_argument("--batch", type=str)
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--model", type=str, default="~/plagascan/model/plagascan_mobilenetv2.keras")
    parser.add_argument("--out", type=str, default="~/plagascan/model/pipeline_check.png")
    args = parser.parse_args()

    model_path = Path(args.model).expanduser()
    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Cargando modelo...")
    model = tf.keras.models.load_model(model_path)

    results = []

    if args.image:
        img_path = Path(args.image).expanduser()
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[!] No se pudo leer: {img_path}")
            return
        result = classify(model, img)
        results.append((img_path.name, img, result))

    elif args.batch:
        batch_dir = Path(args.batch).expanduser()
        files = list(batch_dir.glob("*.jpg")) + list(batch_dir.glob("*.jpeg"))
        random.seed(11)
        chosen = random.sample(files, min(args.n, len(files)))
        for f in chosen:
            img = cv2.imread(str(f))
            if img is None:
                continue
            result = classify(model, img)
            results.append((f.name, img, result))
    else:
        print("Debes pasar --image o --batch")
        return

    print(f"\n{'Archivo':<35} {'Predicción':<8} {'Confianza':<10} Semáforo")
    print("-" * 90)
    for name, _, r in results:
        print(f"{name:<35} {r['pred_class']:<8} {r['confianza']*100:>6.1f}%   {r['semaforo']}")

    make_result_grid(results, out_path)
    print(f"\nGuardado: {out_path}")


if __name__ == "__main__":
    main()
