#!/usr/bin/env python3
"""
PlagaScan — reconvert_tflite_compatible.py

Re-exporta el modelo ya entrenado (plagascan_mobilenetv2.keras) a formato
.tflite SIN cuantización, para máxima compatibilidad con runtimes de
Android más antiguos.

Motivo: la conversión original (en train_model.py) usaba
converter.optimizations = [tf.lite.Optimize.DEFAULT], que activa
cuantización con escala por canal (per-channel quantization). Esto genera
operaciones internas de versión alta (ej. FULLY_CONNECTED v12) que
tensorflow-lite para Android (incluso en versiones recientes como 2.17)
no siempre soporta, causando:

  "Didn't find op for builtin opcode 'FULLY_CONNECTED' version '12'"

No requiere reentrenar el modelo — solo re-exportarlo desde el .keras ya
guardado.

Uso:
    python reconvert_tflite_compatible.py
"""

from pathlib import Path
import tensorflow as tf

MODEL_PATH = Path("~/plagascan/model/plagascan_mobilenetv2.keras").expanduser()
OUT_PATH = Path("~/plagascan/model/plagascan_mobilenetv2_compat.tflite").expanduser()


def main():
    print(f"Cargando modelo desde: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # SIN converter.optimizations -> exporta en float32 plano, sin
    # cuantización, usando únicamente operaciones de versión baja/estable,
    # ampliamente compatibles con cualquier runtime de TensorFlow Lite.
    tflite_model = converter.convert()

    OUT_PATH.write_bytes(tflite_model)
    size_kb = OUT_PATH.stat().st_size / 1024
    print(f"Guardado: {OUT_PATH} ({size_kb:.0f} KB)")
    print("\nCompara con el original:")
    original = Path("~/plagascan/model/plagascan_mobilenetv2.tflite").expanduser()
    if original.exists():
        orig_kb = original.stat().st_size / 1024
        print(f"  Original (cuantizado):     {orig_kb:.0f} KB")
        print(f"  Nuevo (compatible, float32): {size_kb:.0f} KB")


if __name__ == "__main__":
    main()
