#!/usr/bin/env python3
"""Reconvierte el modelo ya entrenado (.keras) a .tflite SIN la
cuantizacion dinamica por defecto, que estaba generando un operador
FULLY_CONNECTED version 12 incompatible con TensorFlow Lite 2.14.0
(la version del runtime en el proyecto Android)."""
import tensorflow as tf
from pathlib import Path

model = tf.keras.models.load_model("model/plagascan_mobilenetv2.keras")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# SIN converter.optimizations -- exporta en float32 estandar, sin la
# cuantizacion que causaba el operador incompatible
tflite_model = converter.convert()

out_path = Path("model/plagascan_mobilenetv2.tflite")
out_path.write_bytes(tflite_model)
size_kb = out_path.stat().st_size / 1024
print(f"Guardado: {out_path} ({size_kb:.0f} KB, sin cuantizar)")
