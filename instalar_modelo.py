#!/usr/bin/env python3
"""
instalar_modelo.py -- Pone en la app el modelo de la version indicada (v6, v7...)
para probarlo en el telefono. Sirve tambien para regresar a una version anterior.

Unico archivo que cambia:
    ~/AndroidStudioProjects/proyecto/app/src/main/assets/plagascan_mobilenetv2.tflite
Origen:
    ~/plagascan/model_<version>/plagascan_mobilenetv2_float32.tflite
Antes de cambiarlo respalda el modelo que tenga la app en:
    ~/plagascan_respaldos/<fecha>_modelo_<version>/

No cambia codigo Kotlin: todas las versiones tienen la misma entrada
(224x224x3 float32) y las mismas 3 clases en el mismo orden.

Uso:
    python3 ~/Descargas/instalar_modelo.py v7 && compilar     (probar el v7)
    python3 ~/Descargas/instalar_modelo.py v6 && compilar     (regresar al v6)
"""
import hashlib
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

APP = Path.home() / "AndroidStudioProjects/proyecto/app/src/main/assets/plagascan_mobilenetv2.tflite"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    if len(sys.argv) != 2 or not re.fullmatch(r"v\d+", sys.argv[1]):
        sys.exit("Uso: python3 instalar_modelo.py v7   (o v6, etc.)")
    ver = sys.argv[1]
    origen = Path.home() / f"plagascan/model_{ver}/plagascan_mobilenetv2_float32.tflite"
    if not APP.exists():
        sys.exit(f"NO SE CAMBIÓ NADA: no existe {APP}")
    if not origen.exists():
        sys.exit(f"NO SE CAMBIÓ NADA: no existe {origen}")
    if origen.stat().st_size < 1_000_000 or origen.read_bytes()[4:8] != b"TFL3":
        sys.exit(f"NO SE CAMBIÓ NADA: {origen} no parece un modelo TFLite válido.")
    if abs(origen.stat().st_size - APP.stat().st_size) > 200_000:
        sys.exit(f"NO SE CAMBIÓ NADA: el tamaño del {ver} ({origen.stat().st_size // 1024} KB) es muy distinto "
                 f"al del modelo de la app ({APP.stat().st_size // 1024} KB); ¿es el archivo float32?")
    if sha(origen) == sha(APP):
        sys.exit(f"NO SE CAMBIÓ NADA: la app ya tiene el modelo {ver}.")

    dest = Path.home() / "plagascan_respaldos" / (datetime.now().strftime("%Y%m%d_%H%M") + f"_modelo_{ver}")
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(APP, dest / APP.name)
    anterior = sha(APP)
    shutil.copy2(origen, APP)
    print(f"Listo: la app ahora tiene el modelo {ver}.")
    print(f"  anterior SHA-256: {anterior[:16]}…  respaldo en: {dest}")
    print(f"  {ver:8s} SHA-256: {sha(APP)[:16]}…  ({APP.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
