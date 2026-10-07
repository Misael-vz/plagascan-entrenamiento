#!/bin/bash
# Extrae el log y las fotos de Recoleccion del dia, las guarda en una
# carpeta con fecha, y genera un collage para revision rapida.
set -e

FECHA=$(date +%Y%m%d)
DESTINO="$HOME/plagascan/recoleccion_campo/$FECHA"
mkdir -p "$DESTINO/frijol" "$DESTINO/otros"

echo "=== Extrayendo log ==="
adb exec-out run-as com.example.proyecto cat files/ground_truth_log.csv > "$DESTINO/ground_truth_log.csv"

echo "=== Extrayendo fotos de frijol ==="
for archivo in $(adb shell run-as com.example.proyecto ls files/ground_truth_captures/frijol/ 2>/dev/null | tr -d '\r'); do
    adb exec-out run-as com.example.proyecto cat "files/ground_truth_captures/frijol/$archivo" > "$DESTINO/frijol/$archivo"
done

echo "=== Extrayendo fotos de otros ==="
for archivo in $(adb shell run-as com.example.proyecto ls files/ground_truth_captures/otros/ 2>/dev/null | tr -d '\r'); do
    adb exec-out run-as com.example.proyecto cat "files/ground_truth_captures/otros/$archivo" > "$DESTINO/otros/$archivo"
done

echo ""
echo "=== Resumen del dia $FECHA ==="
echo "Frijol: $(ls "$DESTINO/frijol" 2>/dev/null | wc -l) fotos"
echo "Otros:  $(ls "$DESTINO/otros" 2>/dev/null | wc -l) fotos"
echo "Guardado en: $DESTINO"
