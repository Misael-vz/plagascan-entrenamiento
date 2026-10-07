# PlagaScan — entrenamiento

Scripts y registros con los que se construyó el dataset y se entrenó el modelo de
**PlagaScan**, la app Android para la detección temprana de pulgón amarillo
(*Aphis gossypii*) en hoja de frijol. El código de la app está en el repositorio
`Misael-vz/plagaScan`.

Proyecto de titulación — Ingeniería en Desarrollo de Software, CETI Plantel Tonalá.

## Qué hay aquí

- **Scripts** (`*.py`, `*.sh`): construcción y ampliación del dataset, entrenamiento,
  conversión del modelo, evaluación y revisiones con Grad-CAM.
  `eval_tflite_app.py` evalúa con el mismo flujo que usa la app (segmentación, recorte
  y 224×224).
- **Decisiones y registros**: `DATASET_DECISIONS.md`, los registros de entrenamiento
  (`train_*log*.txt`) y la carpeta `eval_app/` con los resultados de cada evaluación.
- **Trazabilidad**: `dataset/metadata.csv` y los listados de candidatas, con la fuente
  y la licencia de cada imagen.
- **Modelo final**: `model_v7/`.

## Qué no hay y por qué

- **Las imágenes de los datasets públicos.** Se descargan de su fuente: Makerere iBean
  (MIT), LeLePhid (CC BY 4.0) y observaciones de iNaturalist y Wikimedia con licencia
  Creative Commons.
- **El entorno de Python** (`venv/`) y los paquetes descargados (`fuentes_nuevas/`).
- **Modelos anteriores** y carpetas de pruebas intermedias.

## Entorno

Python 3.12 con TensorFlow, OpenCV, numpy, matplotlib y h5py. No hay archivo de
requisitos; el entorno se creó a mano con `python -m venv venv`.

## Equipo

Brayan Misael Vázquez Mata · Yahir Alejandro Magaña Chávez · Carlos Alberto Sahagún Luna

Asesor metodológico: Rigoberto Ibarra Hernández · Asesor técnico: Enrique Emanuel Vega Rivera
