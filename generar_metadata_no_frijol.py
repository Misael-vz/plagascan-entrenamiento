#!/usr/bin/env python3
import csv
from pathlib import Path

consolidado = Path("dataset_no_frijol/consolidado")
salida = Path("dataset_no_frijol/metadata_no_frijol.csv")

filas = []
for img in sorted(consolidado.glob("*.jpg")):
    partes = img.stem.rsplit("_", 1)
    clase_original = partes[0].replace("_", " ") if len(partes) == 2 else img.stem
    filas.append({
        "filename": img.name,
        "clase_asignada": "no_frijol",
        "clase_original_plantdoc": clase_original,
        "fuente": "PlantDoc Dataset (Singh et al., 2020)",
        "licencia": "CC BY 4.0",
        "url_origen": "https://github.com/pratikkayal/PlantDoc-Dataset",
        "cita": "Singh, D., Jain, N., Jain, P., Kayal, P., Kumawat, S., & Batra, N. (2020). PlantDoc: A Dataset for Visual Plant Disease Detection. Proceedings of the 7th ACM IKDD CoDS and 25th COMAD, 249-253. https://doi.org/10.1145/3371158.3371196",
    })

with open(salida, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=filas[0].keys())
    writer.writeheader()
    writer.writerows(filas)

print(f"Metadata generada: {salida}")
print(f"Total de registros: {len(filas)}")
