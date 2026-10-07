import csv
from collections import defaultdict

with open("/tmp/ground_truth_log.csv") as f:
    filas = list(csv.DictReader(f))

# Agrupar por rango de confianza, comparar prediccion vs verdad
rangos = defaultdict(lambda: {"correctas": 0, "total": 0})
for fila in filas:
    if fila["modo"] not in ("otros", "frijol"):
        continue
    conf = float(fila["modelo_confianza"])
    rango = f"{int(conf*10)*10}-{int(conf*10)*10+10}%"
    es_frijol_real = fila["etiqueta_confirmada"] in ("frijol",)
    predijo_frijol = fila["modelo_predijo"] != "no_frijol"
    correcto = (es_frijol_real == predijo_frijol)
    rangos[rango]["total"] += 1
    if correcto:
        rangos[rango]["correctas"] += 1

print("Rango de confianza | Aciertos reales | % acierto")
for rango in sorted(rangos.keys()):
    d = rangos[rango]
    pct = (d["correctas"] / d["total"]) * 100 if d["total"] else 0
    print(f"{rango:18} | {d['correctas']}/{d['total']:14} | {pct:.0f}%")
