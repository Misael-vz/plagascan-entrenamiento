import csv
from collections import defaultdict

with open("/tmp/ground_truth_log.csv") as f:
    filas = list(csv.DictReader(f))

# Desglosar por el tipo de caso real (otros vs frijol), no mezclados
grupos = defaultdict(lambda: {"correctas": 0, "total": 0})
for fila in filas:
    if fila["modo"] not in ("otros", "frijol"):
        continue
    conf = float(fila["modelo_confianza"])
    rango = f"{int(conf*10)*10}-{int(conf*10)*10+10}%"
    es_frijol_real = fila["etiqueta_confirmada"] == "frijol"
    predijo_frijol = fila["modelo_predijo"] != "no_frijol"
    correcto = (es_frijol_real == predijo_frijol)

    tipo_caso = "ES frijol real" if es_frijol_real else "NO es frijol real"
    clave = f"{tipo_caso} | {rango}"
    grupos[clave]["total"] += 1
    if correcto:
        grupos[clave]["correctas"] += 1

print(f"{'Caso real':22} | {'Confianza':10} | Aciertos | %")
for clave in sorted(grupos.keys()):
    tipo, rango = clave.split(" | ")
    d = grupos[clave]
    pct = (d["correctas"] / d["total"]) * 100 if d["total"] else 0
    print(f"{tipo:22} | {rango:10} | {d['correctas']}/{d['total']:<6} | {pct:.0f}%")
