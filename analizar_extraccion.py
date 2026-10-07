#!/usr/bin/env python3
"""
analizar_extraccion.py -- SOLO LECTURA.
Resume los logs de Recoleccion y el historial Room extraidos del telefono, y
re-evalua las fotos con el TFLite actual de la app (reutiliza eval_tflite_app.py).
Salida: ~/plagascan/eval_app/extraccion_resumen.txt y extraccion_detalle.csv
"""
import csv
import shutil
import sqlite3
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent))
import eval_tflite_app as E  # noqa: E402

PS = Path.home() / "plagascan"
EXTRACCIONES = sorted((PS / "recoleccion_campo").glob("extraccion_*_com.example.proyecto*"))
YA_EN_ENTRENAMIENTO = {p.name for p in PS.glob("evidencia_campo_*/*/*.jpg")}


def fecha(ms, fmt="%Y-%m-%d"):
    return datetime.fromtimestamp(int(ms) / 1000).strftime(fmt)


def esperado_de(fila):
    if fila["modo"] == "otros":
        return "no_frijol"
    return fila["salud_confirmada"] if fila["salud_confirmada"] in ("sano", "plaga") else None


def main():
    it = E.cargar_interprete(E.APP_MODEL)
    out, detalle = [], []

    for d in EXTRACCIONES:
        pkg = d.name.split("_", 2)[2]
        out.append(f"# {pkg}")

        # ---------- Log de recoleccion ----------
        log = d / "files/ground_truth_log.csv"
        if log.exists():
            filas = list(csv.DictReader(open(log, encoding="utf-8")))
            out.append(f"## Log de recoleccion: {len(filas)} filas")
            por_fecha = defaultdict(Counter)
            for f in filas:
                clave = "otros" if f["modo"] == "otros" else f"frijol/{f['salud_confirmada']}"
                por_fecha[fecha(f["timestamp"])][clave] += 1
            for k in sorted(por_fecha):
                out.append(f"  {k}: {dict(por_fecha[k])}")

            grupos = defaultdict(list)
            for f in filas:
                ruta = d / "files" / f["archivo_imagen"].split("/files/", 1)[-1]
                etiqueta = "otros" if f["modo"] == "otros" else f"frijol/{f['salud_confirmada']}"
                visto = Path(ruta).name in YA_EN_ENTRENAMIENTO
                reg = dict(paquete=pkg, fecha=fecha(f["timestamp"], "%Y-%m-%d %H:%M"),
                           archivo=Path(ruta).name, etiqueta=etiqueta,
                           esperado=esperado_de(f) or "", ya_en_entrenamiento=visto,
                           pred_al_capturar=f["modelo_predijo"],
                           conf_al_capturar=f["modelo_confianza"])
                img = cv2.imread(str(ruta))
                if img is None:
                    reg["v4_entrenamiento"] = reg["v4_app"] = "SIN_ARCHIVO"
                else:
                    p = E.predecir(it, E.pipeline_entrenamiento(img, False))
                    reg["v4_entrenamiento"] = E.CLASSES[int(p.argmax())]
                    reg["conf_v4_entrenamiento"] = round(float(p.max()), 4)
                    img224, ratio = E.pipeline_app(img)
                    reg["area_ratio"] = round(ratio, 4)
                    if img224 is None:
                        reg["v4_app"] = "rechazada_segmentador"
                    else:
                        p = E.predecir(it, img224)
                        reg["v4_app"] = E.CLASSES[int(p.argmax())]
                        reg["conf_v4_app"] = round(float(p.max()), 4)
                detalle.append(reg)
                grupos[(etiqueta, "YA EN ENTRENAMIENTO" if visto else "nuevas")].append(reg)

            out.append("## Prediccion al capturar vs re-evaluacion con v4 (tflite actual)")
            for (etiqueta, tipo), rs in sorted(grupos.items()):
                n = len(rs)
                esp = rs[0]["esperado"]
                linea = f"  [{etiqueta} | {tipo}] n={n}"
                if esp:
                    ok = lambda k: sum(r.get(k) == esp for r in rs)
                    linea += (f" | aciertos al capturar {ok('pred_al_capturar')}/{n}"
                              f" | v4 entrenamiento {ok('v4_entrenamiento')}/{n}"
                              f" | v4 app {ok('v4_app')}/{n}")
                out.append(linea)
                out.append(f"      al capturar: {dict(Counter(r['pred_al_capturar'] for r in rs))}")
                out.append(f"      v4 app:      {dict(Counter(r['v4_app'] for r in rs))}")

        # ---------- Historial Room ----------
        db_dir = d / "databases"
        if (db_dir / "plagascan.db").exists():
            tmp = Path(tempfile.mkdtemp())
            for f in db_dir.glob("plagascan.db*"):
                shutil.copy(f, tmp / f.name)  # copia: nunca se toca la extraccion
            con = sqlite3.connect(tmp / "plagascan.db")
            filas = con.execute(
                "SELECT timestamp, classification, confidence, riskLevel, notes, "
                "latitude IS NOT NULL FROM analysis_results ORDER BY timestamp").fetchall()
            con.close()
            out.append(f"## Historial Room: {len(filas)} registros")
            por_fecha = defaultdict(Counter)
            for ts, cls, conf, risk, notes, _ in filas:
                por_fecha[fecha(ts)][f"{cls}/{risk}"] += 1
            for k in sorted(por_fecha):
                out.append(f"  {k}: {dict(por_fecha[k])}")
            out.append(f"  Con ubicacion: {sum(r[5] for r in filas)}/{len(filas)}")
            origen = Counter((r[4] or "").split(" [")[0] for r in filas)
            out.append(f"  Origen (notas): {dict(origen)}")
            out.append("  Ultimos 25:")
            for ts, cls, conf, risk, notes, _ in filas[-25:]:
                out.append(f"    {fecha(ts, '%m-%d %H:%M')} | {cls} {conf:.3f} {risk} | {notes}")
        out.append("")

    texto = "\n".join(out)
    (PS / "eval_app").mkdir(exist_ok=True)
    (PS / "eval_app/extraccion_resumen.txt").write_text(texto, encoding="utf-8")
    campos = ["paquete", "fecha", "archivo", "etiqueta", "esperado", "ya_en_entrenamiento",
              "pred_al_capturar", "conf_al_capturar", "v4_entrenamiento", "conf_v4_entrenamiento",
              "area_ratio", "v4_app", "conf_v4_app"]
    with open(PS / "eval_app/extraccion_detalle.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(detalle)
    print(texto)


if __name__ == "__main__":
    main()
