#!/usr/bin/env python3
"""
eval_tflite_app.py -- Evaluacion de SOLO LECTURA del modelo TFLite real de la app.

No modifica modelos, dataset ni codigo. Compara dos preprocesamientos sobre el
mismo .tflite que va dentro del APK:
  - "entrenamiento": letterbox de la foto completa (igual que build_dataset.py)
  - "app": reduccion a 1280 px + segmentacion HSV + recorte (>=15% de area)
           + letterbox (replica de CameraScreen.kt + LeafSegmenter.kt +
           PlagaScanClassifier.kt). Si no hay hoja -> "rechazada_segmentador".

Salida: ~/plagascan/eval_app/<fecha_hora>/{detalle.csv, resumen.txt}
        y una copia en ~/plagascan/eval_app/ultimo_resumen.txt
"""
import csv
import hashlib
import math
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

HOME = Path.home()
PS = HOME / "plagascan"
APP_MODEL = HOME / "AndroidStudioProjects/proyecto/app/src/main/assets/plagascan_mobilenetv2.tflite"
APP_ASSETS = APP_MODEL.parent
CLASSES = ["no_frijol", "plaga", "sano"]

# Constantes copiadas de LeafSegmenter.kt
HUE_MIN, HUE_MAX, SAT_MIN, VAL_MIN = 25, 95, 30, 30
MIN_AREA_RATIO = 0.15
MARGIN_RATIO = 0.08
TARGET = 224
APP_MAX_SIDE = 1280  # ResolutionSelector(1280x960) de CameraScreen.kt


def kt_round(x):
    """Math.round de Kotlin (redondeo hacia arriba en .5)."""
    return int(math.floor(x + 0.5))


def letterbox(img, rnd):
    h, w = img.shape[:2]
    s = TARGET / max(h, w)
    nh, nw = rnd(h * s), rnd(w * s)
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    top = (TARGET - nh) // 2
    left = (TARGET - nw) // 2
    return cv2.copyMakeBorder(r, top, TARGET - nh - top, left, TARGET - nw - left,
                              cv2.BORDER_REFLECT)


def pipeline_entrenamiento(img_bgr, ya_procesada):
    if ya_procesada:  # archivo de dataset/val: ya es 224x224 con letterbox
        return img_bgr
    return letterbox(img_bgr, lambda v: int(round(v)))  # igual que build_dataset.py


def pipeline_app(img_bgr):
    """Devuelve (imagen_224 o None si se rechaza, area_ratio)."""
    h, w = img_bgr.shape[:2]
    if max(h, w) > APP_MAX_SIDE:
        s = APP_MAX_SIDE / max(h, w)
        img_bgr = cv2.resize(img_bgr, (kt_round(w * s), kt_round(h * s)),
                             interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([HUE_MIN, SAT_MIN, VAL_MIN]),
                       np.array([HUE_MAX, 255, 255]))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total = img_bgr.shape[0] * img_bgr.shape[1]
    if not contours:
        return None, 0.0
    largest = max(contours, key=cv2.contourArea)
    ratio = cv2.contourArea(largest) / total
    if ratio < MIN_AREA_RATIO:
        return None, ratio
    x, y, bw, bh = cv2.boundingRect(largest)
    mx, my = int(bw * MARGIN_RATIO), int(bh * MARGIN_RATIO)
    x0, y0 = max(0, x - mx), max(0, y - my)
    x1 = min(img_bgr.shape[1], x + bw + mx)
    y1 = min(img_bgr.shape[0], y + bh + my)
    return letterbox(img_bgr[y0:y1, x0:x1], kt_round), ratio


def cargar_interprete(path):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
    it = Interpreter(model_path=str(path))
    it.allocate_tensors()
    return it


def predecir(it, img224_bgr):
    inp = it.get_input_details()[0]
    out = it.get_output_details()[0]
    x = cv2.cvtColor(img224_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)[None]
    it.set_tensor(inp["index"], x)
    it.invoke()
    return it.get_tensor(out["index"])[0].astype(float)


def resolver(p):
    p = Path(p)
    return p if p.is_absolute() else PS / p


def reunir_items():
    items = []
    # 1) Validacion: archivo procesado (pipeline entrenamiento) + original (pipeline app)
    with open(PS / "dataset/metadata.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] != "val":
                continue
            items.append(dict(grupo=f"val/{row['clase']}", esperado=row["clase"],
                              archivo=row["archivo"], nota=row["fuente"],
                              ruta_entrenamiento=PS / "dataset" / row["archivo"],
                              ya_procesada=True,
                              ruta_app=resolver(row["origen_local"])))
    # 2) Evidencia de campo (etiquetas del log de recoleccion cuando existen)
    notas_log = {}
    log = PS / "recoleccion_campo/20260831/ground_truth_log.csv"
    if log.exists():
        with open(log, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                notas_log[Path(row["archivo_imagen"]).name] = (
                    f"log:{row['salud_confirmada']},seg={row['segmentador_detecto']}")
    for carpeta in ["evidencia_campo_frijol/entrenar", "evidencia_campo_frijol/prueba_honesta",
                    "evidencia_campo_v2/entrenar", "evidencia_campo_v2/prueba_honesta"]:
        for p in sorted((PS / carpeta).glob("*.jpg")):
            items.append(dict(grupo=f"campo/{carpeta}", esperado="sano", archivo=p.name,
                              nota=notas_log.get(p.name, "sin_log(sano asumido)"),
                              ruta_entrenamiento=p, ya_procesada=False, ruta_app=p))
    # 3) Recoleccion "otros" (confirmadas por el usuario como NO frijol)
    for p in sorted((PS / "recoleccion_campo/20260831/otros").glob("*.jpg")):
        items.append(dict(grupo="campo/otros_no_frijol", esperado="no_frijol", archivo=p.name,
                          nota="log:otros", ruta_entrenamiento=p, ya_procesada=False, ruta_app=p))
    # 4) Imagenes de prueba dentro de assets de la app
    for nombre, clase in [("test_sano.jpg", "sano"), ("test_plaga.jpg", "plaga")]:
        p = APP_ASSETS / nombre
        if p.exists():
            items.append(dict(grupo="assets_app", esperado=clase, archivo=nombre, nota="assets",
                              ruta_entrenamiento=p, ya_procesada=False, ruta_app=p))
    return items


def main():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PS / "eval_app" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    sha = hashlib.sha256(APP_MODEL.read_bytes()).hexdigest()
    it = cargar_interprete(APP_MODEL)
    inp = it.get_input_details()[0]
    out = it.get_output_details()[0]

    filas = []
    for item in reunir_items():
        fila = {k: item[k] for k in ("grupo", "archivo", "esperado", "nota")}
        img_tr = cv2.imread(str(item["ruta_entrenamiento"]))
        if img_tr is None:
            fila["pred_entrenamiento"] = "ERROR_LECTURA"
        else:
            p = predecir(it, pipeline_entrenamiento(img_tr, item["ya_procesada"]))
            fila["pred_entrenamiento"] = CLASSES[int(np.argmax(p))]
            fila["conf_entrenamiento"] = round(float(p.max()), 4)
            fila["probs_entrenamiento"] = " ".join(f"{v:.4f}" for v in p)
        img_app = cv2.imread(str(item["ruta_app"]))
        if img_app is None:
            fila["resultado_app"] = "ERROR_LECTURA_ORIGINAL"
        else:
            img224, ratio = pipeline_app(img_app)
            fila["area_ratio"] = round(ratio, 4)
            if img224 is None:
                fila["resultado_app"] = "rechazada_segmentador"
            else:
                p = predecir(it, img224)
                fila["resultado_app"] = CLASSES[int(np.argmax(p))]
                fila["conf_app"] = round(float(p.max()), 4)
                fila["probs_app"] = " ".join(f"{v:.4f}" for v in p)
        filas.append(fila)

    campos = ["grupo", "archivo", "esperado", "nota", "pred_entrenamiento", "conf_entrenamiento",
              "probs_entrenamiento", "area_ratio", "resultado_app", "conf_app", "probs_app"]
    with open(out_dir / "detalle.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(filas)

    lineas = [
        "# Evaluacion de solo lectura del TFLite de la app",
        f"Fecha: {stamp}",
        f"Modelo: {APP_MODEL}",
        f"SHA-256: {sha}",
        f"Entrada: {inp['shape'].tolist()} {inp['dtype'].__name__} | Salida: {out['shape'].tolist()} {out['dtype'].__name__}",
        "Probabilidades en orden: no_frijol plaga sano",
        "",
    ]
    por_grupo = defaultdict(list)
    for fl in filas:
        por_grupo[fl["grupo"]].append(fl)
    for g, rs in por_grupo.items():
        n = len(rs)
        ok_tr = sum(r.get("pred_entrenamiento") == r["esperado"] for r in rs)
        ok_ap = sum(r.get("resultado_app") == r["esperado"] for r in rs)
        confs = [r["conf_app"] for r in rs if "conf_app" in r]
        media = f"{sum(confs) / len(confs):.3f}" if confs else "-"
        lineas += [
            f"## {g} (n={n}, esperado={rs[0]['esperado']})",
            f"  Pipeline entrenamiento: {ok_tr}/{n} aciertos | {dict(Counter(r.get('pred_entrenamiento') for r in rs))}",
            f"  Pipeline app:           {ok_ap}/{n} aciertos | {dict(Counter(r.get('resultado_app') for r in rs))}",
            f"  Confianza media (app, no rechazadas): {media}",
            "",
        ]
    texto = "\n".join(lineas)
    (out_dir / "resumen.txt").write_text(texto, encoding="utf-8")
    shutil.copy(out_dir / "resumen.txt", PS / "eval_app" / "ultimo_resumen.txt")
    print(texto)
    print(f"Detalle por imagen: {out_dir / 'detalle.csv'}")


if __name__ == "__main__":
    main()
