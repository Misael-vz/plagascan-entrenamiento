#!/usr/bin/env python3
"""
PlagaScan -- mapa_color.py  (EXPERIMENTO, rama `experimento`)

Mapa de color de la hoja: marca sobre la foto las zonas de la hoja cuyo color
se aleja del verde (manchas claras o amarillentas, y zonas cafés o secas) y
reporta qué porcentaje de la hoja ocupan.

Es INFORMATIVO, no un diagnóstico: no dice la causa (sol, sequía, nutrientes,
hongo, plaga...). Y es INDEPENDIENTE del clasificador: no modifica la imagen
que entra al modelo ni el resultado del pulgón. Solo lee la foto original.

La región de la hoja se obtiene con el MISMO rango HSV y la misma limpieza
morfológica que LeafSegmenter.kt / preprocess_opencv.py, pero aquí el contorno
se rellena para incluir las manchas que quedan fuera del rango verde.

Uso:
    python mapa_color.py --image foto.jpg
    python mapa_color.py --batch carpeta/ [--out carpeta_salida/]
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

# --- Región de la hoja: mismos valores que LeafSegmenter.kt ---
HUE_MIN, HUE_MAX, SAT_MIN, VAL_MIN = 25, 95, 30, 30
MIN_AREA_RATIO = 0.15          # igual que MIN_AREA_RATIO en Kotlin
WORK_SIZE = 640                # el mapa se calcula sobre la foto reducida (rapidez)

# --- Umbrales del mapa (EXPERIMENTALES, calibrar con fotos propias) ---
# OpenCV: H 0-179, S 0-255, V 0-255. Verde típico de hoja medido: H~36, S~150.
CLARA_H_MAX = 30     # manchas claras/amarillentas: tono más amarillo que el verde...
CLARA_V_MIN = 150    # ...y más claras que la hoja
CLARA_S_MAX = 140
CAFE_H_MAX = 22      # zonas cafés/rojizas (o H >= 160)
CAFE_S_MIN = 60
CAFE_V_MIN, CAFE_V_MAX = 40, 175
SOMBRA_V_MAX = 45    # muy oscuro: no se evalúa
BRILLO_S_MAX, BRILLO_V_MIN = 25, 235  # reflejo de luz: no se evalúa
MIN_MANCHA_PX = 4    # componentes más pequeños se consideran ruido (a 640 px)
CLARA_S_MIN = 35     # más grisáceo que esto es pared/reflejo, no mancha
MAX_HUECO_RATIO = 0.01  # hueco dentro de la hoja <= 1 % de su área: se evalúa (mancha)
ANILLO_PX = 6           # ancho del anillo alrededor de cada mancha
ANILLO_VERDE_MIN = 0.85 # al menos 85 % del anillo debe ser hoja verde
ALARGADA_LARGO_PX, ALARGADA_RATIO = 25, 4.0  # tiras largas (tutores, tallos): no son manchas

COLOR_CLARA = (255, 0, 255)   # magenta (BGR)
COLOR_CAFE = (255, 176, 0)    # azul claro (BGR)


def region_hoja(img):
    """Máscara rellena del contorno más grande de la hoja (o None)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (HUE_MIN, SAT_MIN, VAL_MIN), (HUE_MAX, 255, 255))
    k = max(5, round(15 * max(img.shape[:2]) / 1280))  # kernel 15 a la resolución del teléfono
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, 0.0
    c = max(cnts, key=cv2.contourArea)
    ratio = cv2.contourArea(c) / (img.shape[0] * img.shape[1])
    if ratio < MIN_AREA_RATIO:
        return None, ratio
    # La región es la parte verde conectada al contorno más grande, más los
    # huecos PEQUEÑOS dentro de ella (manchas). Los huecos grandes son fondo
    # que se ve entre hojas y NO se incluyen.
    lleno = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(lleno, [c], -1, 255, cv2.FILLED)
    region = cv2.bitwise_and(mask, lleno)
    huecos = cv2.bitwise_and(lleno, cv2.bitwise_not(region))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(huecos, connectivity=8)
    max_hueco = MAX_HUECO_RATIO * cv2.contourArea(c)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] <= max_hueco:
            region[lab == i] = 255
    # quitar el borde: ahí se mezcla el color del fondo
    e = max(3, round(0.012 * max(img.shape[:2])))
    region = cv2.erode(region, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (e, e)))
    return region, ratio


def limpiar(m, verde):
    """Conserva solo manchas DENTRO de la hoja: rodeadas de tejido verde y
    compactas. Así se descartan tutores, tallos, macetas y fondo que se cuela
    por el borde."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros(n, bool)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ANILLO_PX + 1, 2 * ANILLO_PX + 1))
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < MIN_MANCHA_PX:
            continue
        x, y, w, h = stats[i, :4]
        x0, y0 = max(0, x - ANILLO_PX - 1), max(0, y - ANILLO_PX - 1)
        x1, y1 = min(m.shape[1], x + w + ANILLO_PX + 1), min(m.shape[0], y + h + ANILLO_PX + 1)
        comp = (lab[y0:y1, x0:x1] == i).astype(np.uint8)
        anillo = cv2.dilate(comp, k) & (1 - comp)
        if anillo.sum() == 0 or (verde[y0:y1, x0:x1][anillo > 0] > 0).mean() < ANILLO_VERDE_MIN:
            continue
        pts = np.column_stack(np.nonzero(comp))
        if len(pts) >= 5:
            (_, _), (a, b), _ = cv2.minAreaRect(pts.astype(np.float32))
            largo, ancho = max(a, b), max(1.0, min(a, b))
            if largo > ALARGADA_LARGO_PX and largo / ancho > ALARGADA_RATIO:
                continue
        keep[i] = True
    return (keep[lab] * 255).astype(np.uint8)


def analizar(img_full):
    s = WORK_SIZE / max(img_full.shape[:2])
    img = cv2.resize(img_full, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img_full.copy()
    region, ratio = region_hoja(img)
    if region is None:
        return dict(hoja=False, area_ratio=ratio, pct_clara=0.0, pct_cafe=0.0, evaluada=0.0, img=img,
                    clara=None, cafe=None, region=None)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    en = region > 0
    excluir = (V <= SOMBRA_V_MAX) | ((S <= BRILLO_S_MAX) & (V >= BRILLO_V_MIN))
    eval_ = en & ~excluir
    cafe = eval_ & ((H <= CAFE_H_MAX) | (H >= 160)) & (S >= CAFE_S_MIN) & (V >= CAFE_V_MIN) & (V <= CAFE_V_MAX)
    clara = eval_ & ~cafe & (H <= CLARA_H_MAX) & (V >= CLARA_V_MIN) & (S <= CLARA_S_MAX) & (S >= CLARA_S_MIN)
    verde = cv2.inRange(hsv, (HUE_MIN, SAT_MIN, VAL_MIN), (HUE_MAX, 255, 255))
    cafe = limpiar(cafe.astype(np.uint8) * 255, verde)
    clara = limpiar(clara.astype(np.uint8) * 255, verde)
    n = max(1, int(eval_.sum()))
    return dict(hoja=True, area_ratio=ratio, evaluada=eval_.sum() / max(1, en.sum()),
                pct_clara=100 * (clara > 0).sum() / n, pct_cafe=100 * (cafe > 0).sum() / n,
                img=img, clara=clara, cafe=cafe, region=region)


def dibujar(r):
    img = r["img"]
    if not r["hoja"]:
        out = img.copy()
        cv2.putText(out, "sin hoja detectada", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return np.vstack([out, np.full((60, out.shape[1], 3), 30, np.uint8)])
    gris = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    out = np.where(r["region"][..., None] > 0, img, (gris * 0.45).astype(np.uint8))
    for m, col in ((r["clara"], COLOR_CLARA), (r["cafe"], COLOR_CAFE)):
        out[m > 0] = col
        # anillo alrededor de cada mancha para que se vea aunque sea pequeña
        d = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
        cnts, _ = cv2.findContours(d, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, col, 2)
    bar = np.full((60, out.shape[1], 3), 30, np.uint8)
    cv2.putText(bar, f"claras/amarillentas: {r['pct_clara']:.1f}%", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_CLARA, 2)
    cv2.putText(bar, f"cafes/secas: {r['pct_cafe']:.1f}%", (8, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_CAFE, 2)
    return np.vstack([out, bar])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image")
    ap.add_argument("--batch")
    ap.add_argument("--out", default="mapa_salida")
    a = ap.parse_args()
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    files = [Path(a.image).expanduser()] if a.image else sorted(
        p for p in Path(a.batch).expanduser().rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    filas = []
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            continue
        r = analizar(img)
        cv2.imwrite(str(out / f"mapa_{f.stem}.jpg"), np.hstack([
            np.vstack([r["img"], np.full((60, r["img"].shape[1], 3), 30, np.uint8)]), dibujar(r)]))
        filas.append([str(f), r["hoja"], f"{r['area_ratio']:.3f}", f"{r['pct_clara']:.2f}", f"{r['pct_cafe']:.2f}"])
        print(f"{f.name:32s} hoja={r['hoja']!s:5s} claras={r['pct_clara']:5.1f}%  cafes={r['pct_cafe']:5.1f}%")
    with open(out / "resumen.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["archivo", "hoja", "area_ratio", "pct_claras", "pct_cafes"])
        w.writerows(filas)
    print(f"\n{len(filas)} fotos -> {out}")


if __name__ == "__main__":
    main()
