#!/usr/bin/env python3
"""
PlagaScan -- mapa_papel.py  (EXPERIMENTO, rama `experimento`)

Mapa de color de la hoja en MODO PAPEL BLANCO: la hoja se coloca sobre una
hoja de papel blanco y se fotografía con luz de frente, sin flash.

Marca zonas de la hoja cuyo color se aleja del verde de esa misma hoja:
  - magenta: zonas pálidas o blanquecinas (pierden color respecto a la hoja)
  - amarillo: zonas amarillentas (el tono se va hacia el amarillo)
  - azul: zonas cafés o secas (tono café/rojizo, o pálidas y además oscuras)

Es INFORMATIVO, no un diagnóstico: no dice la causa. Es INDEPENDIENTE del
clasificador del pulgón: solo lee la foto original y no modifica nada de lo
que entra al modelo.

Por qué papel blanco: el fondo deja de ser verde/café, así que la hoja se
separa por saturación (el papel y su sombra son grises) y el papel sirve
además de referencia de blanco.

Uso:
    python mapa_papel.py --batch carpeta/ [--out carpeta_salida/]
    python mapa_papel.py --image foto.jpg
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

WORK_SIZE = 640
HOJA_H_MIN, HOJA_H_MAX = 22, 95   # tono HSV de OpenCV: de amarillo a verde-azulado
MIN_AREA_RATIO = 0.03      # la hoja debe ocupar al menos 3 % de la foto
BORDE_RATIO = 0.015        # se descarta un borde de ~1.5 % (mezcla con el papel)
LOCAL_KERNEL = 61          # tamaño (px a 640) de la referencia local de color/brillo

# --- Umbrales EXPERIMENTALES (Lab de OpenCV, tono h en grados) ---
# "sat" = croma / luminosidad: casi no cambia con la sombra (ambos bajan juntos),
# pero baja mucho en tejido blanquecino o seco. Se compara contra la propia hoja
# (percentil 75 = su parte más "sana"), no contra un valor fijo.
SAT_BAJA_REL = 0.60        # sat < 60 % de la referencia: zona descolorida
CAFE_H_MAX = 92            # tono claramente café/rojizo
AMARILLA_DH = 9            # tono más amarillo que la referencia de la hoja por > 9 grados
AMARILLA_SAT_REL = 0.80    # ...conservando color (no descolorida)
L_MIN = 35                 # píxeles casi negros: no se evalúan
SOMBRA_REL = 0.75          # papel bajo la hoja con luz < 75 % del papel iluminado: en sombra
APERTURA_PX = 7            # quita líneas de menos de ~7 px (nervaduras a 640 px)
MIN_MANCHA_PX = 12         # manchas más chicas se consideran ruido

COLORES = {"palida": (255, 0, 255), "amarilla": (0, 215, 255), "cafe": (255, 140, 0)}  # BGR
NOMBRES = {"palida": "descoloridas (palidas o secas)", "amarilla": "amarillentas", "cafe": "cafes/rojizas"}


def reducir(img):
    s = WORK_SIZE / max(img.shape[:2])
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img.copy()


def region_hoja(img):
    """La hoja es lo saturado; el papel y su sombra son grises. Umbral de Otsu
    sobre la saturación, contorno más grande y relleno de huecos (las manchas
    pálidas quedan dentro)."""
    S = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 1]
    H = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 0]
    t, m = cv2.threshold(S, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m[(H < HOJA_H_MIN) | (H > HOJA_H_MAX)] = 0
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, 0.0
    c = max(cnts, key=cv2.contourArea)
    ratio = cv2.contourArea(c) / (img.shape[0] * img.shape[1])
    if ratio < MIN_AREA_RATIO:
        return None, ratio
    r = np.zeros(S.shape, np.uint8)
    cv2.drawContours(r, [c], -1, 255, cv2.FILLED)
    e = max(3, round(BORDE_RATIO * max(img.shape[:2])))
    return cv2.erode(r, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (e, e))), ratio


def mediana_local(x, en):
    """Mediana de x en un vecindario grande, solo con píxeles de la hoja: así la
    referencia sigue a la sombra o al degradado de luz sobre la hoja."""
    f = x.copy()
    f[~en] = np.median(x[en])
    lo, hi = float(f.min()), float(f.max())
    u8 = np.clip((f - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    return cv2.medianBlur(u8, LOCAL_KERNEL).astype(np.float32) / 255 * (hi - lo) + lo


def limpiar(m):
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= MIN_MANCHA_PX
    return keep[lab]


def analizar(img_full):
    img = reducir(img_full)
    region, ratio = region_hoja(img)
    if region is None:
        return dict(hoja=False, area_ratio=ratio, img=img, pct={k: 0.0 for k in COLORES})
    en = region > 0
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, a, b = lab[..., 0], lab[..., 1] - 128, lab[..., 2] - 128
    # balance de blancos con el papel bien iluminado
    fuera = ~cv2.dilate(region, np.ones((25, 25), np.uint8)).astype(bool)
    pap = fuera & (L > np.percentile(L[fuera], 70))
    a, b = a - np.median(a[pap]), b - np.median(b[pap])
    h = np.degrees(np.arctan2(b, a))
    C = np.hypot(a, b)
    hoja = en.copy()
    en &= L > L_MIN
    # zonas de la hoja bajo sombra (p. ej. la del teléfono). La sombra se mide en
    # el PAPEL alrededor de la hoja y se extiende por debajo de ella (inpainting):
    # así una hoja verde oscura no se confunde con una hoja en sombra.
    # (se calcula a 1/4 de resolución: la sombra es suave y así es ~10 veces más rápido)
    tapa = cv2.dilate(region, np.ones((15, 15), np.uint8))
    hh, ww = L.shape
    chico = lambda x: cv2.resize(x, (ww // 4, hh // 4), interpolation=cv2.INTER_AREA)
    luz = cv2.inpaint(chico(np.clip(L, 0, 255).astype(np.uint8)), (chico(tapa) > 0).astype(np.uint8), 5, cv2.INPAINT_TELEA)
    luz = cv2.resize(cv2.GaussianBlur(luz.astype(np.float32), (0, 0), 2.5), (ww, hh), interpolation=cv2.INTER_LINEAR)
    en &= luz >= SOMBRA_REL * np.percentile(L[fuera], 90)
    evaluada = en.sum() / max(1, hoja.sum())
    sat = C / np.maximum(L, 20)
    ref_sat = np.percentile(sat[en], 75)
    ref_h = np.percentile(h[en], 75)
    baja = en & (sat < SAT_BAJA_REL * ref_sat)
    cafe = en & (h < CAFE_H_MAX) & (C > 8)
    palida = baja & ~cafe
    amarilla = en & ~baja & ~cafe & (h < ref_h - AMARILLA_DH) & (sat >= AMARILLA_SAT_REL * ref_sat)
    # las nervaduras claras son líneas finas: una apertura las borra y deja las manchas
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (APERTURA_PX, APERTURA_PX))
    abrir = lambda m: cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, k) > 0
    masks = {"palida": limpiar(abrir(palida)), "amarilla": limpiar(abrir(amarilla)), "cafe": limpiar(abrir(cafe))}
    n = max(1, int(en.sum()))
    return dict(hoja=True, area_ratio=ratio, img=img, region=region, masks=masks, evaluada=evaluada,
                sombra=hoja & ~en, pct={k: 100 * m.sum() / n for k, m in masks.items()})


def dibujar(r):
    img = r["img"]
    bar = np.full((80, img.shape[1], 3), 30, np.uint8)
    if not r["hoja"]:
        cv2.putText(bar, "sin hoja detectada", (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return np.vstack([img, bar])
    gris = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    out = np.where(r["region"][..., None] > 0, img, (gris * 0.5).astype(np.uint8))
    sh = r["sombra"]
    out[sh] = (0.5 * out[sh] + 0.5 * np.array((90, 90, 90))).astype(np.uint8)
    for k, m in r["masks"].items():
        u8 = m.astype(np.uint8) * 255
        out[m] = (0.35 * out[m] + 0.65 * np.array(COLORES[k])).astype(np.uint8)
        cnts, _ = cv2.findContours(cv2.dilate(u8, np.ones((7, 7), np.uint8)), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, COLORES[k], 2)
    for i, k in enumerate(COLORES):
        cv2.putText(bar, f"{NOMBRES[k]}: {r['pct'][k]:.1f}%", (8, 22 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORES[k], 2)
    if r["evaluada"] < 0.95:
        cv2.putText(bar, f"evaluada {100 * r['evaluada']:.0f}% (resto en sombra)", (out.shape[1] - 250, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    return np.vstack([out, bar])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image")
    ap.add_argument("--batch")
    ap.add_argument("--out", default="mapa_papel_salida")
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
        orig = np.vstack([r["img"], np.full((80, r["img"].shape[1], 3), 30, np.uint8)])
        cv2.imwrite(str(out / f"mapa_{f.stem}.jpg"), np.hstack([orig, dibujar(r)]))
        p = r["pct"]
        filas.append([str(f), r["hoja"], f"{r['area_ratio']:.3f}"] + [f"{p[k]:.2f}" for k in COLORES])
        ev = f"evaluada={100 * r['evaluada']:3.0f}%" if r["hoja"] else ""
        print(f"{f.name:28s} hoja={r['hoja']!s:5s} " + "  ".join(f"{k}={p[k]:5.1f}%" for k in COLORES) + "  " + ev)
    with open(out / "resumen.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["archivo", "hoja", "area_ratio"] + [f"pct_{k}" for k in COLORES])
        w.writerows(filas)
    print(f"\n{len(filas)} fotos -> {out}")


if __name__ == "__main__":
    main()
