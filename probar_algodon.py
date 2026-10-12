#!/usr/bin/env python3
"""
probar_algodon.py -- Prueba de SOLO LECTURA con "Cotton Aphid Images"
(Sami, 2022, Mendeley Data, DOI 10.17632/5wp23rgtpy.1, CC BY 4.0).

Son fotos de pulgon del algodon (Aphis gossypii, la misma especie del
proyecto) sobre hoja de algodon. El modelo NUNCA las ha visto, asi que sirven
como prueba independiente: ¿que dice hoy la app ante estas fotos?

  - Lo esperado es "plaga" (igual que el limon con pulgon).
  - Si la mayoria sale "no_frijol", el modelo no reconoce el pulgon cuando
    cambia la planta.

Lee las fotos directamente del zip (no extrae nada) y usa el mismo flujo y el
mismo modelo que la app. No toca la app, el modelo ni los datasets.

Deja dos archivos en ~/plagascan/eval_app/:
  algodon_<fecha>.txt          la tabla de resultados
  algodon_muestra_<fecha>.jpg  20 fotos de muestra: a la izquierda la foto,
                               a la derecha lo que ve el modelo (224x224)

Uso:
    cd ~/plagascan
    source venv/bin/activate
    python ~/Descargas/probar_algodon.py 2>/dev/null
"""
import hashlib
import os
import sys
import traceback
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path, PurePosixPath

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2
import numpy as np

PS = Path.home() / "plagascan"
sys.path.insert(0, str(PS))
import eval_tflite_app as ev  # mismo flujo y mismo modelo que la app

CARPETA = PS / "fuentes_nuevas/cotton_aphid"
UMBRAL = 0.80
CATS = ["plaga ROJO", "plaga AMARILLO", "sano VERDE", "sano AMARILLO", "no_frijol", "sin_hoja"]
EXT = (".jpg", ".jpeg", ".png")


def buscar_zip():
    candidatos = [p for p in CARPETA.rglob("*.zip") if p.name != "cotton_aphid.zip"]
    if not candidatos:
        print(f"No encontré el zip interior dentro de {CARPETA}.")
        print("Corre primero, dentro de esa carpeta:  unzip -o -q cotton_aphid.zip")
        sys.exit(1)
    return max(candidatos, key=lambda p: p.stat().st_size)


def huella(img):
    """Huella de 64 bits para reconocer la misma foto aunque este recomprimida o reducida."""
    g = cv2.cvtColor(cv2.resize(img, (9, 8), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    return (g[:, 1:] > g[:, :-1]).ravel()


def clasificar(it, img):
    img224, area = ev.pipeline_app(img)
    if img224 is None:
        return "sin_hoja", 0.0, None, area
    p = ev.predecir(it, img224)
    clase, conf = ev.CLASSES[int(np.argmax(p))], float(p.max())
    if clase == "no_frijol":
        return clase, conf, img224, area
    nivel = ("ROJO" if clase == "plaga" else "VERDE") if conf >= UMBRAL else "AMARILLO"
    return f"{clase} {nivel}", conf, img224, area


def celda(img, img224, texto):
    """Foto (280x210) + lo que ve el modelo (210x210) + renglon de texto."""
    lienzo = np.full((236, 500, 3), 255, np.uint8)
    h, w = img.shape[:2]
    s = min(280 / w, 210 / h)
    mini = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    y0, x0 = (210 - mini.shape[0]) // 2, (280 - mini.shape[1]) // 2
    lienzo[y0:y0 + mini.shape[0], x0:x0 + mini.shape[1]] = mini
    if img224 is not None:
        lienzo[0:210, 285:495] = cv2.resize(img224, (210, 210), interpolation=cv2.INTER_AREA)
    else:
        lienzo[0:210, 285:495] = 225
    cv2.putText(lienzo, texto, (4, 229), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return lienzo


def main():
    ruta_zip = buscar_zip()
    z = zipfile.ZipFile(ruta_zip)
    fotos = sorted((i for i in z.infolist()
                    if not i.is_dir() and PurePosixPath(i.filename).suffix.lower() in EXT),
                   key=lambda i: (len(PurePosixPath(i.filename).parts), i.filename))
    if not fotos:
        print(f"El zip {ruta_zip} no trae fotos.")
        sys.exit(1)

    it = ev.cargar_interprete(ev.APP_MODEL)
    por_carpeta, unicas = {}, Counter()
    vistas_md5, vistas_huella = set(), []
    tamanos, areas, filas, muestras = Counter(), [], [], []
    ilegibles = 0
    for k, info in enumerate(fotos, 1):
        datos = z.read(info)
        img = cv2.imdecode(np.frombuffer(datos, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            ilegibles += 1
            continue
        carpeta = str(PurePosixPath(info.filename).parent)
        cat, conf, img224, area = clasificar(it, img)
        por_carpeta.setdefault(carpeta, Counter())[cat] += 1
        md5 = hashlib.md5(datos).hexdigest()
        hu = huella(img)
        # misma foto = mismo archivo, o huella casi igual (hasta 2 de 64 bits distintos)
        repetida = md5 in vistas_md5 or any(int(np.count_nonzero(hu != h)) <= 2 for h in vistas_huella)
        vistas_md5.add(md5)
        vistas_huella.append(hu)
        if not repetida:
            unicas[cat] += 1
            tamanos[f"{img.shape[1]}x{img.shape[0]}"] += 1
            areas.append(area)
            # se guarda solo la miniatura ya armada (la foto completa ocuparia mucha memoria)
            muestras.append(celda(img, img224, f"{cat} {conf:.2f}"))
        filas.append(f"{info.filename} | {cat} {conf:.2f} | area {area:.2f} | {'repetida' if repetida else 'unica'}")
        if k % 50 == 0:
            print(f"  {k}/{len(fotos)}...")

    marca = datetime.now().strftime("%Y%m%d_%H%M%S")
    L = [f"# Cotton Aphid Images con el flujo y el modelo de la app -- {datetime.now():%Y-%m-%d %H:%M}",
         f"Zip: {ruta_zip}",
         f"Modelo de la app: {ev.APP_MODEL} ({ev.APP_MODEL.stat().st_size // 1024} KB)",
         f"Fotos en el zip: {len(fotos)} (ilegibles: {ilegibles}). Distintas (sin repetidas, aprox.): {sum(unicas.values())}",
         f"Tamaños más comunes (fotos distintas): {dict(tamanos.most_common(4))}",
         f"Área de hoja detectada (mediana, fotos distintas): {float(np.median(areas)) if areas else 0:.2f}",
         "",
         f"{'carpeta':48s} {'n':>4s} | " + " ".join(f"{c:>14s}" for c in CATS)]
    for carpeta, c in por_carpeta.items():
        L.append(f"{carpeta[-48:]:48s} {sum(c.values()):4d} | " + " ".join(f"{c[x]:14d}" for x in CATS))
    L.append(f"{'TOTAL SIN REPETIDAS':48s} {sum(unicas.values()):4d} | " + " ".join(f"{unicas[x]:14d}" for x in CATS))
    n = sum(unicas.values())
    plaga = unicas["plaga ROJO"] + unicas["plaga AMARILLO"]
    L += ["", f"Resumen (fotos distintas): plaga {plaga}/{n} ({unicas['plaga ROJO']} en rojo); "
              f"no_frijol {unicas['no_frijol']}; sano {unicas['sano VERDE'] + unicas['sano AMARILLO']}; "
              f"sin hoja {unicas['sin_hoja']}",
          "", "## Foto por foto"] + filas
    texto = "\n".join(L)
    salida = PS / "eval_app" / f"algodon_{marca}.txt"
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text(texto, encoding="utf-8")

    # Hoja de muestra: hasta 20 fotos distintas, 4 por renglon
    indices = sorted({int(round(i)) for i in np.linspace(0, len(muestras) - 1, min(20, len(muestras)))})
    celdas = [muestras[i] for i in indices]
    while len(celdas) % 4:
        celdas.append(np.full((236, 500, 3), 255, np.uint8))
    hoja = np.vstack([np.hstack(celdas[i:i + 4]) for i in range(0, len(celdas), 4)])
    ruta_hoja = PS / "eval_app" / f"algodon_muestra_{marca}.jpg"
    cv2.imwrite(str(ruta_hoja), hoja, [cv2.IMWRITE_JPEG_QUALITY, 85])

    print("\n".join(L[:L.index("## Foto por foto")]))
    print(f"Tabla completa: {salida}")
    print(f"Fotos de muestra: {ruta_hoja}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        print("ERROR (pégame este texto):")
        print(traceback.format_exc())
        sys.exit(1)
