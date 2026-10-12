#!/usr/bin/env python3
"""
comparar_galeria.py -- Compara lo que dijo el TELEFONO (registro de logcat de
la prueba de galeria) contra lo que dio la COMPUTADORA para las mismas fotos
(~/plagascan/prueba_galeria/esperado_pc.csv). Solo lectura.

No importa en que orden ni de cuantas en cuantas se analizaron las fotos:
cada foto se reconoce por el tamano relativo de la hoja (area_ratio) que
reporta el segmentador, que es practicamente unico por foto, y por sus
probabilidades. Las fotos que no son de la prueba (por ejemplo, de Google)
quedan como "no identificadas" y no se cuentan.

Uso (no necesita el venv):
    python3 ~/Descargas/comparar_galeria.py
    python3 ~/Descargas/comparar_galeria.py archivo1.txt archivo2.txt
Por defecto lee ~/Descargas/galeria_plaga.txt y ~/Descargas/galeria_sano.txt.
"""
import csv
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

PS = Path.home() / "plagascan"
ESPERADO = PS / "prueba_galeria/esperado_pc.csv"
UMBRAL, TOL_AREA = 0.80, 0.004
CATS = ["plaga ROJO", "plaga AMARILLO", "sano VERDE", "sano AMARILLO", "no_frijol", "sin_hoja"]
CORRECTO = {"plaga": ("plaga ROJO", "plaga AMARILLO"), "sano": ("sano VERDE", "sano AMARILLO")}
RE_SEG = re.compile(r"Segmentaci.n: found=(true|false), area_ratio=([0-9.eE+-]+)")
RE_PRED = re.compile(r"Prediccion: (\w+) \(.*?\[no_frijol=([0-9.eE+-]+), plaga=([0-9.eE+-]+), sano=([0-9.eE+-]+)\]")


def categoria(clase, conf):
    if clase in ("sin_hoja", "no_frijol"):
        return clase
    return f"{clase} {('ROJO' if clase == 'plaga' else 'VERDE') if conf >= UMBRAL else 'AMARILLO'}"


def leer_registro(rutas):
    """Lista de dict(area, clase, conf, probs) en el orden en que el telefono analizo cada foto."""
    regs, pendiente = [], None
    for ruta in rutas:
        for linea in ruta.read_text(encoding="utf-8", errors="replace").splitlines():
            m = RE_SEG.search(linea)
            if m:
                area = float(m.group(2))
                if m.group(1) == "false":
                    regs.append(dict(area=area, clase="sin_hoja", conf=0.0, probs=None))
                    pendiente = None
                else:
                    pendiente = area
                continue
            m = RE_PRED.search(linea)
            if m:
                probs = [float(m.group(i)) for i in (2, 3, 4)]
                regs.append(dict(area=pendiente, clase=m.group(1), conf=max(probs), probs=probs))
                pendiente = None
    return regs


def main():
    rutas = [Path(a).expanduser() for a in sys.argv[1:]] or [
        p for p in (Path.home() / "Descargas/galeria_plaga.txt", Path.home() / "Descargas/galeria_sano.txt") if p.exists()]
    faltan = [str(r) for r in rutas if not r.exists()]
    if not rutas or faltan:
        sys.exit("No encuentro el registro: " + (", ".join(faltan) or "~/Descargas/galeria_plaga.txt"))
    if not ESPERADO.exists():
        sys.exit(f"No existe {ESPERADO} (corre antes preparar_prueba_galeria.py)")
    with open(ESPERADO, newline="", encoding="utf-8") as f:
        esperado = list(csv.DictReader(f))
    for e in esperado:
        e["area"] = float(e["area_ratio"])
        e["probs"] = [float(v) for v in e["probs_no_frijol_plaga_sano"].split()] or None
        e["cat_pc"] = categoria(e["resultado_pc"], float(e["confianza_pc"]))

    regs = leer_registro(rutas)
    asignado, repetidas, desconocidas = {}, 0, 0
    for r in regs:
        if r["area"] is None:
            desconocidas += 1
            continue
        cands = []
        for i, e in enumerate(esperado):
            d = abs(e["area"] - r["area"])
            if d <= TOL_AREA:
                dp = sum(abs(a - b) for a, b in zip(e["probs"], r["probs"])) if e["probs"] and r["probs"] else (
                    0.0 if (e["probs"] is None) == (r["probs"] is None) else 2.0)
                cands.append((d / TOL_AREA + dp, i))
        if not cands:
            desconocidas += 1
            continue
        cands.sort()
        libres = [c for c in cands if c[1] not in asignado]
        if libres and libres[0][0] <= cands[0][0] + 0.25:
            asignado[libres[0][1]] = r
        else:
            repetidas += 1

    L = [f"# Prueba de galería en el teléfono contra la computadora -- {datetime.now():%Y-%m-%d %H:%M}",
         "Registro: " + ", ".join(r.name for r in rutas),
         f"Fotos analizadas en el registro: {len(regs)}  (de la prueba: {len(asignado)} distintas, "
         f"{repetidas} repetidas; {desconocidas} que no son de la prueba)", ""]
    grupos = defaultdict(list)
    for i, e in enumerate(esperado):
        grupos[(e["clase_real"], e["fuente"])].append(i)
    L.append(f"{'clase / fuente':34s} {'total':>5s} {'hechas':>6s} | " + " ".join(f"{c:>14s}" for c in CATS) + " | acierta  igual que PC")
    totales = defaultdict(lambda: [0, 0, 0, 0])
    difs = []
    for (clase, fuente), idx in sorted(grupos.items()):
        hechos = [i for i in idx if i in asignado]
        cats = Counter(categoria(asignado[i]["clase"], asignado[i]["conf"]) for i in hechos)
        ok = sum(cats[c] for c in CORRECTO[clase])
        igual = sum(categoria(asignado[i]["clase"], asignado[i]["conf"]) == esperado[i]["cat_pc"] for i in hechos)
        L.append(f"{clase + ' / ' + fuente:34s} {len(idx):5d} {len(hechos):6d} | " + " ".join(f"{cats[c]:14d}" for c in CATS)
                 + f" | {ok:3d}/{len(hechos):<3d}  {igual:3d}/{len(hechos):<3d}")
        t = totales[clase]
        t[0] += len(idx); t[1] += len(hechos); t[2] += ok; t[3] += igual
        for i in hechos:
            r, e = asignado[i], esperado[i]
            cat = categoria(r["clase"], r["conf"])
            if cat != e["cat_pc"]:
                difs.append(f"  {e['archivo']} ({fuente}): computadora = {e['cat_pc']} {float(e['confianza_pc']):.2f}  ->  teléfono = {cat} {r['conf']:.2f}")
    L.append("")
    for clase, (tot, hechas, ok, igual) in sorted(totales.items()):
        pc_ok = sum(e["cat_pc"] in CORRECTO[clase] for e in esperado if e["clase_real"] == clase)
        L.append(f"TOTAL {clase}: analizadas {hechas} de {tot} | teléfono acierta {ok}/{hechas} | "
                 f"mismo resultado que la computadora en {igual}/{hechas} | (computadora, las {tot}: {pc_ok}/{tot})")
    L += ["", f"## Fotos donde el teléfono y la computadora NO coinciden ({len(difs)})"] + (difs or ["  ninguna"])
    pend = [e["archivo"] for i, e in enumerate(esperado) if i not in asignado]
    L += ["", f"## Faltan por analizar en el teléfono ({len(pend)})"]
    por_carpeta = defaultdict(list)
    for a in pend:
        por_carpeta[a.split("/")[0]].append(a.split("/")[1].rsplit(".", 1)[0].split("_")[1])
    L += [f"  {c}: " + " ".join(v) for c, v in sorted(por_carpeta.items())] or ["  ninguna: prueba completa"]
    texto = "\n".join(L)
    salida = PS / "eval_app" / f"galeria_telefono_{datetime.now():%Y%m%d_%H%M%S}.txt"
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text(texto, encoding="utf-8")
    print(texto)
    print(f"\nGuardado en: {salida}")


if __name__ == "__main__":
    main()
