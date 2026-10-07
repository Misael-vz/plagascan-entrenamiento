#!/usr/bin/env python3
"""
PlagaScan — build_dataset.py

Descarga, organiza y documenta el dataset de entrenamiento para el
clasificador sano/plaga (pulgón amarillo, Aphis gossypii) en hoja de frijol.

CORRER EN TU MÁQUINA LOCAL (Fedora, venv con internet), NO en un entorno
sandbox sin red. Ver DATASET_DECISIONS.md para el detalle de fuentes,
licencias y limitaciones.

Uso:
    python build_dataset.py --root ~/plagascan/dataset

Requiere:
    pip install requests pyinaturalist opencv-python pandas matplotlib tqdm
    pip install datasets tabulate
"""

import argparse
import io
import json
import random
import zipfile
from pathlib import Path

import cv2
import numpy as np
import requests
from tqdm import tqdm

random.seed(42)

TARGET_SIZE = 224
MIN_PER_CLASS = 150
TRAIN_RATIO = 0.8

INAT_TAXON_NAME = "Aphis gossypii"
INAT_TARGET = 300


# ---------------------------------------------------------------------------
# Utilidades comunes
# ---------------------------------------------------------------------------

def letterbox_resize(img, size=TARGET_SIZE):
    """Redimensiona preservando proporción y rellena con padding reflejado
    (no negro sólido) para evitar que el modelo use la cantidad/patrón de
    padding como atajo de clasificación en vez del contenido real de la
    imagen — bug detectado vía verificación Grad-CAM."""
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    top = (size - nh) // 2
    bottom = size - nh - top
    left = (size - nw) // 2
    right = size - nw - left
    canvas = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                  borderType=cv2.BORDER_REFLECT)
    return canvas


def save_processed(src_path: Path, dst_path: Path):
    img = cv2.imread(str(src_path))
    if img is None:
        return False
    out = letterbox_resize(img)
    cv2.imwrite(str(dst_path), out, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return True


# ---------------------------------------------------------------------------
# Fuente 1: Makerere iBean (clase sano) — via Hugging Face
# (el bucket original de Google Cloud Storage ya no es público, da 403)
# Licencia real del dataset: MIT (corregido, antes decía CC BY 4.0 por error)
# ---------------------------------------------------------------------------

def fetch_makerere(raw_dir: Path):
    """Descarga el dataset iBean via Hugging Face. Filtra solo la clase
    'healthy' y la guarda como JPG en raw_dir/makerere_ibean/healthy/."""
    from datasets import load_dataset

    dest = raw_dir / "makerere_ibean" / "healthy"
    dest.mkdir(parents=True, exist_ok=True)

    existing = sorted(dest.glob("*.jpg"))
    if existing:
        print(f"[Makerere] {len(existing)} imágenes ya extraídas previamente, reutilizando.")
        return existing

    print("[Makerere] Descargando via Hugging Face (AI-Lab-Makerere/beans)...")
    ds = load_dataset("AI-Lab-Makerere/beans", split="train")
    healthy_label = ds.features["labels"].names.index("healthy")

    files = []
    for i, example in enumerate(ds):
        if example["labels"] != healthy_label:
            continue
        img = example["image"].convert("RGB")
        out_path = dest / f"healthy_{i:04d}.jpg"
        img.save(out_path, "JPEG", quality=95)
        files.append(out_path)

    print(f"[Makerere] {len(files)} imágenes 'healthy' extraídas y guardadas.")
    return files


# ---------------------------------------------------------------------------
# Fuente 2: LeLePhid (clase plaga, refuerzo de patrón)
# Requiere ZIP descargado manualmente de Mendeley (bloquea descarga por API)
# Nombres de archivo confirmados: "aphid.NNN.jpg" y "healthy.NNN.jpg"
# ---------------------------------------------------------------------------

def fetch_lelephid(raw_dir: Path):
    """Extrae LeLePhid desde el ZIP descargado manualmente de Mendeley."""
    dest = raw_dir / "lelephid"
    manual_zip = raw_dir / "lelephid_manual.zip"
    dest.mkdir(parents=True, exist_ok=True)

    already_extracted = any(dest.rglob("*"))
    if not already_extracted:
        if manual_zip.exists():
            print("[LeLePhid] Extrayendo ZIP descargado manualmente...")
            with zipfile.ZipFile(manual_zip) as zf:
                zf.extractall(dest)
        else:
            print("[LeLePhid] No se encontró el ZIP manual.")
            print("  -> Descarga manual requerida: "
                  "https://data.mendeley.com/datasets/tndhs2zng4/1")
            print(f"  -> Guardar como: {manual_zip}")
            return []

    all_files = [p for p in dest.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    print(f"[LeLePhid] {len(all_files)} archivos de imagen totales extraídos.")

    # Filtro por nombre de ARCHIVO (no ruta completa, la carpeta padre
    # contiene la palabra "Aphids" en el título del dataset y confundía el filtro)
    aphid_files = [p for p in all_files if p.name.lower().startswith("aphid")]
    print(f"[LeLePhid] {len(aphid_files)} imágenes de la clase 'aphid' identificadas "
          f"(de {len(all_files)} totales).")
    return aphid_files


# ---------------------------------------------------------------------------
# Fuente 3: iNaturalist (clase plaga, apariencia real del insecto)
# ---------------------------------------------------------------------------

# Curación manual: índices confirmados visualmente como morfo amarillo
# tras revisión de contact sheets (ver DATASET_DECISIONS.md, sección
# "Curación manual iNaturalist" para el detalle del proceso).
INAT_YELLOW_INDICES = [5, 12, 13, 17, 24, 26, 27, 39, 47, 49, 50, 51, 52, 53,
                        65, 66, 89, 90, 97, 106, 117, 129, 145, 146]


def filter_yellow_morph(files):
    """Aplica la curación manual (ver DATASET_DECISIONS.md). Los índices
    fueron confirmados visualmente sobre los primeros 150 archivos
    descargados de iNaturalist, ordenados alfabéticamente. Para ser robusto
    a que la carpeta haya crecido después (target subido), se aíslan los
    150 archivos más antiguos por fecha de modificación (los que no se
    re-descargan porque ya existían) antes de ordenar y aplicar índices."""
    by_mtime = sorted(files, key=lambda p: p.stat().st_mtime)
    original_150 = by_mtime[:150]
    original_150_sorted = sorted(original_150)  # mismo orden que los contact sheets

    kept = [original_150_sorted[i] for i in INAT_YELLOW_INDICES
            if i < len(original_150_sorted)]
    print(f"[Curación manual] {len(kept)} imágenes confirmadas como morfo "
          f"amarillo (de {len(original_150_sorted)} originales aisladas de "
          f"{len(files)} totales en disco).")
    return kept

# Ampliación: 7 imágenes adicionales confirmadas tras búsqueda extendida
# (Aphis gossypii + Aphis craccivora, quality_grade research+needs_id,
# pre-filtradas por color HSV min_ratio=0.25, luego curadas a mano).
# Ver DATASET_DECISIONS.md sección "Ampliación iNaturalist" para el detalle
# del proceso y la tasa de acierto (7/1000, mucho menor que la tanda original).
EXPANDED_YELLOW_INDICES = [17, 43, 46, 51, 52, 59, 68]


def load_expanded_yellow(candidates_file="expanded_candidates_v3.txt"):
    """Carga las imágenes confirmadas de la búsqueda ampliada, a partir del
    archivo de candidatas generado por el filtro de color (orden preservado,
    coincide con los índices usados en los contact sheets de curación)."""
    from pathlib import Path
    path = Path(candidates_file)
    if not path.exists():
        print(f"[Ampliación] No se encontró {candidates_file}, se omite.")
        return []
    with open(path) as f:
        all_candidates = [Path(line.strip()) for line in f]
    kept = [all_candidates[i] for i in EXPANDED_YELLOW_INDICES
            if i < len(all_candidates)]
    print(f"[Ampliación] {len(kept)} imágenes adicionales confirmadas "
          f"como morfo amarillo.")
    return kept

# Tercera fuente: manual_source_2 (Wikimedia + iNaturalist, vía herramienta
# propia del usuario). 7 imágenes confirmadas de 126 candidatas (tras filtro
# de licencia + filtro de color con min_ratio=0.30), de un total de 2000
# descargadas. Ver DATASET_DECISIONS.md sección "Fuente 3: manual_source_2".
MANUAL_SOURCE_2_INDICES = [1, 2, 13, 39, 46, 47, 74]


def load_manual_source_2(candidates_file="candidatas_color_v3.csv"):
    """Carga las imágenes confirmadas de manual_source_2, a partir del CSV
    de candidatas generado por el filtro de licencia + color (orden
    preservado, coincide con los índices usados en los contact sheets)."""
    import pandas as pd
    from pathlib import Path
    path = Path(candidates_file)
    if not path.exists():
        print(f"[manual_source_2] No se encontró {candidates_file}, se omite.")
        return []
    df = pd.read_csv(path)
    files = df['ruta_local'].apply(Path).tolist()
    kept = [files[i] for i in MANUAL_SOURCE_2_INDICES if i < len(files)]
    print(f"[manual_source_2] {len(kept)} imágenes adicionales confirmadas "
          f"como morfo amarillo.")
    return kept

def fetch_inaturalist(raw_dir: Path, target=INAT_TARGET):
    try:
        from pyinaturalist import get_observations
    except ImportError:
        print("[iNaturalist] Falta pyinaturalist. Instala con: "
              "pip install pyinaturalist")
        return []

    dest = raw_dir / "inaturalist"
    dest.mkdir(parents=True, exist_ok=True)

    print(f"[iNaturalist] Buscando observaciones de '{INAT_TAXON_NAME}'...")
    files = []
    page = 1
    per_page = 50
    metadata_rows = []
    while len(files) < target and page <= 10:
        resp = get_observations(
            taxon_name=INAT_TAXON_NAME,
            photos=True,
            quality_grade="research",
            license=["CC-BY", "CC-BY-NC", "CC0"],  # lista, no string con comas
            per_page=per_page,
            page=page,
        )
        results = resp.get("results", [])
        if not results:
            break
        for obs in results:
            photos = obs.get("photos", [])
            if not photos:
                continue
            url = photos[0]["url"].replace("square", "medium")
            obs_id = obs["id"]
            license_code = obs.get("license_code", "unknown")
            out_path = dest / f"inat_{obs_id}.jpg"
            if not out_path.exists():
                try:
                    r = requests.get(url, timeout=20)
                    r.raise_for_status()
                    out_path.write_bytes(r.content)
                except Exception as e:
                    print(f"  [!] error descargando obs {obs_id}: {e}")
                    continue
            files.append(out_path)
            metadata_rows.append({
                "archivo": out_path.name,
                "fuente": "iNaturalist",
                "licencia": license_code,
                "url_original": f"https://www.inaturalist.org/observations/{obs_id}",
            })
            if len(files) >= target:
                break
        page += 1

    (raw_dir / "inat_metadata.json").write_text(json.dumps(metadata_rows, indent=2))
    print(f"[iNaturalist] {len(files)} imágenes descargadas.")
    return files


# ---------------------------------------------------------------------------
# Fuente 4: PlantDoc (clase no_frijol, rechazo de especies distintas a frijol)
# Consolidado previamente en dataset_no_frijol/consolidado/, documentado en
# dataset_no_frijol/metadata_no_frijol.csv. Licencia: CC BY 4.0 (Singh et
# al., 2020). Ver DATASET_DECISIONS.md seccion "Clase no_frijol" para el
# detalle de la decision de agregar esta tercera clase.
# ---------------------------------------------------------------------------

NO_FRIJOL_TARGET = 350  # aprox. mismo orden de magnitud que sano/plaga


def fetch_no_frijol(no_frijol_dir: Path, target=NO_FRIJOL_TARGET):
    """Toma una muestra ESTRATIFICADA (no las 2578 completas, para no
    desbalancear el dataset) del set consolidado de PlantDoc, priorizando
    incluir TODA la clase 'Soyabean_leaf' (la especie mas parecida al
    frijol, mismo genero Fabaceae, hojas trifoliadas -- el caso mas dificil
    y valioso para que el modelo aprenda la distincion fina)."""
    import csv
    import random as _random
    from collections import defaultdict

    consolidado = no_frijol_dir / "consolidado"
    metadata_csv = no_frijol_dir / "metadata_no_frijol.csv"
    if not metadata_csv.exists():
        print(f"[no_frijol] No se encontro {metadata_csv}, se omite la clase.")
        return []

    por_clase = defaultdict(list)
    with open(metadata_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            por_clase[row["clase_original_plantdoc"]].append(row["filename"])

    # Incluir TODA la clase soya primero (caso mas valioso)
    seleccion = list(por_clase.pop("Soyabean leaf", []))
    restantes_objetivo = max(0, target - len(seleccion))

    # Repartir el resto proporcionalmente entre las demas clases
    otras_clases = list(por_clase.keys())
    _random.shuffle(otras_clases)
    por_otra_clase = max(1, restantes_objetivo // max(1, len(otras_clases)))
    for clase in otras_clases:
        archivos = por_clase[clase]
        _random.shuffle(archivos)
        seleccion += archivos[:por_otra_clase]

    files = [consolidado / nombre for nombre in seleccion if (consolidado / nombre).exists()]
    print(f"[no_frijol] {len(files)} imagenes seleccionadas de PlantDoc "
          f"(incluye {min(65, len(files))} de soya, especie mas dificil).")
    return files


# ---------------------------------------------------------------------------
# Organización final: split + resize + metadata.csv
# ---------------------------------------------------------------------------

def build_split(dataset_root: Path, class_name: str, files_with_source):
    """files_with_source: lista de tuplas (Path, fuente, licencia)."""
    random.shuffle(files_with_source)
    n_train = int(len(files_with_source) * TRAIN_RATIO)
    splits = {
        "train": files_with_source[:n_train],
        "val": files_with_source[n_train:],
    }
    metadata_rows = []
    for split_name, items in splits.items():
        out_dir = dataset_root / split_name / class_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for idx, (src, source, license_) in enumerate(tqdm(items, desc=f"{class_name}/{split_name}")):
            fname = f"{class_name}_{source}_{idx:04d}.jpg"
            dst = out_dir / fname
            ok = save_processed(src, dst)
            if ok:
                metadata_rows.append({
                    "archivo": f"{split_name}/{class_name}/{fname}",
                    "clase": class_name,
                    "split": split_name,
                    "fuente": source,
                    "licencia": license_,
                    "origen_local": str(src),
                })
    return metadata_rows


def write_summary(dataset_root: Path, metadata_rows):
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.DataFrame(metadata_rows)
    df.to_csv(dataset_root / "metadata.csv", index=False)

    counts = df.groupby(["clase", "split"]).size().unstack(fill_value=0)

    clases_presentes = sorted(df["clase"].unique())
    fig, axes = plt.subplots(len(clases_presentes), 4, figsize=(12, 3 * len(clases_presentes)))
    if len(clases_presentes) == 1:
        axes = axes.reshape(1, -1)
    for i, clase in enumerate(clases_presentes):
        subset = df[df["clase"] == clase]
        sample = subset.sample(min(4, len(subset))) if len(subset) else subset
        for j, (_, row) in enumerate(sample.iterrows()):
            img_path = dataset_root / row["archivo"]
            img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
            axes[i, j].imshow(img)
            axes[i, j].set_title(f"{clase} ({row['fuente']})", fontsize=8)
            axes[i, j].axis("off")
        for j in range(len(sample), 4):
            axes[i, j].axis("off")
    plt.tight_layout()
    fig.savefig(dataset_root / "ejemplos.png", dpi=120)

    with open(dataset_root / "resumen.md", "w") as f:
        f.write("# Resumen del dataset — PlagaScan\n\n")
        f.write("## Conteo por clase y split\n\n")
        f.write(counts.to_markdown())
        f.write("\n\n## Fuentes utilizadas\n\n")
        f.write(df.groupby(["clase", "fuente"]).size().unstack(fill_value=0).to_markdown())
        f.write("\n\nVer `ejemplos.png` para muestra visual y `metadata.csv` "
                "para trazabilidad completa (archivo, fuente, licencia).\n")
        f.write("\nVer DATASET_DECISIONS.md para la justificación de especie, "
                "fuentes y limitaciones metodológicas.\n")

    print("\n=== RESUMEN ===")
    print(counts)
    print(f"\nArchivos generados en: {dataset_root}")
    print("  - metadata.csv")
    print("  - resumen.md")
    print("  - ejemplos.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="~/plagascan/dataset")
    args = parser.parse_args()

    dataset_root = Path(args.root).expanduser()
    raw_dir = dataset_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # --- Clase sano ---
    makerere_files = fetch_makerere(raw_dir)
    sano_files = [(p, "makerere", "MIT") for p in makerere_files[:200]]

        # --- Clase plaga ---
    lelephid_files = fetch_lelephid(raw_dir)
    inat_files_raw = fetch_inaturalist(raw_dir)
    inat_files = filter_yellow_morph(inat_files_raw)
    inat_expanded_files = load_expanded_yellow()
    manual_source_2_files = load_manual_source_2()

    plaga_files = (
        [(p, "lelephid", "CC-BY-4.0") for p in lelephid_files[:150]]
        + [(p, "inaturalist", "ver-inat_metadata.json") for p in inat_files]
        + [(p, "inaturalist_expanded", "ver-inat_metadata.json") for p in inat_expanded_files]
        + [(p, "manual_source_2", "ver-metadata.csv-por-imagen") for p in manual_source_2_files]
    )

    if len(sano_files) < MIN_PER_CLASS:
        print(f"[!] ADVERTENCIA: solo {len(sano_files)} imágenes 'sano' "
              f"(objetivo mínimo {MIN_PER_CLASS}).")
    if len(plaga_files) < MIN_PER_CLASS:
        print(f"[!] ADVERTENCIA: solo {len(plaga_files)} imágenes 'plaga' "
              f"(objetivo mínimo {MIN_PER_CLASS}). El filtro de morfo amarillo "
              f"puede haber reducido bastante el pool de iNaturalist — "
              f"revisar si LeLePhid + iNaturalist filtrado alcanza; si no, "
              f"considerar bajar min_ratio en filter_yellow_morph().")

    # --- Clase no_frijol (rechazo de especies distintas) ---
    no_frijol_dir = Path("~/plagascan/dataset_no_frijol").expanduser()
    no_frijol_raw = fetch_no_frijol(no_frijol_dir)
    no_frijol_files = [(p, "plantdoc", "CC-BY-4.0") for p in no_frijol_raw]

    if len(no_frijol_files) < MIN_PER_CLASS:
        print(f"[!] ADVERTENCIA: solo {len(no_frijol_files)} imagenes 'no_frijol' "
              f"(objetivo minimo {MIN_PER_CLASS}).")

    all_metadata = []
    all_metadata += build_split(dataset_root, "sano", sano_files)
    all_metadata += build_split(dataset_root, "plaga", plaga_files)
    all_metadata += build_split(dataset_root, "no_frijol", no_frijol_files)

    write_summary(dataset_root, all_metadata)


if __name__ == "__main__":
    main()