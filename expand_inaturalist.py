#!/usr/bin/env python3
"""
PlagaScan — expand_inaturalist.py

Búsqueda AMPLIADA en iNaturalist para encontrar más candidatas de morfo
amarillo, sin tocar build_dataset.py (que ya está cerrado y funcionando).

Cambios respecto a la búsqueda original:
  - Incluye Aphis craccivora además de Aphis gossypii
  - quality_grade incluye "needs_id" además de "research" (más candidatas,
    menos certeza taxonómica -> se revisa igual a mano después)
  - Trae hasta 500 por especie en vez de 150

Uso:
    python expand_inaturalist.py --root ~/plagascan/dataset

Salida:
    ~/plagascan/dataset/raw/inaturalist_expanded/*.jpg
    Contact sheets numerados en la carpeta actual, listos para curar a mano
    igual que hicimos con la primera tanda.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import requests
from pyinaturalist import get_observations

SPECIES = ["Aphis gossypii", "Aphis craccivora"]
TARGET_PER_SPECIES = 500
QUALITY_GRADES = ["research", "needs_id"]


def fetch_species(dest: Path, taxon_name: str, target: int):
    print(f"[{taxon_name}] Buscando (quality_grade={QUALITY_GRADES})...")
    files = []
    metadata_rows = []
    page = 1
    per_page = 50
    while len(files) < target and page <= 20:
        resp = get_observations(
            taxon_name=taxon_name,
            photos=True,
            quality_grade=QUALITY_GRADES,
            license=["CC-BY", "CC-BY-NC", "CC0"],
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
            safe_name = taxon_name.replace(" ", "_")
            out_path = dest / f"{safe_name}_{obs_id}.jpg"
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
                "especie": taxon_name,
                "fuente": "iNaturalist",
                "licencia": license_code,
                "url_original": f"https://www.inaturalist.org/observations/{obs_id}",
            })
            if len(files) >= target:
                break
        page += 1
    print(f"[{taxon_name}] {len(files)} imágenes descargadas.")
    return files, metadata_rows


def make_contact_sheets(files, out_prefix, batch_size=30, n_cols=6):
    for batch_start in range(0, len(files), batch_size):
        batch = files[batch_start:batch_start + batch_size]
        thumbs = []
        for i, f in enumerate(batch):
            img = cv2.imread(str(f))
            if img is None:
                continue
            img = cv2.resize(img, (160, 160))
            idx = batch_start + i
            cv2.rectangle(img, (0, 0), (55, 25), (0, 0, 0), -1)
            cv2.putText(img, str(idx), (5, 18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), 2)
            thumbs.append(img)
        while len(thumbs) % n_cols != 0:
            thumbs.append(np.zeros((160, 160, 3), dtype=np.uint8))
        rows = [np.hstack(thumbs[i:i + n_cols]) for i in range(0, len(thumbs), n_cols)]
        grid = np.vstack(rows)
        out_name = f"{out_prefix}_{batch_start:03d}.jpg"
        cv2.imwrite(out_name, grid)
        print(f"Guardado: {out_name} (índices {batch_start}-{batch_start + len(batch) - 1})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="~/plagascan/dataset")
    args = parser.parse_args()

    dataset_root = Path(args.root).expanduser()
    dest = dataset_root / "raw" / "inaturalist_expanded"
    dest.mkdir(parents=True, exist_ok=True)

    all_files = []
    all_metadata = []
    for species in SPECIES:
        files, meta = fetch_species(dest, species, TARGET_PER_SPECIES)
        all_files.extend(files)
        all_metadata.extend(meta)

    (dest / "expanded_metadata.json").write_text(json.dumps(all_metadata, indent=2))

    # orden estable para poder referenciar índices en la curación manual,
    # igual que se hizo con la tanda original
    all_files_sorted = sorted(set(all_files))
    print(f"\nTotal combinado (ambas especies, sin duplicados): {len(all_files_sorted)}")

    make_contact_sheets(all_files_sorted, "contact_expanded")
    print("\nListo. Revisa los contact_expanded_*.jpg y dime qué índices son "
          "morfo amarillo, igual que la vez pasada.")


if __name__ == "__main__":
    main()
