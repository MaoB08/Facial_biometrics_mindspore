# -*- coding: utf-8 -*-
"""
Prepara la estructura de datos local para entrenamiento.
Crea data/train y data/val con subcarpetas por identidad.
Opcional: descarga un subset pequeño de ejemplo (LFW) para probar.
"""

import os
import sys
import urllib.request
import tarfile
import shutil

# Raíz del proyecto
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")


def create_structure():
    """Crea data/train y data/val vacíos."""
    for d in (TRAIN_DIR, VAL_DIR):
        os.makedirs(d, exist_ok=True)
    print(f"Estructura creada: {DATA_DIR}")
    print("  data/train/<identidad>/  <- pon aquí fotos de cada persona")
    print("  data/val/<identidad>/    <- opcional, fotos para validación")


def download_lfw_subset():
    """
    Descarga un subset muy pequeño de LFW para pruebas rápidas (unas pocas personas).
    LFW está en http://vis-www.cs.umass.edu/lfw/lfw.tgz
    """
    url = "http://vis-www.cs.umass.edu/lfw/lfw.tgz"
    dest_zip = os.path.join(DATA_DIR, "lfw.tgz")
    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.isfile(dest_zip):
        print("Descargando LFW (puede tardar unos minutos)...")
        try:
            urllib.request.urlretrieve(url, dest_zip)
        except Exception as e:
            print(f"Error descargando: {e}. Puedes descargar manualmente {url} y extraer en data/")
            return

    if not os.path.isdir(os.path.join(DATA_DIR, "lfw")):
        print("Extrayendo...")
        with tarfile.open(dest_zip, "r:gz") as tar:
            tar.extractall(DATA_DIR)

    # LFW extrae en data/lfw/ con subcarpetas por persona
    lfw_root = os.path.join(DATA_DIR, "lfw")
    if not os.path.isdir(lfw_root):
        print("No se encontró carpeta lfw tras extraer.")
        return

    persons = [d for d in os.listdir(lfw_root) if os.path.isdir(os.path.join(lfw_root, d))]
    persons = sorted(persons)[:15]  # 15 personas para ejemplo
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)
    for p in persons:
        src_dir = os.path.join(lfw_root, p)
        imgs = [f for f in os.listdir(src_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if len(imgs) < 3:
            continue
        os.makedirs(os.path.join(TRAIN_DIR, p), exist_ok=True)
        os.makedirs(os.path.join(VAL_DIR, p), exist_ok=True)
        for i, f in enumerate(imgs):
            src = os.path.join(src_dir, f)
            if i < 2:
                shutil.copy2(src, os.path.join(VAL_DIR, p, f))
            else:
                shutil.copy2(src, os.path.join(TRAIN_DIR, p, f))
    print(f"Listo: {len(persons)} personas en data/train y data/val (subset LFW).")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--download-lfw", action="store_true", help="Descargar subset de LFW como ejemplo")
    args = p.parse_args()
    create_structure()
    if args.download_lfw:
        download_lfw_subset()
    else:
        print("Para descargar un dataset de ejemplo: python scripts/prepare_data.py --download-lfw")


if __name__ == "__main__":
    main()
