#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para convertir imágenes WEBP a JPG en el dataset.
Convierte todas las imágenes .webp en data/train/ y data/val/ a formato .jpg
"""

import os
import sys
from pathlib import Path
from PIL import Image

# Añadir raíz del proyecto
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, TRAIN_DIR, VAL_DIR


def convert_webp_to_jpg(directory):
    """
    Convierte todas las imágenes .webp en un directorio a .jpg
    
    Args:
        directory: Ruta del directorio a procesar
    """
    if not os.path.exists(directory):
        print(f"⚠️  Directorio no encontrado: {directory}")
        return 0
    
    converted_count = 0
    
    # Recorrer recursivamente todas las carpetas
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.lower().endswith('.webp'):
                webp_path = os.path.join(root, filename)
                
                # Crear nombre del archivo JPG
                jpg_filename = os.path.splitext(filename)[0] + '.jpg'
                jpg_path = os.path.join(root, jpg_filename)
                
                try:
                    # Abrir imagen WEBP y convertir a RGB
                    img = Image.open(webp_path)
                    
                    # Convertir a RGB (necesario para JPG)
                    if img.mode in ('RGBA', 'LA', 'P'):
                        # Crear fondo blanco para imágenes con transparencia
                        background = Image.new('RGB', img.size, (255, 255, 255))
                        if img.mode == 'P':
                            img = img.convert('RGBA')
                        background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                        img = background
                    elif img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Guardar como JPG
                    img.save(jpg_path, 'JPEG', quality=95)
                    
                    # Eliminar archivo WEBP original
                    os.remove(webp_path)
                    
                    converted_count += 1
                    print(f"✅ Convertido: {os.path.relpath(webp_path, directory)} → {jpg_filename}")
                    
                except Exception as e:
                    print(f"❌ Error al convertir {webp_path}: {e}")
    
    return converted_count


def main():
    print("=" * 60)
    print("Conversión de imágenes WEBP a JPG")
    print("=" * 60)
    
    total_converted = 0
    
    # Convertir en train/
    if os.path.exists(TRAIN_DIR):
        print(f"\n📁 Procesando: {TRAIN_DIR}")
        count = convert_webp_to_jpg(TRAIN_DIR)
        total_converted += count
        print(f"   → {count} imágenes convertidas en train/")
    
    # Convertir en val/
    if os.path.exists(VAL_DIR):
        print(f"\n📁 Procesando: {VAL_DIR}")
        count = convert_webp_to_jpg(VAL_DIR)
        total_converted += count
        print(f"   → {count} imágenes convertidas en val/")
    
    print("\n" + "=" * 60)
    print(f"✨ Total: {total_converted} imágenes convertidas a JPG")
    print("=" * 60)
    
    # Verificar resultado
    print("\n📊 Resumen del dataset:")
    for dataset_name, dataset_dir in [("Train", TRAIN_DIR), ("Val", VAL_DIR)]:
        if os.path.exists(dataset_dir):
            personas = [d for d in os.listdir(dataset_dir) 
                       if os.path.isdir(os.path.join(dataset_dir, d))]
            print(f"\n{dataset_name}:")
            for persona in sorted(personas):
                persona_dir = os.path.join(dataset_dir, persona)
                imagenes = [f for f in os.listdir(persona_dir) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                print(f"  - {persona}: {len(imagenes)} imágenes")


if __name__ == "__main__":
    main()
