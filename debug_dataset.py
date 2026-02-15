#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script de diagnóstico para verificar la carga del dataset."""

import os
import sys
import mindspore.dataset as ds
from mindspore.dataset import vision

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import TRAIN_DIR, IMAGE_SIZE

print("=" * 60)
print("Diagnóstico del Dataset")
print("=" * 60)

# Verificar directorio
print(f"\n1. Directorio de entrenamiento: {TRAIN_DIR}")
print(f"   Existe: {os.path.exists(TRAIN_DIR)}")

# Listar personas
personas = [d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))]
print(f"\n2. Personas encontradas: {len(personas)}")
for persona in personas:
    persona_dir = os.path.join(TRAIN_DIR, persona)
    archivos = os.listdir(persona_dir)
    print(f"   - {persona}: {len(archivos)} archivos")
    for archivo in archivos[:3]:  # Mostrar primeros 3
        print(f"      • {archivo}")

# Intentar cargar con ImageFolderDataset
print("\n3. Intentando cargar con ImageFolderDataset...")
try:
    dataset = ds.ImageFolderDataset(
        TRAIN_DIR,
        decode=True,
        shuffle=False,
        extensions=[".jpg", ".jpeg", ".png", ".bmp"],
    )
    
    size = dataset.get_dataset_size()
    print(f"   ✅ Dataset cargado exitosamente")
    print(f"   Tamaño del dataset: {size} muestras")
    
    # Intentar obtener una muestra
    print("\n4. Intentando obtener primera muestra...")
    iterator = dataset.create_dict_iterator(num_epochs=1)
    first_batch = next(iterator, None)
    
    if first_batch:
        print(f"   ✅ Primera muestra obtenida")
        print(f"   Forma de imagen: {first_batch['image'].shape}")
        print(f"   Label: {first_batch['label']}")
    else:
        print(f"   ❌ No se pudo obtener ninguna muestra")
    
    # Probar con transformaciones
    print("\n5. Probando con transformaciones...")
    transforms = [
        vision.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        vision.HWC2CHW(),
        vision.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            is_hwc=False,
        ),
    ]
    
    dataset2 = ds.ImageFolderDataset(
        TRAIN_DIR,
        decode=True,
        shuffle=False,
        extensions=[".jpg", ".jpeg", ".png", ".bmp"],
    )
    dataset2 = dataset2.map(transforms, input_columns="image")
    dataset2 = dataset2.batch(2, drop_remainder=True)
    
    size2 = dataset2.get_dataset_size()
    print(f"   Dataset con transformaciones: {size2} batches")
    
    if size2 == 0:
        print(f"   ❌ PROBLEMA: El dataset tiene 0 batches después de aplicar transformaciones")
        print(f"   Esto puede deberse a drop_remainder=True con muy pocas muestras")
    else:
        print(f"   ✅ Dataset con transformaciones funciona correctamente")
    
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
