# -*- coding: utf-8 -*-
"""Configuración del modelo de biometría facial."""

import os

# Rutas (todo local)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Imagen de entrada (estándar en reconocimiento facial)
IMAGE_SIZE = 112
INPUT_CHANNELS = 3

# Modelo
EMBEDDING_DIM = 128

# Entrenamiento
BATCH_SIZE = 4  # Reducido para datasets pequeños (ajustar a 32 con más datos)
EPOCHS = 30
LEARNING_RATE = 1e-3

# Verificación (umbral de similitud coseno para considerar "misma persona")
VERIFICATION_THRESHOLD = 0.5

# Semilla para reproducibilidad
SEED = 42
