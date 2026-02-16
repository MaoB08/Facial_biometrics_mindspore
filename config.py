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
EPOCHS = 100  # Aumentado para mejor convergencia con Triplet Loss
LEARNING_RATE = 5e-4  # Reducido para entrenamiento más estable

# Verificación (umbral de similitud coseno para considerar "misma persona")
# ALTA SEGURIDAD: Prioriza rechazar impostores sobre aceptar a todos
VERIFICATION_THRESHOLD = 0.75  # Modo de alta seguridad (recomendado)
STRICT_VERIFICATION_THRESHOLD = 0.85  # Modo ultra-seguro (máxima seguridad)

# Triplet Loss
TRIPLET_MARGIN = 0.3  # Margen mínimo entre embeddings de diferentes personas
TRIPLET_WEIGHT = 1.0  # Peso de triplet loss en entrenamiento combinado
CE_WEIGHT = 0.5  # Peso de classification loss

# Semilla para reproducibilidad
SEED = 42
