# -*- coding: utf-8 -*-
"""Inferencia local: extracción de embeddings y verificación 1:1."""

import os
import sys
import json

import mindspore as ms
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CHECKPOINT_DIR, IMAGE_SIZE, EMBEDDING_DIM
from src.model import FaceBiometricsNet

# Normalización ImageNet (igual que en entrenamiento)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


def _load_image_tensor(image_path):
    """Carga una imagen y aplica resize + normalización (CHW, batch=1)."""
    pil = Image.open(image_path).convert("RGB")
    arr = np.array(pil, dtype=np.float32) / 255.0
    # Resize
    img = Image.fromarray((arr * 255).astype(np.uint8))
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    # HWC -> CHW
    arr = np.transpose(arr, (2, 0, 1))
    # Normalize
    arr = (arr - MEAN.reshape(3, 1, 1)) / STD.reshape(3, 1, 1)
    return ms.Tensor(arr[np.newaxis, ...], dtype=ms.float32)


def load_embedding_model(checkpoint_dir=None):
    """Carga el modelo de embeddings desde el directorio de checkpoints."""
    checkpoint_dir = checkpoint_dir or CHECKPOINT_DIR
    config_path = os.path.join(checkpoint_dir, "model_config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"No se encontró {config_path}. Entrena antes con: python train.py"
        )
    with open(config_path) as f:
        config = json.load(f)
    num_classes = config["num_classes"]
    embedding_dim = config.get("embedding_dim", EMBEDDING_DIM)

    # Buscar último checkpoint
    ckpts = [f for f in os.listdir(checkpoint_dir) if f.startswith("face_biometrics") and f.endswith(".ckpt")]
    if not ckpts:
        raise FileNotFoundError(f"No hay .ckpt en {checkpoint_dir}")
    ckpt_path = os.path.join(checkpoint_dir, sorted(ckpts)[-1])

    full_net = FaceBiometricsNet(embedding_dim=embedding_dim, num_classes=num_classes)
    param_dict = ms.load_checkpoint(ckpt_path)
    ms.load_param_into_net(full_net, param_dict)
    full_net.set_train(False)
    return full_net


def get_embedding(model, image_path):
    """Obtiene el vector de embedding (128-d) para una imagen de rostro."""
    x = _load_image_tensor(image_path)
    emb = model.get_embedding(x)
    return emb.asnumpy().flatten()


def verify_pair(model, path_a, path_b, threshold=0.5):
    """
    Verificación 1:1: ¿son la misma persona?
    Devuelve (es_misma_persona: bool, similitud: float).
    """
    emb_a = get_embedding(model, path_a)
    emb_b = get_embedding(model, path_b)
    sim = np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b) + 1e-8)
    return bool(sim >= threshold), float(sim)


