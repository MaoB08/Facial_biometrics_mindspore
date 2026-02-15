# -*- coding: utf-8 -*-
"""Carga y preprocesado de datos para entrenamiento y prueba (100% local)."""

import os
import mindspore.dataset as ds
from mindspore.dataset import vision

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRAIN_DIR, VAL_DIR, IMAGE_SIZE, BATCH_SIZE


def get_train_transforms():
    """Transformaciones para entrenamiento (imagen ya decodificada con decode=True)."""
    return [
        vision.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        vision.RandomHorizontalFlip(0.5),
        vision.HWC2CHW(),
        vision.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            is_hwc=False,
        ),
    ]


def get_eval_transforms():
    """Transformaciones para validación/prueba: resize y normalización."""
    return [
        vision.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        vision.HWC2CHW(),
        vision.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            is_hwc=False,
        ),
    ]


def create_train_dataset(data_dir=None, batch_size=None, shuffle=True):
    """
    Crea el dataset de entrenamiento desde carpetas por identidad.
    Estructura esperada: data_dir/identidad_1/img1.jpg, img2.jpg ...
    """
    data_dir = data_dir or TRAIN_DIR
    batch_size = batch_size or BATCH_SIZE
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Directorio de entrenamiento no encontrado: {data_dir}. "
            "Crea la estructura data/train/<identidad>/<imagenes> o ejecuta scripts/prepare_data.py"
        )
    dataset = ds.ImageFolderDataset(
        data_dir,
        decode=True,
        shuffle=shuffle,
        extensions=[".jpg", ".jpeg", ".png", ".bmp"],
    )
    dataset = dataset.map(get_train_transforms(), input_columns="image")
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset


def create_val_dataset(data_dir=None, batch_size=None):
    """Crea el dataset de validación."""
    data_dir = data_dir or VAL_DIR
    batch_size = batch_size or BATCH_SIZE
    if not os.path.isdir(data_dir):
        return None
    dataset = ds.ImageFolderDataset(
        data_dir,
        decode=True,
        shuffle=False,
        extensions=[".jpg", ".jpeg", ".png", ".bmp"],
    )
    dataset = dataset.map(get_eval_transforms(), input_columns="image")
    dataset = dataset.batch(batch_size, drop_remainder=False)
    return dataset


def get_num_classes_from_dir(data_dir):
    """Obtiene el número de clases (identidades) contando subcarpetas."""
    if not os.path.isdir(data_dir):
        return 0
    return len([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
