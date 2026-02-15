# -*- coding: utf-8 -*-
"""
Entrenamiento local del modelo de biometría facial con MindSpore.
Todo se ejecuta en tu máquina (CPU o GPU).
"""

import json
import os
import sys

import mindspore as ms
from mindspore import nn
from mindspore.train import Model, LossMonitor, TimeMonitor
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint

# Añadir raíz del proyecto
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    TRAIN_DIR,
    VAL_DIR,
    CHECKPOINT_DIR,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    EMBEDDING_DIM,
    SEED,
)
from src.dataset import create_train_dataset, create_val_dataset, get_num_classes_from_dir
from src.model import FaceBiometricsNet


def main():
    ms.set_seed(SEED)
    # Contexto: CPU o GPU según disponibilidad
    ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")  # Cambiar a "GPU" si tienes CUDA

    num_classes = get_num_classes_from_dir(TRAIN_DIR)
    if num_classes < 2:
        raise ValueError(
            f"Se necesitan al menos 2 identidades en {TRAIN_DIR}. "
            "Estructura: data/train/<nombre_persona>/<fotos>.jpg"
        )

    print(f"Identidades en entrenamiento: {num_classes}")
    train_ds = create_train_dataset(batch_size=BATCH_SIZE)
    val_ds = create_val_dataset(batch_size=BATCH_SIZE)

    net = FaceBiometricsNet(embedding_dim=EMBEDDING_DIM, num_classes=num_classes)
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    opt = nn.Adam(net.trainable_params(), learning_rate=LEARNING_RATE)

    model = Model(network=net, loss_fn=loss_fn, optimizer=opt, metrics={"acc"})

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    ckpt_config = CheckpointConfig(
        save_checkpoint_steps=train_ds.get_dataset_size(),
        keep_checkpoint_max=3,
    )
    ckpt_cb = ModelCheckpoint(
        prefix="face_biometrics",
        directory=CHECKPOINT_DIR,
        config=ckpt_config,
    )

    print("Iniciando entrenamiento local...")
    model.train(
        EPOCHS,
        train_ds,
        callbacks=[LossMonitor(50), TimeMonitor(50), ckpt_cb],
        dataset_sink_mode=False,
    )

    if val_ds is not None:
        acc = model.eval(val_ds, dataset_sink_mode=False)
        print(f"Precisión en validación: {acc}")

    # Guardar config para inferencia (num_classes, embedding_dim)
    config_path = os.path.join(CHECKPOINT_DIR, "model_config.json")
    with open(config_path, "w") as f:
        json.dump({"num_classes": num_classes, "embedding_dim": EMBEDDING_DIM}, f)
    print(f"Config guardada en: {config_path}")
    print(f"Checkpoints guardados en: {CHECKPOINT_DIR}")


if __name__ == "__main__":
    main()
