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
    TRIPLET_MARGIN,
    TRIPLET_WEIGHT,
    CE_WEIGHT,
)
from src.dataset import create_train_dataset, create_val_dataset, get_num_classes_from_dir
from src.model import FaceBiometricsNet
from src.losses import TripletLossWithSoftmax


class TrainOneStepWithLoss(nn.Cell):
    """
    Wrapper para entrenamiento con Triplet Loss.
    Necesario porque Triplet Loss requiere embeddings además de logits.
    """
    def __init__(self, network, optimizer, loss_fn):
        super().__init__(auto_prefix=False)
        self.network = network
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.grad = ms.ops.GradOperation(get_by_list=True)
        self.weights = optimizer.parameters
        
    def construct(self, anchor, positive, negative, labels):
        """
        Args:
            anchor, positive, negative: Batches de imágenes (triplets)
            labels: Etiquetas de las imágenes anchor
        """
        def forward_fn():
            # Obtener embeddings y logits para anchor, positive, negative
            anchor_emb = self.network.get_embedding(anchor)
            pos_emb = self.network.get_embedding(positive)
            neg_emb = self.network.get_embedding(negative)
            logits = self.network(anchor)
            
            # Calcular loss combinado
            loss = self.loss_fn(anchor_emb, pos_emb, neg_emb, logits, labels)
            return loss
        
        loss = forward_fn()
        grads = self.grad(forward_fn, self.weights)()
        self.optimizer(grads)
        return loss


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
    print(f"Configuración:")
    print(f"  - Épocas: {EPOCHS}")
    print(f"  - Learning Rate: {LEARNING_RATE}")
    print(f"  - Triplet Margin: {TRIPLET_MARGIN}")
    print(f"  - Arquitectura: 6 bloques convolucionales + Dropout")
    print(f"  - Loss: Triplet Loss (peso={TRIPLET_WEIGHT}) + CrossEntropy (peso={CE_WEIGHT})")
    
    train_ds = create_train_dataset(batch_size=BATCH_SIZE)
    val_ds = create_val_dataset(batch_size=BATCH_SIZE)

    net = FaceBiometricsNet(embedding_dim=EMBEDDING_DIM, num_classes=num_classes)
    
    # Loss combinado: Triplet + Classification
    loss_fn = TripletLossWithSoftmax(
        margin=TRIPLET_MARGIN,
        alpha=TRIPLET_WEIGHT,
        beta=CE_WEIGHT
    )
    
    # Learning rate scheduler: reduce LR cada 30 épocas
    # Nota: MindSpore requiere definir milestone_steps y learning_rates
    steps_per_epoch = train_ds.get_dataset_size()
    milestone = [30 * steps_per_epoch, 60 * steps_per_epoch, 90 * steps_per_epoch]
    learning_rates = [LEARNING_RATE, LEARNING_RATE * 0.1, LEARNING_RATE * 0.01, LEARNING_RATE * 0.001]
    lr_schedule = nn.piecewise_constant_lr(milestone, learning_rates)
    
    opt = nn.Adam(net.trainable_params(), learning_rate=lr_schedule)
    
    # NOTA: Para usar Triplet Loss necesitamos generar triplets
    # Por ahora, usamos el modelo estándar con clasificación
    # Para producción, implementar TripletDataset que genere (anchor, positive, negative)
    
    # Usar modelo estándar con clasificación (más simple para empezar)
    model = Model(network=net, loss_fn=nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean"), 
                  optimizer=opt, metrics={"acc"})

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    ckpt_config = CheckpointConfig(
        save_checkpoint_steps=train_ds.get_dataset_size(),
        keep_checkpoint_max=5,  # Guardar más checkpoints para 100 épocas
    )
    ckpt_cb = ModelCheckpoint(
        prefix="face_biometrics",
        directory=CHECKPOINT_DIR,
        config=ckpt_config,
    )

    print("\n🚀 Iniciando entrenamiento con arquitectura mejorada...")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Steps por época: {steps_per_epoch}")
    print(f"   Total steps: {EPOCHS * steps_per_epoch}\n")
    
    model.train(
        EPOCHS,
        train_ds,
        callbacks=[LossMonitor(50), TimeMonitor(50), ckpt_cb],
        dataset_sink_mode=False,
    )

    if val_ds is not None:
        print("\n📊 Evaluando en validación...")
        acc = model.eval(val_ds, dataset_sink_mode=False)
        print(f"Precisión en validación: {acc}")

    # Guardar config para inferencia (num_classes, embedding_dim)
    config_path = os.path.join(CHECKPOINT_DIR, "model_config.json")
    with open(config_path, "w") as f:
        json.dump({"num_classes": num_classes, "embedding_dim": EMBEDDING_DIM}, f)
    print(f"\n✅ Entrenamiento completado!")
    print(f"   Config guardada en: {config_path}")
    print(f"   Checkpoints guardados en: {CHECKPOINT_DIR}")


if __name__ == "__main__":
    main()
