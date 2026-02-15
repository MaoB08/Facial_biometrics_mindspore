# -*- coding: utf-8 -*-
"""Red neuronal para embeddings faciales (MindSpore)."""

import mindspore as ms
import mindspore.nn as nn
from mindspore.common.initializer import Normal


class ConvBlock(nn.Cell):
    """Bloque Conv2d + BatchNorm + ReLU + MaxPool."""

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size=kernel_size, stride=stride, pad_mode="same", has_bias=False
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def construct(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class FaceEmbeddingNet(nn.Cell):
    """
    Red que extrae vectores de embedding (128-d) a partir de caras.
    Entrada: (B, 3, 112, 112), Salida: (B, embedding_dim).
    """

    def __init__(self, embedding_dim=128):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.features = nn.SequentialCell(
            ConvBlock(3, 32),   # -> 56x56
            ConvBlock(32, 64),  # -> 28x28
            ConvBlock(64, 128), # -> 14x14
            ConvBlock(128, 256), # -> 7x7
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Dense(256 * 7 * 7, embedding_dim, weight_init=Normal(0.02), bias_init="zeros")

    def construct(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = ms.ops.L2Normalize(axis=1)(x)
        return x


class FaceBiometricsNet(nn.Cell):
    """
    Modelo completo para entrenamiento: backbone + cabeza de clasificación.
    Para inferencia/verificación se usa solo el backbone (FaceEmbeddingNet).
    """

    def __init__(self, embedding_dim=128, num_classes=10):
        super().__init__()
        self.backbone = FaceEmbeddingNet(embedding_dim=embedding_dim)
        self.classifier = nn.Dense(
            embedding_dim, num_classes, weight_init=Normal(0.02), bias_init="zeros"
        )

    def construct(self, x):
        embeddings = self.backbone(x)
        logits = self.classifier(embeddings)
        return logits

    def get_embedding(self, x):
        """Devuelve solo el vector de embedding (para verificación)."""
        return self.backbone(x)


