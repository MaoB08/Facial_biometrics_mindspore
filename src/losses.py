# -*- coding: utf-8 -*-
"""Funciones de pérdida para biometría facial de alta seguridad."""

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops


class TripletLoss(nn.Cell):
    """
    Triplet Loss para aprendizaje de embeddings discriminativos.
    
    Fuerza a que:
    - La distancia entre anchor y positive sea pequeña
    - La distancia entre anchor y negative sea grande
    - Con un margen mínimo de separación
    
    Args:
        margin: Margen mínimo entre distancias positivas y negativas (default: 0.3)
        
    Formula:
        loss = max(0, d(anchor, positive) - d(anchor, negative) + margin)
    """
    
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin
        self.sqrt = ops.Sqrt()
        self.reduce_sum = ops.ReduceSum(keep_dims=False)
        self.reduce_mean = ops.ReduceMean(keep_dims=False)
        self.maximum = ops.Maximum()
        
    def construct(self, anchor, positive, negative):
        """
        Args:
            anchor: Embeddings de imágenes ancla (B, embedding_dim)
            positive: Embeddings de la misma persona (B, embedding_dim)
            negative: Embeddings de persona diferente (B, embedding_dim)
            
        Returns:
            Scalar loss value
        """
        # Distancia euclidiana al cuadrado entre anchor y positive
        pos_dist = self.reduce_sum((anchor - positive) ** 2, axis=1)
        
        # Distancia euclidiana al cuadrado entre anchor y negative
        neg_dist = self.reduce_sum((anchor - negative) ** 2, axis=1)
        
        # Triplet loss: queremos que pos_dist < neg_dist - margin
        # Es decir: pos_dist - neg_dist + margin < 0
        # Usamos max(0, ...) para solo penalizar violaciones
        losses = self.maximum(pos_dist - neg_dist + self.margin, 0.0)
        
        return self.reduce_mean(losses)


class TripletLossWithSoftmax(nn.Cell):
    """
    Combinación de Triplet Loss + Softmax CrossEntropy.
    
    Útil durante las primeras épocas para tener una señal de aprendizaje más fuerte.
    
    Args:
        margin: Margen para triplet loss
        alpha: Peso de triplet loss (default: 1.0)
        beta: Peso de classification loss (default: 0.5)
    """
    
    def __init__(self, margin=0.3, alpha=1.0, beta=0.5):
        super().__init__()
        self.triplet_loss = TripletLoss(margin=margin)
        self.ce_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
        self.alpha = alpha
        self.beta = beta
        
    def construct(self, anchor_emb, pos_emb, neg_emb, logits, labels):
        """
        Args:
            anchor_emb: Embeddings ancla
            pos_emb: Embeddings positivos
            neg_emb: Embeddings negativos
            logits: Salida del clasificador (B, num_classes)
            labels: Etiquetas verdaderas (B,)
            
        Returns:
            Combined loss
        """
        triplet = self.triplet_loss(anchor_emb, pos_emb, neg_emb)
        classification = self.ce_loss(logits, labels)
        
        return self.alpha * triplet + self.beta * classification
