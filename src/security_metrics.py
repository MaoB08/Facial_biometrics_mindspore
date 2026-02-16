# -*- coding: utf-8 -*-
"""Métricas de seguridad para evaluación del sistema biométrico."""

import numpy as np
from typing import List, Tuple, Dict


def calculate_similarity_distribution(similarities: List[float], 
                                      labels: List[bool]) -> Dict[str, float]:
    """
    Calcula estadísticas de distribución de similitudes.
    
    Args:
        similarities: Lista de valores de similitud coseno
        labels: Lista de booleanos (True = misma persona, False = diferente)
        
    Returns:
        Diccionario con estadísticas
    """
    genuine_scores = [s for s, l in zip(similarities, labels) if l]
    impostor_scores = [s for s, l in zip(similarities, labels) if not l]
    
    return {
        'genuine_mean': np.mean(genuine_scores) if genuine_scores else 0.0,
        'genuine_std': np.std(genuine_scores) if genuine_scores else 0.0,
        'genuine_min': np.min(genuine_scores) if genuine_scores else 0.0,
        'genuine_max': np.max(genuine_scores) if genuine_scores else 0.0,
        'impostor_mean': np.mean(impostor_scores) if impostor_scores else 0.0,
        'impostor_std': np.std(impostor_scores) if impostor_scores else 0.0,
        'impostor_min': np.min(impostor_scores) if impostor_scores else 0.0,
        'impostor_max': np.max(impostor_scores) if impostor_scores else 0.0,
        'num_genuine': len(genuine_scores),
        'num_impostor': len(impostor_scores),
    }


def calculate_far_frr(similarities: List[float], 
                      labels: List[bool], 
                      threshold: float) -> Tuple[float, float]:
    """
    Calcula FAR (False Acceptance Rate) y FRR (False Rejection Rate).
    
    FAR: Porcentaje de impostores aceptados incorrectamente
    FRR: Porcentaje de personas genuinas rechazadas incorrectamente
    
    Args:
        similarities: Lista de valores de similitud coseno
        labels: Lista de booleanos (True = misma persona, False = diferente)
        threshold: Umbral de decisión
        
    Returns:
        Tupla (FAR, FRR)
    """
    genuine_scores = [s for s, l in zip(similarities, labels) if l]
    impostor_scores = [s for s, l in zip(similarities, labels) if not l]
    
    # FAR: Impostores que pasan el umbral (falsos positivos)
    if impostor_scores:
        false_accepts = sum(1 for s in impostor_scores if s >= threshold)
        far = false_accepts / len(impostor_scores)
    else:
        far = 0.0
    
    # FRR: Genuinos que no pasan el umbral (falsos negativos)
    if genuine_scores:
        false_rejects = sum(1 for s in genuine_scores if s < threshold)
        frr = false_rejects / len(genuine_scores)
    else:
        frr = 0.0
    
    return far, frr


def calculate_eer(similarities: List[float], 
                  labels: List[bool]) -> Tuple[float, float]:
    """
    Calcula EER (Equal Error Rate) - punto donde FAR = FRR.
    
    Args:
        similarities: Lista de valores de similitud coseno
        labels: Lista de booleanos (True = misma persona, False = diferente)
        
    Returns:
        Tupla (EER, threshold_at_eer)
    """
    # Probar diferentes umbrales
    thresholds = np.linspace(0, 1, 100)
    min_diff = float('inf')
    eer = 0.0
    eer_threshold = 0.0
    
    for threshold in thresholds:
        far, frr = calculate_far_frr(similarities, labels, threshold)
        diff = abs(far - frr)
        
        if diff < min_diff:
            min_diff = diff
            eer = (far + frr) / 2
            eer_threshold = threshold
    
    return eer, eer_threshold


def get_confidence_level(similarity: float) -> str:
    """
    Determina el nivel de confianza basado en la similitud.
    
    Args:
        similarity: Valor de similitud coseno (0-1)
        
    Returns:
        String describiendo el nivel de confianza
    """
    if similarity < 0.5:
        return "RECHAZADO - Definitivamente NO es la persona"
    elif similarity < 0.75:
        return "RECHAZADO - Dudoso (rechazado por seguridad)"
    elif similarity < 0.85:
        return "ACEPTADO - Probable coincidencia"
    else:
        return "ACEPTADO - Alta confianza"


def print_security_report(similarities: List[float], 
                         labels: List[bool], 
                         threshold: float = 0.75):
    """
    Imprime un reporte completo de seguridad.
    
    Args:
        similarities: Lista de valores de similitud coseno
        labels: Lista de booleanos (True = misma persona, False = diferente)
        threshold: Umbral de decisión
    """
    print("\n" + "="*70)
    print("REPORTE DE SEGURIDAD - SISTEMA DE BIOMETRÍA FACIAL")
    print("="*70)
    
    # Distribución de similitudes
    dist = calculate_similarity_distribution(similarities, labels)
    print("\n📊 DISTRIBUCIÓN DE SIMILITUDES:")
    print(f"  Pares Genuinos (misma persona): {dist['num_genuine']}")
    print(f"    - Media: {dist['genuine_mean']:.4f} ± {dist['genuine_std']:.4f}")
    print(f"    - Rango: [{dist['genuine_min']:.4f}, {dist['genuine_max']:.4f}]")
    print(f"\n  Pares Impostores (diferente persona): {dist['num_impostor']}")
    print(f"    - Media: {dist['impostor_mean']:.4f} ± {dist['impostor_std']:.4f}")
    print(f"    - Rango: [{dist['impostor_min']:.4f}, {dist['impostor_max']:.4f}]")
    
    # FAR y FRR
    far, frr = calculate_far_frr(similarities, labels, threshold)
    print(f"\n🔒 MÉTRICAS DE SEGURIDAD (Umbral = {threshold}):")
    print(f"  FAR (False Acceptance Rate): {far*100:.2f}%")
    print(f"    → Porcentaje de impostores aceptados (CRÍTICO)")
    print(f"  FRR (False Rejection Rate): {frr*100:.2f}%")
    print(f"    → Porcentaje de personas genuinas rechazadas")
    
    # EER
    eer, eer_threshold = calculate_eer(similarities, labels)
    print(f"\n⚖️  EER (Equal Error Rate): {eer*100:.2f}%")
    print(f"    → Umbral óptimo: {eer_threshold:.4f}")
    
    # Evaluación de seguridad
    print("\n🎯 EVALUACIÓN DE SEGURIDAD:")
    if far < 0.02:
        print("  ✅ EXCELENTE: FAR < 2% (muy pocos impostores aceptados)")
    elif far < 0.05:
        print("  ✓ BUENO: FAR < 5% (aceptable para alta seguridad)")
    elif far < 0.10:
        print("  ⚠️  MODERADO: FAR < 10% (considerar aumentar umbral)")
    else:
        print("  ❌ INSEGURO: FAR > 10% (AUMENTAR UMBRAL URGENTEMENTE)")
    
    if frr < 0.10:
        print("  ✅ EXCELENTE: FRR < 10% (buena experiencia de usuario)")
    elif frr < 0.20:
        print("  ✓ ACEPTABLE: FRR < 20% (algunos usuarios legítimos rechazados)")
    else:
        print("  ⚠️  ALTO: FRR > 20% (muchos rechazos de usuarios legítimos)")
    
    print("\n" + "="*70)


def evaluate_threshold_range(similarities: List[float], 
                             labels: List[bool],
                             thresholds: List[float] = None):
    """
    Evalúa múltiples umbrales y muestra tabla comparativa.
    
    Args:
        similarities: Lista de valores de similitud coseno
        labels: Lista de booleanos (True = misma persona, False = diferente)
        thresholds: Lista de umbrales a evaluar (default: [0.5, 0.65, 0.75, 0.85])
    """
    if thresholds is None:
        thresholds = [0.5, 0.65, 0.75, 0.85]
    
    print("\n" + "="*70)
    print("COMPARACIÓN DE UMBRALES")
    print("="*70)
    print(f"{'Umbral':<10} {'FAR':<15} {'FRR':<15} {'Recomendación':<30}")
    print("-"*70)
    
    for threshold in thresholds:
        far, frr = calculate_far_frr(similarities, labels, threshold)
        
        if threshold == 0.5:
            rec = "Muy permisivo (INSEGURO)"
        elif threshold == 0.65:
            rec = "Permisivo"
        elif threshold == 0.75:
            rec = "Alta seguridad (RECOMENDADO)"
        elif threshold == 0.85:
            rec = "Ultra seguro"
        else:
            rec = ""
        
        print(f"{threshold:<10.2f} {far*100:<14.2f}% {frr*100:<14.2f}% {rec:<30}")
    
    print("="*70)
