# -*- coding: utf-8 -*-
"""
Pruebas locales del modelo de biometría facial.
- Evaluación en validación (accuracy por identidad).
- Verificación 1:1 con pares de imágenes (opcional).
"""

import argparse
import os
import sys

import mindspore as ms
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CHECKPOINT_DIR, VAL_DIR, VERIFICATION_THRESHOLD, STRICT_VERIFICATION_THRESHOLD
from src.dataset import create_val_dataset, get_num_classes_from_dir
from src.inference import load_embedding_model, verify_pair
from src.security_metrics import (
    print_security_report,
    evaluate_threshold_range,
    calculate_far_frr,
)


def eval_validation():
    """Evalúa el modelo en el dataset de validación (clasificación)."""
    from src.model import FaceBiometricsNet
    from config import EMBEDDING_DIM
    import json

    if not os.path.isdir(VAL_DIR):
        print(f"No existe directorio de validación: {VAL_DIR}. Omisión de eval.")
        return

    config_path = os.path.join(CHECKPOINT_DIR, "model_config.json")
    if not os.path.isfile(config_path):
        print("No hay modelo entrenado. Ejecuta primero: python train.py")
        return

    with open(config_path) as f:
        config = json.load(f)
    num_classes = config["num_classes"]
    val_num = get_num_classes_from_dir(VAL_DIR)
    if val_num != num_classes:
        print(f"Aviso: validación tiene {val_num} clases, modelo {num_classes}. Usando num_classes del modelo.")

    val_ds = create_val_dataset(batch_size=1)
    net = FaceBiometricsNet(embedding_dim=EMBEDDING_DIM, num_classes=num_classes)
    ckpts = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".ckpt") and "face_biometrics" in f]
    if not ckpts:
        print("No se encontró checkpoint.")
        return
    ms.load_param_into_net(net, ms.load_checkpoint(os.path.join(CHECKPOINT_DIR, sorted(ckpts)[-1])))
    net.set_train(False)

    correct, total = 0, 0
    for batch in val_ds.create_dict_iterator():
        logits = net(batch["image"])
        pred = np.argmax(logits.asnumpy(), axis=1)
        label = batch["label"].asnumpy().flatten()
        correct += (pred == label).sum()
        total += label.size
    acc = correct / total if total else 0
    print(f"Validación: {correct}/{total} correctos, accuracy = {acc:.4f}")


def run_verification(pair1, pair2, threshold=None):
    """Verificación 1:1 entre dos imágenes con nivel de confianza."""
    threshold = threshold or VERIFICATION_THRESHOLD
    model = load_embedding_model()
    misma, sim, confianza = verify_pair(model, pair1, pair2, threshold=threshold)
    print(f"\n{'='*70}")
    print(f"VERIFICACIÓN 1:1")
    print(f"{'='*70}")
    print(f"Imagen 1: {pair1}")
    print(f"Imagen 2: {pair2}")
    print(f"\nSimilitud: {sim:.4f}")
    print(f"Umbral: {threshold:.2f}")
    print(f"\nResultado: {confianza}")
    print(f"{'='*70}\n")
    return misma, sim


def run_security_evaluation():
    """Evalúa métricas de seguridad (FAR/FRR) en dataset de validación."""
    if not os.path.isdir(VAL_DIR):
        print(f"No existe directorio de validación: {VAL_DIR}")
        return
    
    print("\n🔍 Cargando modelo y generando pares de verificación...")
    model = load_embedding_model()
    
    # Recopilar todas las imágenes por identidad
    from collections import defaultdict
    identity_images = defaultdict(list)
    
    for identity in os.listdir(VAL_DIR):
        identity_path = os.path.join(VAL_DIR, identity)
        if os.path.isdir(identity_path):
            for img_file in os.listdir(identity_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    identity_images[identity].append(os.path.join(identity_path, img_file))
    
    # Generar pares genuinos e impostores
    similarities = []
    labels = []  # True = misma persona, False = diferente
    
    identities = list(identity_images.keys())
    print(f"Identidades encontradas: {len(identities)}")
    
    # Pares genuinos (misma persona)
    for identity, images in identity_images.items():
        if len(images) >= 2:
            for i in range(len(images)):
                for j in range(i+1, min(i+3, len(images))):  # Limitar pares
                    _, sim, _ = verify_pair(model, images[i], images[j])
                    similarities.append(sim)
                    labels.append(True)
    
    # Pares impostores (diferentes personas)
    for i, id1 in enumerate(identities):
        for id2 in identities[i+1:i+4]:  # Limitar comparaciones
            if identity_images[id1] and identity_images[id2]:
                _, sim, _ = verify_pair(model, 
                                       identity_images[id1][0], 
                                       identity_images[id2][0])
                similarities.append(sim)
                labels.append(False)
    
    print(f"\nPares evaluados: {len(similarities)}")
    print(f"  - Genuinos: {sum(labels)}")
    print(f"  - Impostores: {len(labels) - sum(labels)}")
    
    # Generar reporte de seguridad
    print_security_report(similarities, labels, threshold=VERIFICATION_THRESHOLD)
    
    # Comparar múltiples umbrales
    evaluate_threshold_range(similarities, labels, 
                            thresholds=[0.5, 0.65, 0.75, 0.85])


def main():
    parser = argparse.ArgumentParser(description="Pruebas del modelo de biometría facial")
    parser.add_argument("--eval", action="store_true", help="Evaluar en dataset de validación")
    parser.add_argument("--verify", nargs=2, metavar=("IMG1", "IMG2"), help="Verificar par de imágenes")
    parser.add_argument("--security-eval", action="store_true", help="Evaluar métricas de seguridad (FAR/FRR)")
    parser.add_argument("--threshold", type=float, default=None, help="Umbral de verificación (default: config)")
    args = parser.parse_args()

    if args.eval:
        eval_validation()
    elif args.verify:
        run_verification(args.verify[0], args.verify[1], threshold=args.threshold)
    elif args.security_eval:
        run_security_evaluation()
    else:
        parser.print_help()
        print("\nEjemplos:")
        print("  python test.py --eval")
        print("  python test.py --verify foto1.jpg foto2.jpg")
        print("  python test.py --security-eval")


if __name__ == "__main__":
    main()
