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
from config import CHECKPOINT_DIR, VAL_DIR, VERIFICATION_THRESHOLD
from src.dataset import create_val_dataset, get_num_classes_from_dir
from src.inference import load_embedding_model, verify_pair


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
    """Verificación 1:1 entre dos imágenes."""
    threshold = threshold or VERIFICATION_THRESHOLD
    model = load_embedding_model()
    misma, sim = verify_pair(model, pair1, pair2, threshold=threshold)
    print(f"Similitud: {sim:.4f} | Misma persona: {misma}")
    return misma, sim


def main():
    parser = argparse.ArgumentParser(description="Pruebas del modelo de biometría facial")
    parser.add_argument("--eval", action="store_true", help="Evaluar en dataset de validación")
    parser.add_argument("--verify", nargs=2, metavar=("IMG1", "IMG2"), help="Verificar par de imágenes")
    parser.add_argument("--threshold", type=float, default=None, help="Umbral de verificación (default: config)")
    args = parser.parse_args()

    if args.eval:
        eval_validation()
    elif args.verify:
        run_verification(args.verify[0], args.verify[1], threshold=args.threshold)
    else:
        parser.print_help()
        print("\nEjemplos:")
        print("  python test.py --eval")
        print("  python test.py --verify foto1.jpg foto2.jpg")


if __name__ == "__main__":
    main()
