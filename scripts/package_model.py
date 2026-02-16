#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para empaquetar el modelo entrenado para transferencia a otro computador.
Crea un archivo .tar.gz con todo lo necesario para usar el modelo.
"""

import os
import sys
import tarfile
from datetime import datetime

# Añadir raíz del proyecto
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BASE_DIR, CHECKPOINT_DIR

def package_model(output_name=None):
    """
    Empaqueta el modelo y archivos necesarios en un archivo tar.gz
    
    Args:
        output_name: Nombre del archivo de salida (opcional)
    """
    if output_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"modelo_biometria_facial_{timestamp}.tar.gz"
    
    output_path = os.path.join(BASE_DIR, output_name)
    
    # Verificar que existan checkpoints
    if not os.path.exists(CHECKPOINT_DIR):
        print(f"❌ Error: No se encontró la carpeta {CHECKPOINT_DIR}")
        print("   Debes entrenar el modelo primero con: python3 train.py")
        return False
    
    ckpts = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.ckpt')]
    if not ckpts:
        print(f"❌ Error: No hay archivos .ckpt en {CHECKPOINT_DIR}")
        print("   Debes entrenar el modelo primero con: python3 train.py")
        return False
    
    # Archivos y carpetas a incluir
    files_to_include = [
        'checkpoints/',
        'src/',
        'config.py',
        'test.py',
        'requirements.txt',
        'README.md',
    ]
    
    print("=" * 60)
    print("📦 Empaquetando modelo para transferencia")
    print("=" * 60)
    
    try:
        with tarfile.open(output_path, "w:gz") as tar:
            for item in files_to_include:
                item_path = os.path.join(BASE_DIR, item)
                if os.path.exists(item_path):
                    arcname = os.path.join("Facial_biometrics_mindspore", item)
                    tar.add(item_path, arcname=arcname)
                    print(f"✅ Agregado: {item}")
                else:
                    print(f"⚠️  No encontrado (omitido): {item}")
        
        # Obtener tamaño del archivo
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        
        print("\n" + "=" * 60)
        print(f"✅ Paquete creado exitosamente")
        print("=" * 60)
        print(f"📁 Archivo: {output_path}")
        print(f"📊 Tamaño: {size_mb:.2f} MB")
        print("\n📝 Instrucciones para el computador de destino:")
        print("-" * 60)
        print("1. Transferir el archivo (USB, email, Drive, etc.)")
        print(f"2. Extraer: tar -xzf {output_name}")
        print("3. cd Facial_biometrics_mindspore")
        print("4. python3 -m venv venv")
        print("5. source venv/bin/activate  # Linux/Mac")
        print("   o: venv\\Scripts\\activate  # Windows")
        print("6. pip install -r requirements.txt")
        print("7. python3 test.py --verify foto1.jpg foto2.jpg")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error al crear el paquete: {e}")
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Empaquetar modelo para transferencia a otro computador"
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Nombre del archivo de salida (default: modelo_biometria_facial_TIMESTAMP.tar.gz)'
    )
    
    args = parser.parse_args()
    package_model(args.output)


if __name__ == "__main__":
    main()
