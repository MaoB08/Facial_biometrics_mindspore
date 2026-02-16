#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de ayuda para verificar la estructura del dataset antes de entrenar.
Muestra estadísticas y detecta problemas comunes.
"""

import os
import sys
from collections import defaultdict

# Colores para terminal
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def check_dataset_structure(data_dir="data"):
    """Verifica y muestra la estructura del dataset."""
    
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    
    print(f"\n{Colors.BOLD}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}📊 VERIFICACIÓN DE DATASET - SISTEMA DE BIOMETRÍA FACIAL{Colors.END}")
    print(f"{Colors.BOLD}{'='*70}{Colors.END}\n")
    
    # Verificar existencia de directorios
    if not os.path.exists(train_dir):
        print(f"{Colors.RED}❌ ERROR: No existe el directorio {train_dir}{Colors.END}")
        return False
    
    if not os.path.exists(val_dir):
        print(f"{Colors.YELLOW}⚠️  ADVERTENCIA: No existe el directorio {val_dir}{Colors.END}")
        print(f"   Se creará automáticamente, pero es recomendado tener datos de validación.\n")
    
    # Analizar train
    train_stats = analyze_directory(train_dir, "ENTRENAMIENTO")
    
    # Analizar val si existe
    val_stats = None
    if os.path.exists(val_dir):
        val_stats = analyze_directory(val_dir, "VALIDACIÓN")
    
    # Verificar consistencia
    print(f"\n{Colors.BOLD}🔍 VERIFICACIÓN DE CONSISTENCIA{Colors.END}")
    print(f"{Colors.BOLD}{'-'*70}{Colors.END}")
    
    issues = []
    warnings = []
    
    # Verificar número mínimo de personas
    if train_stats['num_identities'] < 2:
        issues.append("Se necesitan al menos 2 personas en train/")
    elif train_stats['num_identities'] < 10:
        warnings.append(f"Solo {train_stats['num_identities']} personas. Recomendado: 10+")
    
    # Verificar fotos por persona
    if train_stats['min_images'] < 5:
        issues.append(f"Alguna persona tiene menos de 5 fotos (mínimo: {train_stats['min_images']})")
    
    # Verificar validación
    if val_stats:
        if val_stats['num_identities'] != train_stats['num_identities']:
            warnings.append("Diferentes personas en train y val")
        
        if val_stats['min_images'] < 2:
            warnings.append("Alguna persona tiene menos de 2 fotos en val")
    
    # Mostrar resultados
    if not issues and not warnings:
        print(f"{Colors.GREEN}✅ Todo correcto! El dataset está listo para entrenar.{Colors.END}\n")
    else:
        if issues:
            print(f"\n{Colors.RED}❌ PROBLEMAS CRÍTICOS:{Colors.END}")
            for issue in issues:
                print(f"   • {issue}")
        
        if warnings:
            print(f"\n{Colors.YELLOW}⚠️  ADVERTENCIAS:{Colors.END}")
            for warning in warnings:
                print(f"   • {warning}")
        print()
    
    # Recomendaciones
    print(f"{Colors.BOLD}💡 RECOMENDACIONES{Colors.END}")
    print(f"{Colors.BOLD}{'-'*70}{Colors.END}")
    
    if train_stats['num_identities'] < 10:
        print(f"📸 Agregar más personas para mejor generalización")
        print(f"   Actual: {train_stats['num_identities']} personas → Recomendado: 10-50 personas\n")
    
    if train_stats['avg_images'] < 10:
        print(f"📷 Agregar más fotos por persona")
        print(f"   Promedio actual: {train_stats['avg_images']:.1f} fotos → Recomendado: 10-20 fotos\n")
    
    # Configuración recomendada
    print(f"{Colors.BOLD}⚙️  CONFIGURACIÓN RECOMENDADA{Colors.END}")
    print(f"{Colors.BOLD}{'-'*70}{Colors.END}")
    
    total_images = train_stats['total_images']
    if total_images < 100:
        batch_size = 4
        dataset_size = "pequeño"
    elif total_images < 500:
        batch_size = 16
        dataset_size = "mediano"
    else:
        batch_size = 32
        dataset_size = "grande"
    
    print(f"Dataset: {Colors.BLUE}{dataset_size}{Colors.END}")
    print(f"BATCH_SIZE recomendado: {Colors.BLUE}{batch_size}{Colors.END}")
    print(f"EPOCHS: {Colors.BLUE}100{Colors.END}")
    print(f"\nEdita {Colors.BLUE}config.py{Colors.END} si es necesario.\n")
    
    # Comando para entrenar
    print(f"{Colors.BOLD}🚀 SIGUIENTE PASO{Colors.END}")
    print(f"{Colors.BOLD}{'-'*70}{Colors.END}")
    if not issues:
        print(f"{Colors.GREEN}source venv/bin/activate && python train.py{Colors.END}\n")
    else:
        print(f"{Colors.RED}Corrige los problemas críticos antes de entrenar.{Colors.END}\n")
    
    print(f"{Colors.BOLD}{'='*70}{Colors.END}\n")
    
    return len(issues) == 0


def analyze_directory(directory, label):
    """Analiza un directorio de datos y retorna estadísticas."""
    
    print(f"{Colors.BOLD}📁 {label}{Colors.END}")
    print(f"{Colors.BOLD}{'-'*70}{Colors.END}")
    
    identities = {}
    total_images = 0
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    # Recopilar información
    for identity in os.listdir(directory):
        identity_path = os.path.join(directory, identity)
        
        if not os.path.isdir(identity_path):
            continue
        
        images = []
        for file in os.listdir(identity_path):
            file_lower = file.lower()
            if any(file_lower.endswith(ext) for ext in valid_extensions):
                images.append(file)
        
        if images:
            identities[identity] = images
            total_images += len(images)
    
    num_identities = len(identities)
    
    if num_identities == 0:
        print(f"{Colors.RED}❌ No se encontraron identidades{Colors.END}\n")
        return {
            'num_identities': 0,
            'total_images': 0,
            'min_images': 0,
            'max_images': 0,
            'avg_images': 0
        }
    
    # Calcular estadísticas
    image_counts = [len(imgs) for imgs in identities.values()]
    min_images = min(image_counts)
    max_images = max(image_counts)
    avg_images = total_images / num_identities
    
    # Mostrar resumen
    print(f"Identidades (personas): {Colors.BLUE}{num_identities}{Colors.END}")
    print(f"Total de imágenes: {Colors.BLUE}{total_images}{Colors.END}")
    print(f"Imágenes por persona:")
    print(f"  • Mínimo: {Colors.BLUE}{min_images}{Colors.END}")
    print(f"  • Máximo: {Colors.BLUE}{max_images}{Colors.END}")
    print(f"  • Promedio: {Colors.BLUE}{avg_images:.1f}{Colors.END}")
    
    # Mostrar detalle por identidad
    print(f"\nDetalle por identidad:")
    for identity, images in sorted(identities.items()):
        count = len(images)
        
        # Color según cantidad
        if count < 5:
            color = Colors.RED
            status = "❌"
        elif count < 10:
            color = Colors.YELLOW
            status = "⚠️ "
        else:
            color = Colors.GREEN
            status = "✅"
        
        print(f"  {status} {identity:20s} {color}{count:3d} fotos{Colors.END}")
    
    print()
    
    return {
        'num_identities': num_identities,
        'total_images': total_images,
        'min_images': min_images,
        'max_images': max_images,
        'avg_images': avg_images,
        'identities': identities
    }


if __name__ == "__main__":
    # Cambiar al directorio del proyecto
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    os.chdir(project_dir)
    
    # Verificar dataset
    success = check_dataset_structure()
    
    # Exit code
    sys.exit(0 if success else 1)
