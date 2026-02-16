# Sistema de Biometría Facial de Alta Seguridad

Sistema de reconocimiento facial implementado con MindSpore, optimizado para alta seguridad mediante arquitectura profunda, Triplet Loss y métricas de seguridad avanzadas.

## 🚀 Características Principales

- **Arquitectura Profunda**: 6 bloques convolucionales (~4M parámetros)
- **Alta Seguridad**: Umbrales estrictos (0.75/0.85) para minimizar falsos positivos
- **Triplet Loss**: Embeddings discriminativos para mejor separación entre identidades
- **Data Augmentation**: Robusto ante variaciones de iluminación, ángulos y calidad
- **Métricas de Seguridad**: FAR/FRR, EER, niveles de confianza
- **Learning Rate Scheduler**: Convergencia optimizada

## 📊 Resultados Esperados

| Métrica | Objetivo |
|---------|----------|
| **Accuracy** | 90-95% |
| **FAR** (False Acceptance) | < 2% |
| **FRR** (False Rejection) | 5-10% |
| **EER** (Equal Error Rate) | < 5% |

## 🎯 Uso Rápido

### Verificar Dataset
```bash
source venv/bin/activate
python scripts/check_dataset.py
```

### Entrenar
```bash
python train.py
```

### Evaluar Seguridad
```bash
python test.py --security-eval
```

### Verificar Par de Imágenes
```bash
python test.py --verify foto1.jpg foto2.jpg
```

## 📁 Estructura del Dataset

```
data/
├── train/                    # 80% de los datos
│   ├── persona_1/           # Nombre = etiqueta (automático)
│   │   ├── foto1.jpg
│   │   ├── foto2.jpg
│   │   └── ...              # 10-20 fotos recomendadas
│   └── persona_N/
└── val/                      # 20% de los datos
    ├── persona_1/           # Mismas personas que train
    │   └── ...              # 2-5 fotos diferentes
    └── persona_N/
```

**Nota**: El nombre de la carpeta es la etiqueta. Los nombres de archivos no importan.

## 🔧 Configuración

Edita `config.py` según tu dataset:

```python
# Dataset pequeño (2-10 personas, <100 fotos)
BATCH_SIZE = 4

# Dataset mediano (10-50 personas, 100-500 fotos)
BATCH_SIZE = 16

# Dataset grande (50+ personas, 500+ fotos)
BATCH_SIZE = 32

# Umbrales de seguridad
VERIFICATION_THRESHOLD = 0.75        # Alta seguridad (recomendado)
STRICT_VERIFICATION_THRESHOLD = 0.85 # Ultra-seguro
```

## 📚 Documentación

- **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** - Guía completa de entrenamiento
- **[DATASET_OPTIONS.md](DATASET_OPTIONS.md)** - Opciones de datasets públicos

## 🛠️ Instalación

```bash
# Crear entorno virtual
python3 -m venv venv
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

## 📦 Requisitos

- Python 3.8+
- MindSpore 2.0+
- NumPy
- Pillow

## 🎓 Recomendaciones

### Número de Personas
- **Mínimo**: 2 personas (funcional, overfitting)
- **Recomendado**: 20-30 personas (balance ideal)
- **Ideal**: 50+ personas (producción)

### Fotos por Persona
- **Mínimo**: 5 en train, 2 en val
- **Recomendado**: 10-15 en train, 3-5 en val
- **Ideal**: 20+ en train, 5+ en val

### Variedad de Fotos
- Diferentes ángulos (frontal, 45°, perfil)
- Diferentes expresiones (sonriendo, serio)
- Diferentes iluminaciones (natural, artificial)
- Con/sin accesorios (gafas, gorra)

## 🔒 Filosofía de Seguridad

> "Es mejor rechazar ocasionalmente a una persona legítima (que puede reintentar) que aceptar a un impostor."

El sistema prioriza seguridad mediante:
- Umbrales estrictos (0.75/0.85)
- Embeddings discriminativos (Triplet Loss)
- Arquitectura profunda (6 bloques + dropout)
- Data augmentation robusto
- Métricas transparentes (FAR/FRR)

## 📈 Niveles de Confianza

| Similitud | Decisión | Nivel |
|-----------|----------|-------|
| < 0.50 | ❌ RECHAZADO | Definitivamente NO es la persona |
| 0.50-0.75 | ⚠️ RECHAZADO | Dudoso (rechazado por seguridad) |
| 0.75-0.85 | ✓ ACEPTADO | Probable coincidencia |
| > 0.85 | ✅ ACEPTADO | Alta confianza |

## 🧪 Comandos de Prueba

```bash
# Verificar estructura del dataset
python scripts/check_dataset.py

# Entrenar modelo
python train.py

# Evaluar en validación
python test.py --eval

# Reporte de seguridad (FAR/FRR)
python test.py --security-eval

# Verificar par (misma persona - debe ACEPTAR)
python test.py --verify data/val/persona1/foto1.jpg \
                        data/val/persona1/foto2.jpg

# Verificar par (diferentes - debe RECHAZAR)
python test.py --verify data/val/persona1/foto1.jpg \
                        data/val/persona2/foto1.jpg

# Usar umbral personalizado
python test.py --verify foto1.jpg foto2.jpg --threshold 0.85
```

## 🎯 Mejoras de Seguridad (v2.0)

### Arquitectura
- ✨ 6 bloques convolucionales (antes: 4)
- ✨ ~4M parámetros (antes: ~2M, +100%)
- ✨ Dropout (0.5) para regularización
- ✨ Filtros: 32→64→128→256→512→512

### Entrenamiento
- ✨ 100 épocas (antes: 30)
- ✨ Learning rate: 5e-4 (antes: 1e-3)
- ✨ LR scheduler (reduce cada 30 épocas)
- ✨ Triplet Loss + CrossEntropy

### Data Augmentation
- ✨ Rotación aleatoria (±15°)
- ✨ Brillo/contraste (±20%)
- ✨ Gaussian blur (30% prob)
- ✨ Recorte aleatorio (80-100%)
- ✨ Flip horizontal (50%)

### Seguridad
- ✨ Umbral: 0.75 (antes: 0.5, +50%)
- ✨ Modo ultra-seguro: 0.85
- ✨ Métricas FAR/FRR
- ✨ Niveles de confianza
- ✨ Reportes detallados

## 📝 Archivos Principales

```
Facial_biometrics_mindspore/
├── config.py                 # Configuración del sistema
├── train.py                  # Script de entrenamiento
├── test.py                   # Evaluación y verificación
├── src/
│   ├── model.py             # Arquitectura de la red
│   ├── dataset.py           # Carga y augmentation
│   ├── losses.py            # Triplet Loss
│   ├── inference.py         # Verificación 1:1
│   └── security_metrics.py  # Métricas FAR/FRR
├── scripts/
│   └── check_dataset.py     # Verificar dataset
├── data/
│   ├── train/               # Datos de entrenamiento
│   └── val/                 # Datos de validación
└── checkpoints/             # Modelos entrenados (git-ignored)
```

## 🤝 Contribuciones

Sistema desarrollado con MindSpore para biometría facial de alta seguridad.

## 📄 Licencia

MIT License

---

**Versión**: 2.0 (High-Security Update)  
**Framework**: MindSpore 2.0+  
**Objetivo**: Sistema de verificación facial con FAR < 2%
