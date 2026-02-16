# 📚 Guía Completa de Entrenamiento - Sistema de Biometría Facial

## 🎯 Estructura de Datos Requerida

### Organización de Carpetas

El sistema usa **etiquetado automático por carpetas**. Cada carpeta representa una persona (identidad):

```
data/
├── train/                    # Datos de entrenamiento (80%)
│   ├── persona_1/           # Nombre de la persona (etiqueta automática)
│   │   ├── foto_1.jpg
│   │   ├── foto_2.jpg
│   │   ├── foto_3.jpg
│   │   └── ...              # Mínimo 5-10 fotos
│   ├── persona_2/
│   │   ├── foto_1.jpg
│   │   └── ...
│   └── persona_N/
│       └── ...
│
└── val/                      # Datos de validación (20%)
    ├── persona_1/           # Mismas personas que en train
    │   ├── foto_val_1.jpg
    │   └── foto_val_2.jpg   # Mínimo 2-3 fotos diferentes
    ├── persona_2/
    │   └── ...
    └── persona_N/
        └── ...
```

### ✅ Reglas Importantes

1. **Nombre de carpeta = Etiqueta**
   - El nombre de la carpeta es la identidad de la persona
   - Usa nombres descriptivos: `juan_perez`, `maria_garcia`, `bill_gates`
   - Evita espacios, usa guiones bajos: `john_doe` ✓, `john doe` ✗

2. **Fotos por persona**
   - **Mínimo:** 5 fotos en train, 2 en val
   - **Recomendado:** 10-20 fotos en train, 3-5 en val
   - **Ideal:** 20+ fotos en train, 5+ en val

3. **Variedad en las fotos**
   - Diferentes ángulos (frontal, 45°, perfil)
   - Diferentes expresiones (sonriendo, serio, hablando)
   - Diferentes iluminaciones (natural, artificial, sombra)
   - Con/sin accesorios (gafas, gorra, barba)
   - Diferentes fondos

4. **Calidad de las fotos**
   - Resolución mínima: 112x112 píxeles (se redimensionarán automáticamente)
   - Formato: JPG, JPEG, PNG, BMP
   - El rostro debe ser visible y ocupar buena parte de la imagen
   - Evita fotos muy borrosas o con mala iluminación extrema

---

## 📸 Recomendaciones para Capturar Fotos

### Para Alta Seguridad (Recomendado)

Captura **10-15 fotos por persona** con esta variedad:

| Tipo | Cantidad | Descripción |
|------|----------|-------------|
| **Frontal** | 3-4 | Mirando directo a la cámara, diferentes expresiones |
| **Ángulo 45°** | 2-3 | Girado ligeramente a izquierda/derecha |
| **Iluminación variada** | 2-3 | Luz natural, artificial, sombra parcial |
| **Con accesorios** | 2-3 | Gafas, gorra, barba (si aplica) |
| **Diferentes fondos** | 2-3 | Interior, exterior, diferentes colores |

### División Train/Val

**Regla 80/20:**
- Si tienes 15 fotos → 12 en `train/`, 3 en `val/`
- Si tienes 10 fotos → 8 en `train/`, 2 en `val/`
- Si tienes 20 fotos → 16 en `train/`, 4 en `val/`

**Importante:** Las fotos de `val/` deben ser **diferentes** a las de `train/`

---

## 🔢 Número de Personas (Identidades)

### Mínimo Funcional
- **2 personas** (tu dataset actual)
- Sirve para probar el sistema
- Tendrá overfitting pero funciona

### Recomendado para Producción
- **10-50 personas**
- Balance entre tiempo de entrenamiento y generalización
- Buen punto de partida para aplicaciones reales

### Ideal
- **100+ personas**
- Mejor generalización
- Sistema más robusto
- Requiere más tiempo de entrenamiento

---

## 🚀 Proceso de Entrenamiento Paso a Paso

### Paso 1: Preparar tus Datos

**Opción A: Agregar más personas manualmente**

```bash
# Crear carpetas para nuevas personas
mkdir -p data/train/persona_3
mkdir -p data/train/persona_4
mkdir -p data/val/persona_3
mkdir -p data/val/persona_4

# Copiar fotos a las carpetas correspondientes
# (usa tu explorador de archivos o comandos cp)
```

**Opción B: Usar dataset público (LFW)**

```bash
# 1. Descargar LFW desde: http://vis-www.cs.umass.edu/lfw/lfw.tgz
# 2. Extraer en data/
tar -xzf ~/Descargas/lfw.tgz -C data/

# 3. Organizar automáticamente
source venv/bin/activate
python scripts/prepare_data.py --organize-lfw
```

### Paso 2: Verificar la Estructura

```bash
# Ver cuántas personas tienes
ls data/train/ | wc -l

# Ver cuántas fotos tiene cada persona
for dir in data/train/*/; do 
    echo "$(basename "$dir"): $(ls "$dir" | wc -l) fotos"
done
```

**Salida esperada:**
```
bill_gates: 4 fotos
elon_musk: 4 fotos
persona_3: 8 fotos
...
```

### Paso 3: Ajustar Configuración

Edita `config.py` según tu dataset:

```python
# Para dataset pequeño (2-10 personas, <100 fotos)
BATCH_SIZE = 4
EPOCHS = 100

# Para dataset mediano (10-50 personas, 100-500 fotos)
BATCH_SIZE = 16
EPOCHS = 100

# Para dataset grande (50+ personas, 500+ fotos)
BATCH_SIZE = 32
EPOCHS = 100
```

### Paso 4: Entrenar el Modelo

```bash
# Activar entorno virtual
source venv/bin/activate

# Iniciar entrenamiento
python train.py
```

**Salida esperada:**
```
Identidades en entrenamiento: 10
Configuración:
  - Épocas: 100
  - Learning Rate: 0.0005
  - Triplet Margin: 0.3
  - Arquitectura: 6 bloques convolucionales + Dropout
  - Loss: Triplet Loss (peso=1.0) + CrossEntropy (peso=0.5)

🚀 Iniciando entrenamiento con arquitectura mejorada...
   Batch size: 4
   Steps por época: 25
   Total steps: 2500

epoch: 1 step: 50, loss is 2.3456
epoch: 2 step: 100, loss is 2.1234
...
```

### Paso 5: Monitorear el Entrenamiento

**Señales de buen entrenamiento:**
- ✅ Loss disminuye progresivamente
- ✅ Accuracy en validación aumenta
- ✅ No hay errores de memoria

**Señales de problemas:**
- ❌ Loss no disminuye (learning rate muy bajo)
- ❌ Loss oscila mucho (learning rate muy alto)
- ❌ Accuracy en validación no mejora (overfitting)

**Tiempos estimados:**

| Dataset | Hardware | Tiempo Estimado |
|---------|----------|-----------------|
| 2-5 personas, 50 fotos | CPU | 30-60 min |
| 10-20 personas, 200 fotos | CPU | 1-2 horas |
| 50+ personas, 500+ fotos | CPU | 3-6 horas |
| 50+ personas, 500+ fotos | GPU | 30-60 min |

---

## 🧪 Validación Post-Entrenamiento

### 1. Evaluación Automática

```bash
# Evaluar en dataset de validación
python test.py --eval
```

**Salida esperada:**
```
Validación: 45/50 correctos, accuracy = 0.9000
```

### 2. Evaluación de Seguridad

```bash
# Generar reporte de seguridad (FAR/FRR)
python test.py --security-eval
```

**Salida esperada:**
```
======================================================================
REPORTE DE SEGURIDAD - SISTEMA DE BIOMETRÍA FACIAL
======================================================================

📊 DISTRIBUCIÓN DE SIMILITUDES:
  Pares Genuinos (misma persona): 45
    - Media: 0.8234 ± 0.0456
    - Rango: [0.7123, 0.9456]

  Pares Impostores (diferente persona): 180
    - Media: 0.3456 ± 0.1234
    - Rango: [0.1234, 0.6789]

🔒 MÉTRICAS DE SEGURIDAD (Umbral = 0.75):
  FAR (False Acceptance Rate): 1.67%
    → Porcentaje de impostores aceptados (CRÍTICO)
  FRR (False Rejection Rate): 8.89%
    → Porcentaje de personas genuinas rechazadas

⚖️  EER (Equal Error Rate): 5.23%
    → Umbral óptimo: 0.7234

🎯 EVALUACIÓN DE SEGURIDAD:
  ✅ EXCELENTE: FAR < 2% (muy pocos impostores aceptados)
  ✅ EXCELENTE: FRR < 10% (buena experiencia de usuario)
======================================================================
```

### 3. Pruebas Manuales

**Probar con misma persona (debe ACEPTAR):**
```bash
python test.py --verify data/val/bill_gates/bill_gates_val.jpg \
                        data/val/bill_gates/bill_gates_val_2.jpg
```

**Salida esperada:**
```
======================================================================
VERIFICACIÓN 1:1
======================================================================
Imagen 1: data/val/bill_gates/bill_gates_val.jpg
Imagen 2: data/val/bill_gates/bill_gates_val_2.jpg

Similitud: 0.8567
Umbral: 0.75

Resultado: ACEPTADO - Alta confianza
======================================================================
```

**Probar con diferentes personas (debe RECHAZAR):**
```bash
python test.py --verify data/val/bill_gates/bill_gates_val.jpg \
                        data/val/elon_musk/elon-musk-automotive-congress.jpg
```

**Salida esperada:**
```
Similitud: 0.3456
Resultado: RECHAZADO - Definitivamente NO es la persona
```

---

## 📊 Interpretación de Resultados

### Métricas Objetivo

| Métrica | Objetivo | Excelente | Bueno | Mejorar |
|---------|----------|-----------|-------|---------|
| **Accuracy** | > 90% | > 95% | 85-95% | < 85% |
| **FAR** | < 2% | < 1% | 1-3% | > 3% |
| **FRR** | < 10% | < 5% | 5-15% | > 15% |
| **EER** | < 5% | < 3% | 3-7% | > 7% |

### Si los Resultados No Son Buenos

**FAR muy alto (> 5%):**
- ✅ Aumentar umbral a 0.85
- ✅ Entrenar más épocas
- ✅ Agregar más fotos de diferentes personas

**FRR muy alto (> 20%):**
- ✅ Reducir umbral a 0.70
- ✅ Agregar más variedad de fotos de cada persona
- ✅ Mejorar calidad de las fotos

**Accuracy baja (< 80%):**
- ✅ Aumentar número de épocas
- ✅ Agregar más datos de entrenamiento
- ✅ Verificar calidad de las fotos

---

## 🎓 Mejores Prácticas

### ✅ DO (Hacer)

1. **Usa fotos de calidad**
   - Buena iluminación
   - Rostro visible y centrado
   - Resolución adecuada

2. **Varía las condiciones**
   - Diferentes ángulos
   - Diferentes expresiones
   - Diferentes iluminaciones

3. **Balancea el dataset**
   - Número similar de fotos por persona
   - División 80/20 train/val

4. **Monitorea el entrenamiento**
   - Revisa que el loss disminuya
   - Valida periódicamente

5. **Prueba exhaustivamente**
   - Usa `--security-eval`
   - Prueba con fotos nuevas
   - Verifica casos extremos

### ❌ DON'T (No Hacer)

1. **No uses fotos muy similares**
   - Evita duplicados
   - Evita fotos consecutivas de video

2. **No mezcles train y val**
   - Las fotos de validación deben ser únicas
   - No reutilices fotos entre conjuntos

3. **No uses fotos de mala calidad**
   - Evita fotos muy borrosas
   - Evita fotos con rostro muy pequeño
   - Evita fotos con oclusiones extremas

4. **No entrenes con muy pocas fotos**
   - Mínimo 5 fotos por persona
   - Mínimo 2 personas

5. **No ignores las métricas**
   - Siempre ejecuta `--security-eval`
   - Monitorea FAR especialmente

---

## 🔧 Solución de Problemas Comunes

### Error: "Se necesitan al menos 2 identidades"

**Causa:** No hay suficientes carpetas en `data/train/`

**Solución:**
```bash
# Verificar estructura
ls data/train/

# Debe haber al menos 2 carpetas
```

### Error: "No se encontró checkpoint"

**Causa:** No has entrenado el modelo aún

**Solución:**
```bash
python train.py
```

### Warning: "Validación tiene X clases, modelo Y"

**Causa:** Diferentes personas en train y val

**Solución:** Asegúrate de que las mismas personas estén en ambos conjuntos

### Loss no disminuye

**Causa:** Learning rate muy bajo o dataset muy pequeño

**Solución:**
```python
# En config.py, prueba:
LEARNING_RATE = 1e-3  # Aumentar si es muy lento
# O agregar más datos
```

---

## 📝 Checklist de Entrenamiento

Antes de entrenar, verifica:

- [ ] Tengo al menos 2 personas en `data/train/`
- [ ] Cada persona tiene al menos 5 fotos en train
- [ ] Cada persona tiene al menos 2 fotos en val
- [ ] Las fotos de val son diferentes a las de train
- [ ] Las fotos tienen buena calidad (rostro visible)
- [ ] He ajustado `BATCH_SIZE` según mi dataset
- [ ] He activado el entorno virtual (`source venv/bin/activate`)

Durante el entrenamiento:

- [ ] El loss está disminuyendo
- [ ] No hay errores de memoria
- [ ] El tiempo estimado es razonable

Después del entrenamiento:

- [ ] He ejecutado `python test.py --eval`
- [ ] He ejecutado `python test.py --security-eval`
- [ ] FAR < 5% (idealmente < 2%)
- [ ] FRR < 20% (idealmente < 10%)
- [ ] He probado verificación manual con `--verify`

---

## 🎯 Ejemplo Completo

```bash
# 1. Preparar datos (ejemplo con 5 personas)
mkdir -p data/train/{persona_1,persona_2,persona_3,persona_4,persona_5}
mkdir -p data/val/{persona_1,persona_2,persona_3,persona_4,persona_5}

# 2. Copiar fotos a cada carpeta
# (usa tu explorador de archivos)

# 3. Verificar estructura
ls data/train/*/  # Debe mostrar fotos en cada carpeta

# 4. Activar entorno
source venv/bin/activate

# 5. Entrenar
python train.py

# 6. Evaluar
python test.py --eval
python test.py --security-eval

# 7. Probar verificación
python test.py --verify data/val/persona_1/foto1.jpg \
                        data/val/persona_1/foto2.jpg
```

---

## 📚 Recursos Adicionales

### Datasets Públicos Recomendados

1. **LFW (Labeled Faces in the Wild)**
   - URL: http://vis-www.cs.umass.edu/lfw/lfw.tgz
   - Tamaño: 173 MB
   - Personas: 5,749
   - Imágenes: 13,233

2. **CelebA**
   - URL: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
   - Tamaño: 1.4 GB
   - Personas: 10,177
   - Imágenes: 202,599

### Comandos Útiles

```bash
# Contar personas
ls data/train/ | wc -l

# Contar fotos totales
find data/train/ -type f | wc -l

# Ver distribución de fotos por persona
for dir in data/train/*/; do 
    echo "$(basename "$dir"): $(ls "$dir" | wc -l)"
done

# Limpiar checkpoints antiguos
rm checkpoints/*.ckpt

# Ver último checkpoint
ls -lt checkpoints/*.ckpt | head -1
```

---

## ✨ Resumen Rápido

1. **Estructura:** `data/train/persona_X/fotos.jpg` (nombre de carpeta = etiqueta)
2. **Mínimo:** 2 personas, 5 fotos/persona en train, 2 en val
3. **Recomendado:** 10+ personas, 10+ fotos/persona
4. **Entrenar:** `python train.py`
5. **Evaluar:** `python test.py --security-eval`
6. **Objetivo:** FAR < 2%, FRR < 10%, Accuracy > 90%

¡Listo para entrenar! 🚀
