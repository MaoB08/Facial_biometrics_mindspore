# Biometría facial con MindSpore (Huawei)

Modelo de **reconocimiento facial** con **MindSpore**, ejecutado **100% en local**: entrenamiento, validación e inferencia en local.

---

## Publicación para competencia

Este proyecto se publica con motivo de una **competencia**. El código y el modelo son de autoría propia; no se ha reutilizado código de terceros. Las dependencias (MindSpore, numpy, Pillow, scikit-learn, tqdm) se usan bajo sus respectivas licencias permisivas; véase la sección **Licencia y atribuciones** más abajo.

## Requisitos

- Python 3.8+
- MindSpore (CPU o GPU)

## Instalación

1. **Clonar / entrar al proyecto**

```bash
cd Facial_biometrics_mindspore
```

2. **Entorno virtual (recomendado)**

```bash
python3 -m venv venv
source venv/bin/activate   # Linux/macOS
# o: venv\Scripts\activate  # Windows
```

3. **Instalar MindSpore**

Elige según tu hardware:

- **Solo CPU:**  
  https://www.mindspore.cn/install  
  Ejemplo (Linux x86):  
  `pip install mindspore`

- **GPU (CUDA):**  
  `pip install mindspore-gpu`  
  (revisa versión de CUDA en la documentación oficial)

4. **Dependencias del proyecto**

```bash
pip install -r requirements.txt
```

## Estructura del proyecto

```
├── config.py           # Rutas e hiperparámetros
├── train.py            # Entrenamiento local
├── test.py             # Pruebas (validación y verificación 1:1)
├── data/
│   ├── train/          # Fotos por identidad (carpeta = persona)
│   └── val/            # Validación (misma estructura)
├── src/
│   ├── model.py        # Red de embeddings (CNN) + cabeza de clasificación
│   ├── dataset.py      # Carga de datos (ImageFolder)
│   └── inference.py    # Carga de modelo y verificación
├── scripts/
│   └── prepare_data.py # Crear datos y opcional LFW
└── checkpoints/        # Modelos guardados (se crea al entrenar)
```

## Uso (todo local)

### 1. Preparar datos

Crear carpetas y, opcionalmente, descargar un subset de ejemplo (LFW):

```bash
python scripts/prepare_data.py --download-lfw
```

O organizar tus propias fotos en `data/train/<identidad>/` y `data/val/<identidad>/` (ver `data/README.md`).

### 2. Entrenar

```bash
python train.py
```

El entrenamiento corre en tu máquina (CPU o GPU según hayas instalado MindSpore). Los checkpoints y la config se guardan en `checkpoints/`.

### 3. Probar

**Evaluar en validación:**

```bash
python test.py --eval
```

**Verificación 1:1 (¿son la misma persona?):**

```bash
python test.py --verify ruta/foto1.jpg ruta/foto2.jpg
```

Opcional: `--threshold 0.5` para cambiar el umbral de similitud.

---

## Ejemplo de entrenamiento (paso a paso)

A continuación, un ejemplo completo de cómo debe realizarse el entrenamiento en tu máquina.

### Paso 1: Entorno

```bash
cd Facial_biometrics_mindspore
source venv/bin/activate   # o: venv\Scripts\activate en Windows
```

### Paso 2: Datos

**Opción A — Dataset de ejemplo (LFW):**

```bash
python scripts/prepare_data.py --download-lfw
```

Esto crea `data/train` y `data/val` con varias identidades. Cada carpeta dentro de `train/` y `val/` es una persona; dentro van las fotos (.jpg, .png).

**Opción B — Tus propias fotos:**

- Crea `data/train/` y `data/val/`.
- Dentro, una carpeta por persona, por ejemplo:
  - `data/train/Maria/`  → varias fotos de María
  - `data/train/Juan/`   → varias fotos de Juan
  - (mínimo 2 personas para poder entrenar.)
- En `data/val/` la misma estructura (otras fotos de las mismas personas para validar).

Comprueba que hay al menos 2 identidades:

```bash
ls data/train
# Deberías ver varias carpetas (nombres de personas).
```

### Paso 3: Lanzar el entrenamiento

```bash
python train.py
```

Salida típica:

```
Identidades en entrenamiento: 15
Iniciando entrenamiento local...
epoch: 1 step: 50, loss is 2.8234
epoch: 1 step: 100, loss is 2.1021
...
epoch: 30 step: 50, loss is 0.2145
Config guardada en: checkpoints/model_config.json
Checkpoints guardados en: checkpoints/
```

- **loss** suele bajar con las épocas; si se estanca o sube, puedes bajar el learning rate en `config.py` o aumentar épocas.
- Al terminar, en `checkpoints/` tendrás archivos `face_biometrics-*.ckpt` y `model_config.json`.

### Paso 4: Validar y verificar

Evaluar en el conjunto de validación:

```bash
python test.py --eval
```

Ejemplo de salida: `Validación: 42/50 correctos, accuracy = 0.8400`.

Verificar si dos fotos son de la misma persona:

```bash
python test.py --verify data/val/persona_1/foto1.jpg data/val/persona_1/foto2.jpg
# Esperado: Misma persona: True, similitud alta

python test.py --verify data/val/persona_1/foto1.jpg data/val/persona_2/foto1.jpg
# Esperado: Misma persona: False, similitud baja
```

### Resumen del flujo

| Paso | Comando / acción |
|------|-------------------|
| 1 | `cd` al proyecto, activar `venv` |
| 2 | `python scripts/prepare_data.py --download-lfw` o montar tus carpetas en `data/train` y `data/val` |
| 3 | `python train.py` (esperar a que termine; revisar `loss`) |
| 4 | `python test.py --eval` y `python test.py --verify img1.jpg img2.jpg` |

## Configuración

En `config.py` puedes ajustar:

- `IMAGE_SIZE`: tamaño de entrada (por defecto 112×112)
- `EMBEDDING_DIM`: dimensión del vector de embedding (128)
- `BATCH_SIZE`, `EPOCHS`, `LEARNING_RATE`
- `VERIFICATION_THRESHOLD`: umbral para verificación 1:1

Para usar GPU en el entrenamiento, en `train.py` cambia:

```python
ms.set_context(..., device_target="GPU")
```

## Tecnología

- **MindSpore** (Huawei): cálculo y entrenamiento.
- **Modelo**: CNN ligera que produce vectores de 128 dimensiones (L2 normalizados); cabeza de clasificación para entrenar por identidad; en inferencia se usan solo los embeddings para verificación por similitud coseno.

## Licencia y atribuciones

- **Este proyecto**: el código de este repositorio (modelo, scripts, pipeline) es original y se distribuye para uso y evaluación en el marco de la competencia.
- **MindSpore** (Huawei): framework de deep learning usado bajo su licencia (Apache 2.0). Documentación e instalación: [mindspore.cn](https://www.mindspore.cn).
- **Otras dependencias**: numpy, Pillow, scikit-learn y tqdm se usan bajo sus licencias (BSD, PIL, BSD-3-Clause, etc.). Al publicar o redistribuir este proyecto, se deben respetar los términos de cada dependencia.
