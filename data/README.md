# Datos para biometría facial (uso local)

## Estructura

```
data/
├── train/           # Entrenamiento: una carpeta por identidad
│   ├── persona_1/
│   │   ├── foto1.jpg
│   │   └── foto2.jpg
│   └── persona_2/
│       └── ...
└── val/             # Validación (opcional): misma estructura
    ├── persona_1/
    └── persona_2/
```

- Cada **subcarpeta** es una identidad (una persona).
- Dentro de cada carpeta, todas las **imágenes** (.jpg, .png, etc.) son de esa persona.
- Se necesitan **al menos 2 identidades** para entrenar.

## Preparar datos de ejemplo

Desde la raíz del proyecto:

```bash
python scripts/prepare_data.py --download-lfw
```

Esto descarga un subset de LFW (Labeled Faces in the Wild) y lo deja en `data/train` y `data/val`.

## Usar tus propias fotos

1. Crea `data/train` y `data/val`.
2. Crea una carpeta por persona, por ejemplo `data/train/Maria` y `data/train/Juan`.
3. Copia en cada carpeta varias fotos del rostro de esa persona (recomendado: caras recortadas y frontalmente orientadas para mejor resultado).
