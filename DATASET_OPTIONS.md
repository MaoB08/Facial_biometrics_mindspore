# Opciones para Dataset Grande

## ❌ Problema: Sin conexión a internet

No se pudo descargar el dataset LFW automáticamente porque no hay conexión a internet disponible.

---

## ✅ Opciones disponibles:

### **Opción 1: Descargar LFW manualmente** (Recomendado)

1. **Descargar el archivo** desde tu navegador:
   - URL: http://vis-www.cs.umass.edu/lfw/lfw.tgz
   - Tamaño: ~173 MB
   - Contiene: ~13,000 imágenes de ~5,700 personas

2. **Extraer en el proyecto**:
   ```bash
   # Si descargaste a ~/Descargas/
   tar -xzf ~/Descargas/lfw.tgz -C /home/mauricio/Escritorio/Facial_biometrics_mindspore/data/
   
   # Luego ejecutar el script de preparación
   cd /home/mauricio/Escritorio/Facial_biometrics_mindspore
   source venv/bin/activate
   python3 scripts/prepare_data.py --organize-lfw
   ```

---

### **Opción 2: Agregar más personas al dataset actual**

Puedes simplemente agregar más carpetas de personas a `data/train/` y `data/val/`:

```bash
data/train/
├── bill_gates/
├── elon_musk/
├── persona_3/    # Nueva
├── persona_4/    # Nueva
└── persona_N/    # Nueva
```

**Recomendación mínima**:
- Al menos 10-20 personas diferentes
- 5-10 fotos por persona en train
- 2-3 fotos por persona en val

---

### **Opción 3: Usar otro dataset público**

Otros datasets que puedes descargar manualmente:

| Dataset | Personas | Imágenes | Tamaño | URL |
|---------|----------|----------|--------|-----|
| **LFW** | 5,749 | 13,233 | 173 MB | http://vis-www.cs.umass.edu/lfw/lfw.tgz |
| **CelebA** | 10,177 | 202,599 | 1.4 GB | https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html |
| **VGGFace2** | 9,131 | 3.31M | ~37 GB | http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/ |

---

### **Opción 4: Continuar con el dataset actual**

Si solo quieres practicar el proceso, puedes:
- Mantener el dataset pequeño actual (2 personas)
- Experimentar con hiperparámetros
- Entender el flujo completo

**Nota**: El modelo tendrá overfitting, pero es válido para aprendizaje.

---

## 📝 Próximos pasos

Una vez que tengas el dataset:

1. **Ajustar BATCH_SIZE** en `config.py`:
   - Para LFW completo: `BATCH_SIZE = 32`
   - Para dataset pequeño: mantener `BATCH_SIZE = 4`

2. **Entrenar**:
   ```bash
   source venv/bin/activate
   python3 train.py
   ```

3. **Validar**:
   ```bash
   python3 test.py --eval
   python3 test.py --verify foto1.jpg foto2.jpg
   ```

---

## 🎯 Estado actual

✅ **Versión base funcional** guardada en git (commit f1d4966)  
✅ **Código completo** y probado  
⏸️ **Esperando dataset** para entrenamiento a escala real  
