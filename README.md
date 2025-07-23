# Deep Learning para predecir interacciones proteína-proteína

Este proyecto corresponde al Trabajo Fin de Máster (TFM) en Inteligencia Artificial de Santiago Martinez Willi. El objetivo principal es desarrollar modelos de aprendizaje profundo para predecir interacciones proteína-proteína (PPI) a partir de secuencias de aminoácidos, utilizando únicamente datos públicos y técnicas computacionales reproducibles.

## 🌱 Motivación

Las interacciones proteína-proteína son fundamentales para el funcionamiento celular y los mecanismos de enfermedades humanas. Sin embargo, los métodos experimentales actuales para identificarlas (como Y2H o AP/MS) son costosos y presentan alta tasa de falsos positivos y negativos. Este proyecto explora el uso de redes neuronales profundas, incluyendo arquitecturas siamesas y embeddings preentrenados, como alternativa eficiente y escalable.

## 🧠 Modelos implementados

El proyecto evalúa tres modelos principales:

1. **Modelo siamés convolucional**
   - Entrada: codificación numérica simple con un máximo de longitud de secuencia de 1024
   - Arquitectura: red neuronal siamesa con convoluciones 1D
   - Accuracy: 84.72%, F1: 0.8501, ROC AUC: 0.9317

2. **Modelo basado en la arquitectura PIPR**
   - Entrada: codificación numérica simple con un máximo de longitud de secuencia de 768
   - Arquitectura: siamesa con capas residuales RCNN
   - Accuracy: 75.23%, F1: 0.7422, ROC AUC: 0.8309

3. **Modelo basado en embeddings preentrenados (ESM-2)**
   - Entrada: vectores de embedding generados con ESM-2 (t6_8M_UR50D)
   - Arquitectura: Siamese MLP
   - Accuracy: 84.36%, F1: 0.8426, ROC AUC: 0.9220

## 📊 Resultados

| Modelo                                       | Accuracy (%) | F1 Score | ROC AUC | Épocas | Tiempo por época | Tiempo total aprox. |
|---------------------------------------------|--------------|----------|---------|--------|------------------|----------------------|
| Siamese convolucional                       | 84.72        | 0.8501   | 0.9317  | 50     | ~8 min           | ~7 h                 |
| Arquitectura PIPR (RCNN residual)           | 75.23        | 0.7422   | 0.8309  | 9      | ~80 min          | ~12 h                |
| Siamese MLP + embeddings ESM-2 (t6_8M_UR50D)| 84.36        | 0.8426   | 0.9220  | 36     | ~2 min           | ~1 h + 15 h embeddings |

## 📁 Estructura del proyecto

```
TFM/
├── notebooks/           # Notebooks Jupyter para experimentación y entrenamiento
├── datasets/            # Datos originales y procesados
├── processed_data/      # Datos transformados y embeddings
├── models/              # Modelos entrenados y checkpoints
├── src/                 # Código fuente principal
│   ├── computation/     # Integración con Azure ML
│   ├── encoders/        # Encoders de secuencias proteicas
│   ├── models/          # Arquitecturas y pipelines de entrenamiento
│   ├── utils.py         # Utilidades generales
│   ├── proteins_preprocessor.py
│   ├── protein_dataset.py
│   ├── ...
├── requirements.txt     # Dependencias del proyecto
├── .env                 # Variables de entorno para Azure ML
```

## 🔧 Instalación

1. **Clona el repositorio:**
     ```bash
     git clone <url-del-repo>
     cd TFM
     ```

2. **Crea y activa un entorno virtual:**
     ```bash
     python3 -m venv tfm_env
     source tfm_env/bin/activate
     ```

3. **Instala las dependencias:**
     ```bash
     pip install -r requirements.txt
     ```

4. **Configura las variables de entorno para Azure ML si planeas usarlo:**  
     A partir del archivo `.env.example`, crea el archivo `.env` con tus credenciales y recursos de Azure.

## 📚 Referencias

- Chen et al. (2019). *Siamese residual RCNN for protein–protein interaction prediction*. Bioinformatics. https://doi.org/10.1093/bioinformatics/btz328

- Spinello et al. (2022). *Deep learning approaches to protein-protein interaction prediction: a systematic review*. Frontiers in Bioinformatics. https://doi.org/10.3389/fbinf.2022.837721

- HuggingFace Transformers: [https://huggingface.co](https://huggingface.co)

- STRING: [https://string-db.org](https://string-db.org)

- Ensembl: [https://www.ensembl.org](https://www.ensembl.org)


## 📄 Licencia
Este proyecto está bajo licencia MIT. Uso académico permitido con cita adecuada.
