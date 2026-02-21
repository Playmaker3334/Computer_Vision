# Blood Cell Detection — YOLO11

Proyecto completo de detección de células sanguíneas (RBC, WBC, Platelets)
usando YOLO11 entrenado sobre el dataset BCCD de Roboflow.

---

## Arquitectura del Proyecto

```
blood_cell_detection/
│
├── configs/
│   ├── config.yaml                  # Configuración global del proyecto
│   └── data.yaml                    # Configuración del dataset para YOLO
│
├── data/
│   ├── raw/
│   │   ├── images/                  # Imágenes originales descargadas
│   │   └── annotations/             # Anotaciones originales (XML / JSON)
│   ├── processed/
│   │   ├── train/
│   │   │   ├── images/              # Imágenes de entrenamiento
│   │   │   └── labels/              # Labels YOLO (.txt) de entrenamiento
│   │   ├── val/
│   │   │   ├── images/              # Imágenes de validación
│   │   │   └── labels/              # Labels YOLO (.txt) de validación
│   │   └── test/
│   │       ├── images/              # Imágenes de test
│   │       └── labels/              # Labels YOLO (.txt) de test
│   └── video/
│       ├── raw/                     # Videos originales (.mp4, .avi)
│       └── frames/                  # Frames extraídos de los videos
│
├── src/
│   ├── data/
│   │   ├── download_dataset.py      # Descarga dataset de Roboflow
│   │   └── extract_video_frames.py  # Extrae frames de videos
│   ├── training/
│   │   └── train.py                 # Entrena el modelo YOLO11
│   ├── evaluation/
│   │   └── evaluate.py              # Evalúa el modelo en test set
│   ├── inference/
│   │   └── predict.py               # Inferencia en imágenes o video
│   └── utils/
│       └── visualize.py             # Visualizaciones y gráficas
│
├── models/
│   ├── weights/
│   │   └── best.pt                  # Mejor modelo entrenado
│   └── experiments/
│       └── run/                     # Resultados del training (plots, logs)
│           └── weights/
│               ├── best.pt
│               └── last.pt
│
├── outputs/
│   ├── predictions/
│   │   ├── images/                  # Imágenes con bboxes dibujados
│   │   └── video/                   # Video anotado con detecciones
│   ├── reports/
│   │   └── evaluation_YYYYMMDD.json # Reporte de métricas por ejecución
│   └── visualizations/
│       ├── class_distribution.png   # Distribución de clases en dataset
│       └── metrics_summary.png      # Resumen de métricas del modelo
│
├── notebooks/                       # Jupyter notebooks de exploración
│
├── run_pipeline.py                  # Script maestro del pipeline completo
└── requirements.txt                 # Dependencias del proyecto
```

---

## Paths Clave

| Recurso | Path |
|---|---|
| Configuración global | `configs/config.yaml` |
| Configuración YOLO dataset | `configs/data.yaml` |
| Imágenes train | `data/processed/train/images/` |
| Labels train | `data/processed/train/labels/` |
| Imágenes val | `data/processed/val/images/` |
| Labels val | `data/processed/val/labels/` |
| Imágenes test | `data/processed/test/images/` |
| Labels test | `data/processed/test/labels/` |
| Videos raw | `data/video/raw/` |
| Frames extraídos | `data/video/frames/` |
| Mejor modelo | `models/weights/best.pt` |
| Experimentos de training | `models/experiments/run/` |
| Predicciones imágenes | `outputs/predictions/images/` |
| Predicciones video | `outputs/predictions/video/` |
| Reportes de evaluación | `outputs/reports/` |
| Visualizaciones | `outputs/visualizations/` |

---

## Instalación

```bash
pip install -r requirements.txt
```

---

## Uso

### Pipeline completo
```bash
python run_pipeline.py --steps download train evaluate visualize
```

### Solo entrenar
```bash
python run_pipeline.py --steps train
```

### Solo inferencia en imagen
```bash
python run_pipeline.py --steps inference --source data/processed/test/images/
```

### Solo inferencia en video
```bash
python run_pipeline.py --steps inference --source data/video/raw/sample.mp4
```

### Pasos individuales
```bash
# Descargar dataset
python src/data/download_dataset.py

# Extraer frames de video
python src/data/extract_video_frames.py

# Entrenar
python src/training/train.py

# Evaluar
python src/evaluation/evaluate.py

# Inferencia
python src/inference/predict.py --source data/processed/test/images/ --mode image
python src/inference/predict.py --source data/video/raw/sample.mp4 --mode video
```

---

## Clases

| Índice | Clase | Descripción |
|---|---|---|
| 0 | RBC | Red Blood Cells |
| 1 | WBC | White Blood Cells |
| 2 | Platelets | Plaquetas |

---

## Métricas de referencia (Roboflow baseline)

| Métrica | Valor |
|---|---|
| mAP@50 | 93.1% |
| Precision | 87.2% |
| Recall | 89.8% |
