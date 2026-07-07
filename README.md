# Predicción de Diabetes con Machine Learning

Trabajo final del curso *Machine Learning con Python*. El proyecto aplica técnicas de clasificación binaria para predecir la presencia de diabetes a partir de datos clínicos y fisiológicos.

## Objetivo

Determinar, a partir de variables biométricas (glucosa, presión arterial, IMC, edad, etc.), si un paciente presenta o no diabetes, comparando el desempeño de dos modelos de clasificación: **Regresión Logística** y **Random Forest**.

## Dataset

[Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) (National Institute of Diabetes and Digestive and Kidney Diseases), cargado directamente vía `kagglehub`. Contiene 768 registros con 8 variables independientes y una variable objetivo binaria (`Outcome`).

## Contenido del notebook

1. **Carga y exploración de datos** : inspección inicial y estadísticas descriptivas.
2. **Limpieza de datos** : eliminación de valores 0 biológicamente imposibles en glucosa, presión arterial, IMC e insulina (dataset final: 392 casos).
3. **Análisis exploratorio** : matriz de correlación y distribución de clases (66.8% negativos / 33.2% positivos).
4. **Modelado** : entrenamiento de Regresión Logística y Random Forest sobre un split 80/20.
5. **Evaluación** : métricas (Accuracy, Precision, Recall, F1) y validación cruzada estratificada (5-fold).
6. **Ajuste de umbral de decisión** : análisis de un umbral más sensible (15%) para priorizar la detección de casos positivos.
7. **Conclusiones** : comparación de modelos y discusión de resultados.

## Resultados principales

Ambos modelos obtienen un desempeño similar (Accuracy ≈ 0.78), sin una diferencia estadísticamente clara entre ellos según la validación cruzada. Random Forest muestra una ligera ventaja en Recall y F1, pero dentro del margen de la desviación estándar. El ajuste de umbral resultó más determinante que la elección del modelo para mejorar la detección de casos positivos.

## Requisitos

```
kagglehub
pandas
numpy
seaborn
matplotlib
scikit-learn
```

## Cómo ejecutar

1. Instalar las dependencias: `pip install kagglehub pandas numpy seaborn matplotlib scikit-learn`
2. Abrir `ML_TT_GH_jul26.ipynb` en Jupyter y ejecutar las celdas en orden.

