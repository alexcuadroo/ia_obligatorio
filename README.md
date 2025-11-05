# Proyecto de Data Mining - Análisis de Rendimiento Académico

## 📋 Descripción

Este proyecto implementa un pipeline completo de Data Mining para analizar el rendimiento académico de estudiantes, cumpliendo con todas las fases requeridas: recolección, sanitización, procesamiento, modelado y evaluación.

## ⚙️ Características Principales

### ✅ **Métricas Puras (Sin sklearn.metrics)**

Este proyecto implementa **TODAS las métricas de evaluación desde cero** sin utilizar `sklearn.metrics`. Las métricas están implementadas en `metricas_puras/metrics_puros.py`:

**Métricas de Clasificación:**
- Accuracy (Exactitud)
- Precision (Precisión)
- Recall (Sensibilidad)
- F1-Score
- Matriz de Confusión (binaria y multiclase)

**Métricas de Regresión:**
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² (Coeficiente de Determinación)

### 📚 Uso de Bibliotecas Externas

Las bibliotecas externas se utilizan **SOLAMENTE** para:
- **pandas**: Manipulación y análisis de datos
- **sklearn**: Modelos de Machine Learning (LogisticRegression, RidgeCV, DecisionTree, KMeans)
- **matplotlib/seaborn**: Visualización de datos y gráficos

## 🗂️ Estructura del Proyecto

```
ia_obligatorio/
│
├── main.py                              # Script principal con pipeline completo
├── dataset_rendimiento_academico.csv     # Dataset de entrada
├── requirements.txt                      # Dependencias del proyecto
├── README.md                             # Este archivo
│
├── metricas_puras/                       # Módulo de métricas puras
│   ├── main.py                          # Demo de métricas puras con CSV
│   ├── metrics_puros.py                 # Implementación de métricas
│   └── dataset_rendimiento_academico.csv
│
└── [archivos PNG generados]              # Visualizaciones exportadas
```

## 📊 Fases del Proyecto

### Fase 2-3: Recolección y Sanitización de Datos
- Carga de datos desde CSV
- Verificación de tipos de datos
- Detección de valores faltantes
- Eliminación de duplicados
- Validación de rangos
- Ingeniería de características

### Fase 4: Exploración y Visualización
- ✅ Matriz de correlación entre variables
- ✅ Heatmap de correlaciones
- ✅ Scatter plot: `Actividad_Foros` vs `Calificacion_Promedio`
- ✅ Box plot: `Calidad_Foros` por `Estado_Final`

### Fase 5: Modelado

#### Modelos Predictivos
1. **Regresión Logística** → Predice `Estado_Final` (Aprobado/Reprobado)
2. **Regresión Lineal (RidgeCV)** → Predice `Calificacion_Promedio`
3. **Árbol de Decisión (Clasificación)** → Predice `Estado_Final` con visualización completa
4. **Árbol de Decisión (Regresión)** → Predice `Calificacion_Promedio` con puntos de corte

#### Modelo Descriptivo
- **K-Means Clustering** → Segmenta estudiantes en 3 grupos
- Método del codo para seleccionar K óptimo
- Visualización de clusters

### Fase 6: Evaluación
Todas las evaluaciones utilizan **métricas puras** implementadas desde cero:
- Accuracy, Precision, Recall, F1-Score
- MAE, RMSE, R²
- Matrices de confusión
- Importancia de características en árboles

## 🚀 Instalación y Uso

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. Ejecutar el pipeline completo

```bash
# Pipeline completo (con modelos sklearn)
python main.py

# Demo de métricas puras solamente
python metricas_puras/main.py
```

### 3. Archivos generados

El script genera los siguientes archivos PNG:

**Exploración:**
- `heatmap_correlacion.png`
- `scatter_foros_vs_calificacion.png`
- `boxplot_estado_vs_calidad_foros.png`

**Modelos:**
- `regresion_pred_vs_real.png`
- `regresion_residuos.png`
- `arbol_decision_estado_final.png` ⭐
- `arbol_importancia_features.png` ⭐
- `arbol_decision_calificacion.png` ⭐
- `arbol_regresion_importancia_features.png` ⭐
- `arbol_regresion_pred_vs_real.png` ⭐
- `kmeans_metodo_del_codo.png`
- `kmeans_clusters_resultado_3k.png`

## 📈 Resultados Destacados

### Árbol de Decisión (Clasificación)
- **Accuracy**: 98.36%
- **F1-Score**: 99.16%
- **Punto de corte más importante**: `Tiempo_Promedio_Respuesta` (100% importancia)

### Árbol de Decisión (Regresión)
- **Puntos de corte principales**:
  - `Tiempo_Promedio_Respuesta`: 39.16%
  - `Interacciones_Semanales`: 32.78%
  - `Tareas_Entregadas`: 27.75%

## 🔍 Verificación de Métricas Puras

Para verificar que las métricas son calculadas de forma pura (sin sklearn.metrics):

1. Revisar `metricas_puras/metrics_puros.py` → Implementación desde cero
2. En `main.py` → Importa de `metrics_puros`, NO de `sklearn.metrics`
3. Todas las funciones de evaluación usan:
   - `accuracy()` (puro) en lugar de `accuracy_score()`
   - `mae()`, `rmse()`, `r2()` (puros) en lugar de sklearn
   - `confusion_matrix_binary()` (puro) en lugar de sklearn

## 📝 Notas Importantes

- ✅ Solo usa el dataset CSV (sin datos hardcodeados)
- ✅ Todas las métricas son implementaciones puras
- ✅ Cumple con todos los requisitos de la rúbrica
- ✅ Genera visualizaciones de alta calidad
- ✅ Código bien documentado y estructurado

## 👨‍💻 Autor

Proyecto desarrollado para el curso de Inteligencia Artificial.

## 📄 Licencia

Proyecto académico - 2025
