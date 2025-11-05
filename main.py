"""
Pipeline de Data Mining para Análisis de Rendimiento Académico

Este script implementa un pipeline completo que incluye:
- Carga y sanitización de datos (Fases 2-3)
- Exploración y visualización (Fase 4)
- Modelado predictivo y descriptivo (Fase 5)
- Evaluación de modelos (Fase 6)

IMPORTANTE: Este proyecto utiliza MÉTRICAS PURAS (implementadas desde cero)
para todas las evaluaciones de modelos. Las métricas están en metricas_puras/metrics_puros.py
y NO se utilizan las métricas de sklearn.

Bibliotecas externas usadas SOLO para:
- pandas: Manipulación de datos
- sklearn: Modelos de ML (LogisticRegression, RidgeCV, DecisionTree, KMeans)
- matplotlib/seaborn: Visualización de datos
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
import warnings
from typing import Optional
import math
import sys
import os

# Importar métricas puras desde metricas_puras/metrics_puros.py
# TODAS las métricas de evaluación son implementadas desde cero (sin sklearn.metrics)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'metricas_puras'))
from metrics_puros import (  # type: ignore
    accuracy,
    precision_recall_f1_binary,
    confusion_matrix_binary,
    mae,
    rmse,
    r2,
)

# --- FASE 2: RECOLECCIÓN DE DATOS y FASE 3: SANITIZACIÓN ---

def _coaccionar_tipos_y_rangos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Asegura tipos numéricos esperados y aplica clamps suaves a rangos plausibles.
    No elimina filas; corrige valores fuera de rango con límites razonables.
    """
    df = df.copy()

    # Definición de columnas y rangos esperados
    numeric_specs = {
        'Actividad_Foros': (int, 0, 30),
        'Calidad_Foros': (float, 0.0, 5.0),
        'Tareas_Entregadas': (int, 0, 10),
        'Tiempo_Promedio_Respuesta': (float, 0.0, 10.0),
        'Interacciones_Semanales': (float, 0.0, 10.0),
        'Calificacion_Promedio': (float, 0.0, 10.0),
    }

    for col, (dtype, lo, hi) in numeric_specs.items():
        if col in df.columns:
            # Coaccionar a numérico
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Aplicar clamps de rango
            fuera_rango = ((df[col] < lo) | (df[col] > hi)).sum()
            if fuera_rango > 0:
                print(f"Advertencia: {fuera_rango} valores fuera de rango en '{col}' -> se acotan a [{lo}, {hi}].")
            df[col] = df[col].clip(lower=lo, upper=hi)
            # Ajuste de tipo entero si corresponde
            if dtype is int:
                # Redondeo al entero más cercano tras clamps
                df[col] = df[col].round().astype('Int64')
                # Reconvertir a int nativo si no hay nulos
                if df[col].isna().sum() == 0:
                    df[col] = df[col].astype(int)
    return df


def _eliminar_duplicados(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina duplicados basados en ('Estudiante_ID', 'Periodo_Academico') si existen.
    """
    df = df.copy()
    claves = [c for c in ['Estudiante_ID', 'Periodo_Academico'] if c in df.columns]
    if len(claves) == 2:
        antes = len(df)
        df = df.drop_duplicates(subset=claves)
        despues = len(df)
        eliminados = antes - despues
        if eliminados > 0:
            print(f"Se eliminaron {eliminados} duplicados por {claves}.")
    return df


def _agregar_indicadores_cap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade indicadores binarios de posibles valores 'cappeados' por el sistema de medición.
    - Tiempo_Promedio_Respuesta <= 0.5 (min cap) o >= 10.0 (max cap)
    - Interacciones_Semanales == 0.0
    """
    df = df.copy()
    if 'Tiempo_Promedio_Respuesta' in df.columns:
        df['TiempoResp_cap_min'] = (df['Tiempo_Promedio_Respuesta'] <= 0.5).astype(int)
        df['TiempoResp_cap_max'] = (df['Tiempo_Promedio_Respuesta'] >= 10.0 - 1e-9).astype(int)
    if 'Interacciones_Semanales' in df.columns:
        df['Interacciones_es_cero'] = (df['Interacciones_Semanales'] == 0.0).astype(int)
    return df


def cargar_y_sanear_datos(ruta_archivo: str) -> Optional[pd.DataFrame]:
    """
    Carga los datos desde un CSV y realiza una sanitización básica.
    (Fases 2 y 3 del proyecto)
    """
    print(f"--- Iniciando Fase 2 y 3: Carga y Sanitización ---")
    try:
        # Fase 2: Carga de datos
        df = pd.read_csv(ruta_archivo)
        print(f"Datos cargados exitosamente desde: {ruta_archivo}")
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo en la ruta: {ruta_archivo}")
        return None

    # Fase 3: Sanitización de datos
    print("\n[Fase 3: Sanitización]")
    print("Verificando tipos de datos y valores no nulos (df.info()):")
    df.info()

    print("\nVerificando valores faltantes (df.isnull().sum()):")
    print(df.isnull().sum())
    # En este punto, si hubiera valores faltantes, se decidiría una estrategia
    # (ej. df.dropna() o df.fillna(media, mediana, etc.))
    # Por la salida de df.info(), asumimos que no hay nulos.

    print("\nEstadísticas descriptivas (df.describe()):")
    # Esto ayuda a detectar valores erróneos (ej. Calificación > 10 o < 0)
    print(df.describe())

    print("\nVerificando valores únicos en columnas categóricas:")
    print(f"Valores en 'Estado_Final': {df['Estado_Final'].unique()}")
    print(f"Valores en 'Nivel_Rendimiento': {df['Nivel_Rendimiento'].unique()}")
    print(f"Valores en 'Periodo_Academico': {df['Periodo_Academico'].unique()}")

    # Sanitización adicional: duplicados y tipos/rangos
    df = _eliminar_duplicados(df)
    df = _coaccionar_tipos_y_rangos(df)
    df = _agregar_indicadores_cap(df)

    # Validación de columnas casi constantes (cero varianza)
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df[col].nunique(dropna=True) <= 1:
            print(f"Aviso: La columna numérica '{col}' es constante/casi constante; podría no aportar al modelo.")

    # Eliminamos columnas que no son features para el modelado (IDs)
    df_limpio = df.drop(columns=[c for c in ['Estudiante_ID', 'Periodo_Academico'] if c in df.columns])
    print("\nColumnas 'Estudiante_ID' y 'Periodo_Academico' eliminadas para el análisis.")

    print("--- Fin Fase 2 y 3 ---")
    return df_limpio

# --- FASE 4: PROCESAMIENTO (EXPLORACIÓN Y VISUALIZACIÓN) ---

def exploracion_y_visualizacion(df: Optional[pd.DataFrame]) -> None:
    """
    Genera la matriz de correlación y las visualizaciones requeridas.
    (Fase 4 del proyecto)
    """
    if df is None:
        return
    print(f"\n--- Iniciando Fase 4: Exploración y Visualización ---")

    # Seleccionar solo columnas numéricas para la correlación
    df_numeric = df.select_dtypes(include=[np.number])

    # 1. Matriz de Correlación (Datos)
    print("\n[Fase 4: Matriz de Correlación]")
    corr_matrix = df_numeric.corr()
    print(corr_matrix)

    # 2. Heatmap de Correlación (Visualización)
    print("Generando 'heatmap_correlacion.png'...")
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Heatmap de Correlación entre Variables Numéricas')
    plt.tight_layout()
    plt.savefig('heatmap_correlacion.png')
    # plt.show() # Descomentar si se ejecuta interactivamente

    # 3. Scatter Plot (Visualización)
    # Pregunta de investigación: ¿Participación en foros vs. rendimiento?
    print("Generando 'scatter_foros_vs_calificacion.png'...")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Actividad_Foros', y='Calificacion_Promedio', hue='Estado_Final', alpha=0.7)
    plt.title('Relación entre Actividad en Foros y Calificación Promedio')
    plt.xlabel('Actividad en Foros (Cantidad)')
    plt.ylabel('Calificación Promedio')
    plt.legend(title='Estado Final')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('scatter_foros_vs_calificacion.png')
    # plt.show()

    # 4. Box Plot (Visualización)
    print("Generando 'boxplot_estado_vs_calidad_foros.png'...")
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Estado_Final', y='Calidad_Foros')
    plt.title('Distribución de la Calidad en Foros según el Estado Final')
    plt.xlabel('Estado Final')
    plt.ylabel('Calidad de Intervenciones en Foros')
    plt.tight_layout()
    plt.savefig('boxplot_estado_vs_calidad_foros.png')
    # plt.show()
    
    print("--- Fin Fase 4 ---")

# --- FASE 5: MODELADO (PREDICTIVO) y FASE 6: EVALUACIÓN ---

def modelado_predictivo_logistico(df: Optional[pd.DataFrame]) -> None:
    """
    Aplica un modelo de Regresión Logística para predecir 'Estado_Final'
    y evalúa su rendimiento.
    (Fases 5 y 6 del proyecto)
    """
    if df is None:
        return
    print(f"\n--- Iniciando Fase 5 (Predictivo) y 6 (Evaluación) ---")
    print("Modelo Predictivo: Regresión Logística (Target: Estado_Final)")

    # Copia para evitar SettingWithCopyWarning
    df_modelo = df.copy()

    # 1. Preparación de datos
    # Codificar variable objetivo 'Estado_Final' (Aprobado=1, Reprobado=0)
    # Asumimos que solo hay 'Aprobado' y 'Reprobado'
    df_modelo['Estado_Final_Num'] = df_modelo['Estado_Final'].map({'Aprobado': 1, 'Reprobado': 0})
    
    # Verificar si hay valores nulos después del mapeo (si había otros valores)
    if df_modelo['Estado_Final_Num'].isnull().any():
        print("Advertencia: Se encontraron valores en 'Estado_Final' no esperados ('Aprobado'/'Reprobado').")
        # Decidir qué hacer: por ahora, eliminamos esas filas
        df_modelo.dropna(subset=['Estado_Final_Num'], inplace=True)
        
    df_modelo['Estado_Final_Num'] = df_modelo['Estado_Final_Num'].astype(int)

    # Definir Features (X) y Target (y)
    # Excluimos 'Nivel_Rendimiento' y 'Calificacion_Promedio' por ser
    # resultados directos o muy colineales con 'Estado_Final'
    features = [
        'Actividad_Foros', 
        'Calidad_Foros', 
        'Tareas_Entregadas', 
        'Tiempo_Promedio_Respuesta', 
        'Interacciones_Semanales'
    ]
    target = 'Estado_Final_Num'

    X = df_modelo[features]
    y = df_modelo[target]

    print(f"\nFeatures (X): {', '.join(features)}")
    print(f"Target (y): {target} (1=Aprobado, 0=Reprobado)")

    # 2. Balanceo mínimo para permitir estratificación en el split
    test_size = 0.3
    vc = y.value_counts(dropna=False)
    print("\nDistribución de clases antes del split:")
    print(vc)

    min_req_test = math.ceil(1 / test_size)
    min_req_train = math.ceil(1 / (1 - test_size))
    min_req_total = max(2, min_req_test, min_req_train)

    # Si alguna clase tiene menos de min_req_total instancias, sobre-muestreamos por duplicación
    if (vc.min() < min_req_total):
        print(f"Advertencia: Clase minoritaria con {vc.min()} muestras. Se realizará sobre-muestreo simple hasta {min_req_total} para habilitar estratificación.")
        df_oversampled = df_modelo.copy()
        target_col = target
        clases = vc.index.tolist()
        for clase in clases:
            df_clase = df_oversampled[df_oversampled[target_col] == clase]
            cnt = len(df_clase)
            if cnt < min_req_total:
                reps = min_req_total - cnt
                # Muestreo con reemplazo de las filas minoritarias
                df_oversampled = pd.concat([df_oversampled, df_clase.sample(n=reps, replace=True, random_state=42)], ignore_index=True)
        # Actualizar X, y tras el sobre-muestreo
        X = df_oversampled[features]
        y = df_oversampled[target_col]
        print("Nueva distribución de clases tras sobre-muestreo mínimo:")
        print(y.value_counts())

    # 3. División de datos (Train/Test Split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    print(f"Datos divididos: {len(X_train)} para entrenamiento, {len(X_test)} para prueba.")

    # 3. Escalado de características
    # Importante para modelos como Regresión Logística y K-Means
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Datos escalados (StandardScaler).")

    # 4. Modelado (Fase 5 - Predictivo)
    # Suprimir advertencias de convergencia para mantener limpia la salida
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    
    model = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
    model.fit(X_train_scaled, y_train)
    print("Modelo de Regresión Logística entrenado.")
    warnings.filterwarnings("default", category=ConvergenceWarning)

    # 5. Evaluación (Fase 6) - Usando métricas puras
    y_pred = model.predict(X_test_scaled)
    
    # Convertir a listas para métricas puras
    y_test_list = y_test.tolist()
    y_pred_list = y_pred.tolist()
    
    acc = accuracy(y_test_list, y_pred_list)
    prec, rec, f1 = precision_recall_f1_binary(y_test_list, y_pred_list)
    cm = confusion_matrix_binary(y_test_list, y_pred_list)

    print("\n--- [Fase 6: Evaluación del Modelo Predictivo] ---")
    print(f"Accuracy (Precisión Global): {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-Score (para clase 'Aprobado'): {f1:.4f}")
    
    print("\nMatriz de Confusión (métricas puras):")
    print(f"TP={cm['TP']}, TN={cm['TN']}, FP={cm['FP']}, FN={cm['FN']}")
    
    # Matriz en formato tabla
    mat_cm = [[cm['TN'], cm['FP']], [cm['FN'], cm['TP']]]
    print("\nMatriz (filas=real, columnas=predicho):")
    print(f"        Pred 0  Pred 1")
    print(f"Real 0    {mat_cm[0][0]:3d}     {mat_cm[0][1]:3d}")
    print(f"Real 1    {mat_cm[1][0]:3d}     {mat_cm[1][1]:3d}")

    print("\nJustificación de Métricas (Fase 6):")
    print("- Accuracy: Es útil si las clases (Aprobado/Reprobado) están balanceadas. Mide el porcentaje total de aciertos.")
    print("- F1-Score: Es más relevante si las clases están desbalanceadas. Es una media armónica entre 'precision' y 'recall'. Si fuera muy importante detectar a los 'Reprobados' (clase minoritaria), nos fijaríamos en el F1-Score de esa clase en específico.")
    
    print("--- Fin Fase 5 (Predictivo) y 6 ---")


def modelado_regresion_calificacion(df: Optional[pd.DataFrame]) -> None:
    """
    Modelo de regresión para predecir Calificacion_Promedio (alternativa robusta ante desbalance).
    Usa RidgeCV con estandarización y reporta MAE, RMSE y R^2.
    """
    if df is None:
        return
    print(f"\n--- Iniciando Fase 5 (Regresión) y 6 (Evaluación) ---")
    print("Modelo Predictivo: RidgeCV (Target: Calificacion_Promedio)")

    df_modelo = df.copy()

    target = 'Calificacion_Promedio'
    # Features: todas menos el target y etiquetas puramente de resultado
    features = [
        'Actividad_Foros', 'Calidad_Foros', 'Tareas_Entregadas',
        'Tiempo_Promedio_Respuesta', 'Interacciones_Semanales',
        # Indicadores de capping añadidos en sanitización
        'TiempoResp_cap_min', 'TiempoResp_cap_max', 'Interacciones_es_cero'
    ]
    features = [f for f in features if f in df_modelo.columns]

    X = df_modelo[features]
    y = df_modelo[target]

    print(f"\nFeatures (X): {', '.join(features)}")
    print(f"Target (y): {target}")

    test_size = 0.3
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    print(f"Datos divididos: {len(X_train)} para entrenamiento, {len(X_test)} para prueba.")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Datos escalados (StandardScaler).")

    # Ridge con selección de alpha por CV interna
    alphas = np.logspace(-3, 3, 13)
    model = RidgeCV(alphas=alphas, cv=5)
    model.fit(X_train_scaled, y_train)
    print(f"Modelo RidgeCV entrenado. Alpha óptimo: {model.alpha_}")

    y_pred = model.predict(X_test_scaled)
    
    # Convertir a listas para métricas puras
    y_test_list = y_test.tolist()
    y_pred_list = y_pred.tolist()
    
    mae_val = mae(y_test_list, y_pred_list)
    rmse_val = rmse(y_test_list, y_pred_list)
    r2_val = r2(y_test_list, y_pred_list)

    print("\n--- [Fase 6: Evaluación del Modelo de Regresión (Métricas Puras)] ---")
    print(f"MAE: {mae_val:.4f}")
    print(f"RMSE: {rmse_val:.4f}")
    print(f"R²: {r2_val:.4f}")

    # Gráficos de diagnóstico
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.xlabel('Calificación Real')
    plt.ylabel('Calificación Predicha')
    plt.title('Predicción vs Real (RidgeCV)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('regresion_pred_vs_real.png')

    plt.figure(figsize=(8, 6))
    residuos = y_test - y_pred
    plt.scatter(y_pred, residuos, alpha=0.7)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Calificación Predicha')
    plt.ylabel('Residuo (Real - Pred)')
    plt.title('Residuos vs Predicción (RidgeCV)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('regresion_residuos.png')
    print("Gráficos 'regresion_pred_vs_real.png' y 'regresion_residuos.png' generados.")
    print("--- Fin Fase 5 (Regresión) y 6 ---")


def modelado_predictivo_bajo_rendimiento(df: Optional[pd.DataFrame]) -> None:
    """
    Clasificación alternativa: 'Rendimiento_Bajo' si Nivel_Rendimiento == 'Regular' o Estado_Final == 'Reprobado'.
    Esto incrementa casos positivos y mitiga el problema de 1 único reprobado.
    """
    if df is None:
        return
    print(f"\n--- Iniciando Fase 5 (Clasificación Alternativa) y 6 (Evaluación) ---")
    print("Modelo Predictivo: Regresión Logística (Target: Rendimiento_Bajo)")

    df_modelo = df.copy()
    # Crear etiqueta binaria
    df_modelo['Rendimiento_Bajo'] = (
        (df_modelo.get('Nivel_Rendimiento') == 'Regular') |
        (df_modelo.get('Estado_Final') == 'Reprobado')
    ).astype(int)

    features = [
        'Actividad_Foros', 'Calidad_Foros', 'Tareas_Entregadas',
        'Tiempo_Promedio_Respuesta', 'Interacciones_Semanales',
        'TiempoResp_cap_min', 'TiempoResp_cap_max', 'Interacciones_es_cero'
    ]
    features = [f for f in features if f in df_modelo.columns]

    X = df_modelo[features]
    y = df_modelo['Rendimiento_Bajo']

    print(f"\nFeatures (X): {', '.join(features)}")
    print("Target (y): Rendimiento_Bajo (1=Bajo, 0=No Bajo)")
    print("Distribución de clases:")
    print(y.value_counts())

    test_size = 0.3
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    print(f"Datos divididos: {len(X_train)} para entrenamiento, {len(X_test)} para prueba.")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Datos escalados (StandardScaler).")

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    model = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
    model.fit(X_train_scaled, y_train)
    warnings.filterwarnings("default", category=ConvergenceWarning)
    print("Modelo entrenado.")

    y_pred = model.predict(X_test_scaled)
    
    # Convertir a listas para métricas puras
    y_test_list = y_test.tolist()
    y_pred_list = y_pred.tolist()
    
    acc = accuracy(y_test_list, y_pred_list)
    prec, rec, f1 = precision_recall_f1_binary(y_test_list, y_pred_list)
    cm = confusion_matrix_binary(y_test_list, y_pred_list)
    
    print("\n--- [Fase 6: Evaluación Clasificador Rendimiento_Bajo (Métricas Puras)] ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 (clase 1=Bajo): {f1:.4f}")
    print(f"\nMatriz de Confusión: TP={cm['TP']}, TN={cm['TN']}, FP={cm['FP']}, FN={cm['FN']}")
    print(f"        Pred 0  Pred 1")
    print(f"Real 0    {cm['TN']:3d}     {cm['FP']:3d}")
    print(f"Real 1    {cm['FN']:3d}     {cm['TP']:3d}")
    print("--- Fin Fase 5 (Clasificación Alternativa) y 6 ---")


def modelado_arbol_decision_clasificacion(df: Optional[pd.DataFrame]) -> None:
    """
    Árbol de Decisión para predecir Estado_Final.
    Muestra puntos de corte importantes y visualiza el árbol.
    (Fases 5 y 6 del proyecto)
    """
    if df is None:
        return
    print(f"\n--- Iniciando Fase 5 (Árbol de Decisión - Clasificación) y 6 (Evaluación) ---")
    print("Modelo Predictivo: Árbol de Decisión (Target: Estado_Final)")

    df_modelo = df.copy()
    
    # Codificar variable objetivo
    df_modelo['Estado_Final_Num'] = df_modelo['Estado_Final'].map({'Aprobado': 1, 'Reprobado': 0})
    
    if df_modelo['Estado_Final_Num'].isnull().any():
        print("Advertencia: Se encontraron valores en 'Estado_Final' no esperados.")
        df_modelo.dropna(subset=['Estado_Final_Num'], inplace=True)
    
    df_modelo['Estado_Final_Num'] = df_modelo['Estado_Final_Num'].astype(int)

    # Features
    features = [
        'Actividad_Foros', 'Calidad_Foros', 'Tareas_Entregadas',
        'Tiempo_Promedio_Respuesta', 'Interacciones_Semanales'
    ]
    target = 'Estado_Final_Num'

    X = df_modelo[features]
    y = df_modelo[target]

    print(f"\nFeatures (X): {', '.join(features)}")
    print(f"Target (y): {target} (1=Aprobado, 0=Reprobado)")

    # Balanceo si es necesario
    test_size = 0.3
    vc = y.value_counts(dropna=False)
    print("\nDistribución de clases:")
    print(vc)

    min_req_test = math.ceil(1 / test_size)
    min_req_train = math.ceil(1 / (1 - test_size))
    min_req_total = max(2, min_req_test, min_req_train)

    if (vc.min() < min_req_total):
        print(f"Sobre-muestreo para habilitar estratificación...")
        df_oversampled = df_modelo.copy()
        for clase in vc.index.tolist():
            df_clase = df_oversampled[df_oversampled[target] == clase]
            cnt = len(df_clase)
            if cnt < min_req_total:
                reps = min_req_total - cnt
                df_oversampled = pd.concat([df_oversampled, df_clase.sample(n=reps, replace=True, random_state=42)], ignore_index=True)
        X = df_oversampled[features]
        y = df_oversampled[target]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    print(f"Datos divididos: {len(X_train)} entrenamiento, {len(X_test)} prueba.")

    # Entrenar Árbol de Decisión
    # max_depth limitado para mejor interpretabilidad
    model = DecisionTreeClassifier(
        max_depth=4, 
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    print("Árbol de Decisión entrenado.")

    # Evaluación con métricas puras
    y_pred = model.predict(X_test)
    
    # Convertir a listas para métricas puras
    y_test_list = y_test.tolist()
    y_pred_list = y_pred.tolist()
    
    acc = accuracy(y_test_list, y_pred_list)
    prec, rec, f1 = precision_recall_f1_binary(y_test_list, y_pred_list)
    cm = confusion_matrix_binary(y_test_list, y_pred_list)

    print("\n--- [Fase 6: Evaluación Árbol de Decisión (Métricas Puras)] ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"\nMatriz de Confusión: TP={cm['TP']}, TN={cm['TN']}, FP={cm['FP']}, FN={cm['FN']}")
    print(f"        Pred 0  Pred 1")
    print(f"Real 0    {cm['TN']:3d}     {cm['FP']:3d}")
    print(f"Real 1    {cm['FN']:3d}     {cm['TP']:3d}")

    # Importancia de características (puntos de corte importantes)
    print("\n--- Importancia de Características (Puntos de Corte) ---")
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    print(feature_importance_df)

    # Visualización del árbol
    print("\nGenerando visualización del árbol...")
    plt.figure(figsize=(20, 10))
    plot_tree(
        model, 
        feature_names=features,
        class_names=['Reprobado', 'Aprobado'],
        filled=True,
        rounded=True,
        fontsize=10
    )
    plt.title('Árbol de Decisión - Predicción Estado_Final')
    plt.tight_layout()
    plt.savefig('arbol_decision_estado_final.png', dpi=300, bbox_inches='tight')
    print("Árbol guardado en 'arbol_decision_estado_final.png'")

    # Gráfico de importancia de características
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
    plt.xlabel('Importancia')
    plt.title('Importancia de Características en Árbol de Decisión')
    plt.tight_layout()
    plt.savefig('arbol_importancia_features.png')
    print("Importancia guardada en 'arbol_importancia_features.png'")
    
    print("--- Fin Fase 5 (Árbol Clasificación) y 6 ---")


def modelado_arbol_decision_regresion(df: Optional[pd.DataFrame]) -> None:
    """
    Árbol de Decisión para predecir Calificacion_Promedio (regresión).
    Muestra puntos de corte importantes y visualiza el árbol.
    (Fases 5 y 6 del proyecto)
    """
    if df is None:
        return
    print(f"\n--- Iniciando Fase 5 (Árbol de Decisión - Regresión) y 6 (Evaluación) ---")
    print("Modelo Predictivo: Árbol de Decisión Regresión (Target: Calificacion_Promedio)")

    df_modelo = df.copy()
    
    target = 'Calificacion_Promedio'
    features = [
        'Actividad_Foros', 'Calidad_Foros', 'Tareas_Entregadas',
        'Tiempo_Promedio_Respuesta', 'Interacciones_Semanales',
        'TiempoResp_cap_min', 'TiempoResp_cap_max', 'Interacciones_es_cero'
    ]
    features = [f for f in features if f in df_modelo.columns]

    X = df_modelo[features]
    y = df_modelo[target]

    print(f"\nFeatures (X): {', '.join(features)}")
    print(f"Target (y): {target}")

    # Split
    test_size = 0.3
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    print(f"Datos divididos: {len(X_train)} entrenamiento, {len(X_test)} prueba.")

    # Entrenar Árbol de Decisión (Regresión)
    model = DecisionTreeRegressor(
        max_depth=4,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    model.fit(X_train, y_train)
    print("Árbol de Decisión (Regresión) entrenado.")

    # Evaluación con métricas puras
    y_pred = model.predict(X_test)
    
    # Convertir a listas para métricas puras
    y_test_list = y_test.tolist()
    y_pred_list = y_pred.tolist()
    
    mae_val = mae(y_test_list, y_pred_list)
    rmse_val = rmse(y_test_list, y_pred_list)
    r2_val = r2(y_test_list, y_pred_list)

    print("\n--- [Fase 6: Evaluación Árbol de Decisión Regresión (Métricas Puras)] ---")
    print(f"MAE: {mae_val:.4f}")
    print(f"RMSE: {rmse_val:.4f}")
    print(f"R²: {r2_val:.4f}")

    # Importancia de características
    print("\n--- Importancia de Características (Puntos de Corte) ---")
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    print(feature_importance_df)

    # Visualización del árbol
    print("\nGenerando visualización del árbol...")
    plt.figure(figsize=(20, 10))
    plot_tree(
        model,
        feature_names=features,
        filled=True,
        rounded=True,
        fontsize=9
    )
    plt.title('Árbol de Decisión - Predicción Calificacion_Promedio')
    plt.tight_layout()
    plt.savefig('arbol_decision_calificacion.png', dpi=300, bbox_inches='tight')
    print("Árbol guardado en 'arbol_decision_calificacion.png'")

    # Gráfico de importancia
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
    plt.xlabel('Importancia')
    plt.title('Importancia de Características en Árbol de Decisión (Regresión)')
    plt.tight_layout()
    plt.savefig('arbol_regresion_importancia_features.png')
    print("Importancia guardada en 'arbol_regresion_importancia_features.png'")

    # Gráficos de diagnóstico
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.xlabel('Calificación Real')
    plt.ylabel('Calificación Predicha')
    plt.title('Predicción vs Real (Árbol de Decisión)')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('arbol_regresion_pred_vs_real.png')
    print("Gráfico 'arbol_regresion_pred_vs_real.png' generado.")

    print("--- Fin Fase 5 (Árbol Regresión) y 6 ---")


# --- FASE 5: MODELADO (DESCRIPTIVO) ---

def modelado_descriptivo_kmeans(df: Optional[pd.DataFrame]) -> None:
    """
    Aplica un modelo K-Means para segmentar a los estudiantes.
    (Fase 5 del proyecto)
    """
    if df is None:
        return
    print(f"\n--- Iniciando Fase 5 (Descriptivo) ---")
    print("Modelo Descriptivo: K-Means Clustering")

    # 1. Preparación de datos
    # Vamos a segmentar usando 'Actividad_Foros' y 'Calificacion_Promedio'
    features_cluster = ['Actividad_Foros', 'Calificacion_Promedio']
    df_cluster = df[features_cluster].copy()

    # 2. Escalado de características
    # K-Means es sensible a la escala de los datos
    scaler = StandardScaler()
    df_cluster_scaled = scaler.fit_transform(df_cluster)
    print("Datos escalados para K-Means.")

    # 3. Encontrar el 'K' óptimo (Método del Codo)
    inertias = []
    K_range = range(1, 11)
    
    print("Calculando 'Método del Codo' para encontrar K óptimo...")
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(df_cluster_scaled)
        inertias.append(kmeans.inertia_)

    # Graficar el Método del Codo
    print("Generando 'kmeans_metodo_del_codo.png'...")
    plt.figure(figsize=(10, 6))
    plt.plot(K_range, inertias, 'bo-')
    plt.xlabel('Número de Clusters (k)')
    plt.ylabel('Inercia (Suma de distancias al cuadrado)')
    plt.title('Método del Codo para K-Means')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig('kmeans_metodo_del_codo.png')
    # plt.show()
    print("Revisar 'kmeans_metodo_del_codo.png' para elegir K. Usaremos K=3 por defecto.")

    # 4. Aplicar K-Means
    # Basado en el gráfico del codo, elegimos un K (ej. 3 o 4)
    K_OPTIMO = 3
    kmeans = KMeans(n_clusters=K_OPTIMO, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(df_cluster_scaled)

    # 5. Visualización de Clusters
    df_resultado_cluster = df.copy()
    df_resultado_cluster['Cluster'] = clusters
    df_resultado_cluster['Cluster'] = df_resultado_cluster['Cluster'].astype('category')

    print(f"Generando 'kmeans_clusters_resultado_{K_OPTIMO}k.png'...")
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=df_resultado_cluster, 
        x='Actividad_Foros', 
        y='Calificacion_Promedio', 
        hue='Cluster', 
        palette='viridis', 
        alpha=0.8,
        s=100
    )
    plt.title(f'Segmentación de Estudiantes (K={K_OPTIMO})')
    plt.xlabel('Actividad en Foros')
    plt.ylabel('Calificación Promedio')
    plt.legend(title='Cluster')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'kmeans_clusters_resultado_{K_OPTIMO}k.png')
    # plt.show()

    print(f"\nInterpretación de los {K_OPTIMO} clusters (centroides en valores escalados):")
    print(kmeans.cluster_centers_)
    print("\n(Estos centroides ayudarían a definir los perfiles, ej. 'Alto rendimiento, baja participación', etc.)")
    print("--- Fin Fase 5 (Descriptivo) ---")


# --- Ejecución Principal ---

def main() -> None:
    """
    Función principal para orquestar el pipeline de Data Mining.
    """
    # Ruta al archivo CSV. Asegúrate de que este archivo esté en el mismo
    # directorio que el script, o proporciona la ruta completa.
    RUTA_ARCHIVO = 'dataset_rendimiento_academico.csv'

    # Fases 2 y 3
    datos_limpios = cargar_y_sanear_datos(RUTA_ARCHIVO)
    
    if datos_limpios is not None:
        # Fase 4
        exploracion_y_visualizacion(datos_limpios)

        # Fases 5-6: Clasificación original (Estado_Final)
        modelado_predictivo_logistico(datos_limpios)

        # Fases 5-6: Regresión alternativa (Calificacion_Promedio)
        modelado_regresion_calificacion(datos_limpios)

        # Fases 5-6: Clasificación alternativa (Rendimiento_Bajo)
        modelado_predictivo_bajo_rendimiento(datos_limpios)

        # Fases 5-6: Árbol de Decisión - Clasificación (Estado_Final)
        modelado_arbol_decision_clasificacion(datos_limpios)

        # Fases 5-6: Árbol de Decisión - Regresión (Calificacion_Promedio)
        modelado_arbol_decision_regresion(datos_limpios)

        # Fase 5 (Descriptivo)
        modelado_descriptivo_kmeans(datos_limpios)

        print("\n--- ANÁLISIS COMPLETO ---")
        print("Se generaron los siguientes archivos:")
        print("- heatmap_correlacion.png")
        print("- scatter_foros_vs_calificacion.png")
        print("- boxplot_estado_vs_calidad_foros.png")
        print("- regresion_pred_vs_real.png")
        print("- regresion_residuos.png")
        print("- arbol_decision_estado_final.png")
        print("- arbol_importancia_features.png")
        print("- arbol_decision_calificacion.png")
        print("- arbol_regresion_importancia_features.png")
        print("- arbol_regresion_pred_vs_real.png")
        print("- kmeans_metodo_del_codo.png")
        print("- kmeans_clusters_resultado_3k.png")

if __name__ == "__main__":
    main()