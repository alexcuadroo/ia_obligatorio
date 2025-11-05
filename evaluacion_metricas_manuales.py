"""
Evaluación de rendimiento de modelos utilizando métricas implementadas manualmente.

Este script implementa desde cero (sin sklearn.metrics):
- Accuracy (Exactitud)
- F1-Score
- Precision y Recall
- Matriz de Confusión

Solo usa librerías de visualización (matplotlib) para graficar resultados.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from typing import List, Dict, Tuple


# =====================================================
# IMPLEMENTACIÓN MANUAL DE MÉTRICAS (SIN SKLEARN.METRICS)
# =====================================================

def calcular_matriz_confusion(y_real: List[int], y_pred: List[int]) -> Dict[str, int]:
    """
    Calcula la matriz de confusión para clasificación binaria.
    
    Args:
        y_real: Lista de valores reales (0 o 1)
        y_pred: Lista de valores predichos (0 o 1)
    
    Returns:
        Diccionario con TP, TN, FP, FN
    """
    tp = tn = fp = fn = 0
    
    for real, pred in zip(y_real, y_pred):
        if real == 1 and pred == 1:
            tp += 1  # Verdadero Positivo
        elif real == 0 and pred == 0:
            tn += 1  # Verdadero Negativo
        elif real == 0 and pred == 1:
            fp += 1  # Falso Positivo
        elif real == 1 and pred == 0:
            fn += 1  # Falso Negativo
    
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}


def calcular_accuracy(y_real: List[int], y_pred: List[int]) -> float:
    """
    Calcula la exactitud (accuracy) del modelo.
    
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    
    Es el porcentaje de predicciones correctas sobre el total.
    """
    if len(y_real) == 0:
        return 0.0
    
    aciertos = sum(1 for real, pred in zip(y_real, y_pred) if real == pred)
    total = len(y_real)
    
    accuracy = aciertos / total
    return accuracy


def calcular_precision(y_real: List[int], y_pred: List[int]) -> float:
    """
    Calcula la precisión del modelo.
    
    Precision = TP / (TP + FP)
    
    De todas las predicciones positivas, ¿cuántas son correctas?
    """
    cm = calcular_matriz_confusion(y_real, y_pred)
    tp = cm["TP"]
    fp = cm["FP"]
    
    if (tp + fp) == 0:
        return 0.0
    
    precision = tp / (tp + fp)
    return precision


def calcular_recall(y_real: List[int], y_pred: List[int]) -> float:
    """
    Calcula el recall (sensibilidad) del modelo.
    
    Recall = TP / (TP + FN)
    
    De todos los casos positivos reales, ¿cuántos detectamos?
    """
    cm = calcular_matriz_confusion(y_real, y_pred)
    tp = cm["TP"]
    fn = cm["FN"]
    
    if (tp + fn) == 0:
        return 0.0
    
    recall = tp / (tp + fn)
    return recall


def calcular_f1_score(y_real: List[int], y_pred: List[int]) -> float:
    """
    Calcula el F1-Score del modelo.
    
    F1 = 2 * (Precision * Recall) / (Precision + Recall)
    
    Es la media armónica entre precisión y recall.
    Útil cuando hay desbalance de clases.
    """
    precision = calcular_precision(y_real, y_pred)
    recall = calcular_recall(y_real, y_pred)
    
    if (precision + recall) == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


# =====================================================
# FUNCIONES DE VISUALIZACIÓN
# =====================================================

def graficar_matriz_confusion(cm: Dict[str, int], titulo: str = "Matriz de Confusión", 
                               guardar_como: str = None):
    """
    Grafica la matriz de confusión de forma visual.
    """
    # Crear matriz 2x2
    matriz = np.array([[cm['TN'], cm['FP']], 
                       [cm['FN'], cm['TP']]])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matriz, cmap='Blues', alpha=0.7)
    
    # Etiquetas
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Negativo (0)', 'Positivo (1)'])
    ax.set_yticklabels(['Negativo (0)', 'Positivo (1)'])
    ax.set_xlabel('Predicción', fontsize=12, fontweight='bold')
    ax.set_ylabel('Valor Real', fontsize=12, fontweight='bold')
    ax.set_title(titulo, fontsize=14, fontweight='bold')
    
    # Añadir valores en cada celda
    for i in range(2):
        for j in range(2):
            valor = matriz[i, j]
            color = 'white' if valor > matriz.max() / 2 else 'black'
            ax.text(j, i, f'{valor}', ha='center', va='center', 
                   color=color, fontsize=20, fontweight='bold')
    
    # Añadir etiquetas descriptivas
    ax.text(0, -0.3, 'TN (Verdadero Negativo)', ha='center', fontsize=9, style='italic')
    ax.text(1, -0.3, 'FP (Falso Positivo)', ha='center', fontsize=9, style='italic')
    ax.text(0, 2.3, 'FN (Falso Negativo)', ha='center', fontsize=9, style='italic')
    ax.text(1, 2.3, 'TP (Verdadero Positivo)', ha='center', fontsize=9, style='italic')
    
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    
    if guardar_como:
        plt.savefig(guardar_como, dpi=300, bbox_inches='tight')
        print(f"✓ Matriz de confusión guardada en: {guardar_como}")
    
    plt.close()


def graficar_comparacion_metricas(metricas_dict: Dict[str, Dict[str, float]], 
                                   guardar_como: str = None):
    """
    Grafica una comparación visual de las métricas de diferentes modelos.
    """
    modelos = list(metricas_dict.keys())
    accuracy_vals = [metricas_dict[m]['Accuracy'] for m in modelos]
    f1_vals = [metricas_dict[m]['F1-Score'] for m in modelos]
    precision_vals = [metricas_dict[m]['Precision'] for m in modelos]
    recall_vals = [metricas_dict[m]['Recall'] for m in modelos]
    
    x = np.arange(len(modelos))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    bars1 = ax.bar(x - 1.5*width, accuracy_vals, width, label='Accuracy', color='#2E86AB')
    bars2 = ax.bar(x - 0.5*width, f1_vals, width, label='F1-Score', color='#A23B72')
    bars3 = ax.bar(x + 0.5*width, precision_vals, width, label='Precision', color='#F18F01')
    bars4 = ax.bar(x + 1.5*width, recall_vals, width, label='Recall', color='#C73E1D')
    
    # Añadir valores sobre las barras
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Modelos', fontsize=12, fontweight='bold')
    ax.set_ylabel('Valor de Métrica', fontsize=12, fontweight='bold')
    ax.set_title('Comparación de Métricas de Rendimiento (Implementación Manual)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(modelos, rotation=15, ha='right')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if guardar_como:
        plt.savefig(guardar_como, dpi=300, bbox_inches='tight')
        print(f"✓ Comparación de métricas guardada en: {guardar_como}")
    
    plt.close()


def graficar_regiones_decision_knn(X, y, modelo_knn, features_names: List[str],
                                    guardar_como: str = None):
    """
    Grafica las regiones de decisión del modelo KNN en 2D.
    Usa las dos primeras features para visualización.
    """
    # Tomar solo las dos primeras columnas para visualización
    X_2d = X[:, :2] if len(X.shape) > 1 else X
    
    # Crear malla de puntos
    h = 0.02  # tamaño de paso en la malla
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Crear un array con las predicciones en cada punto de la malla
    # Necesitamos rellenar las otras features con valores medios
    n_features = X.shape[1]
    if n_features > 2:
        # Calcular medias para las features restantes
        other_features_mean = X[:, 2:].mean(axis=0)
        # Crear array completo para predicción
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        # Añadir las features restantes con sus valores medios
        mesh_full = np.column_stack([
            mesh_points,
            np.tile(other_features_mean, (mesh_points.shape[0], 1))
        ])
        Z = modelo_knn.predict(mesh_full)
    else:
        Z = modelo_knn.predict(np.c_[xx.ravel(), yy.ravel()])
    
    Z = Z.reshape(xx.shape)
    
    # Graficar
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Colorear las regiones de decisión
    contour = ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlGn', levels=[-0.5, 0.5, 1.5])
    
    # Graficar los puntos de entrenamiento
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y, 
                        cmap='RdYlGn', edgecolors='black', 
                        s=80, alpha=0.8, linewidths=1.5)
    
    ax.set_xlabel(f'{features_names[0]} (Escalado)', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{features_names[1]} (Escalado)', fontsize=12, fontweight='bold')
    ax.set_title(f'Regiones de Decisión KNN (k={modelo_knn.n_neighbors})\n'
                f'Basado en {features_names[0]} y {features_names[1]}',
                fontsize=14, fontweight='bold')
    
    # Leyenda
    legend_labels = ['Reprobado (0)', 'Aprobado (1)']
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                  markerfacecolor=scatter.cmap(scatter.norm(i)), 
                                  markersize=10, label=legend_labels[i])
                      for i in range(2)]
    ax.legend(handles=legend_elements, loc='best', fontsize=10)
    
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    if guardar_como:
        plt.savefig(guardar_como, dpi=300, bbox_inches='tight')
        print(f"✓ Regiones de decisión KNN guardadas en: {guardar_como}")
    
    plt.close()


# =====================================================
# FUNCIÓN PRINCIPAL DE EVALUACIÓN
# =====================================================

def evaluar_modelo_con_metricas_manuales(modelo, X_test, y_test, 
                                         nombre_modelo: str = "Modelo") -> Dict[str, float]:
    """
    Evalúa un modelo usando métricas implementadas manualmente.
    
    Args:
        modelo: Modelo entrenado con método predict()
        X_test: Features de prueba
        y_test: Etiquetas reales de prueba
        nombre_modelo: Nombre descriptivo del modelo
    
    Returns:
        Diccionario con todas las métricas calculadas
    """
    print(f"\n{'='*60}")
    print(f"EVALUACIÓN DEL MODELO: {nombre_modelo}")
    print(f"{'='*60}")
    
    # Realizar predicciones
    y_pred = modelo.predict(X_test)
    
    # Convertir a listas de Python nativos
    y_test_list = y_test.tolist() if hasattr(y_test, 'tolist') else list(y_test)
    y_pred_list = y_pred.tolist() if hasattr(y_pred, 'tolist') else list(y_pred)
    
    # Calcular matriz de confusión
    cm = calcular_matriz_confusion(y_test_list, y_pred_list)
    
    # Calcular métricas
    accuracy = calcular_accuracy(y_test_list, y_pred_list)
    precision = calcular_precision(y_test_list, y_pred_list)
    recall = calcular_recall(y_test_list, y_pred_list)
    f1 = calcular_f1_score(y_test_list, y_pred_list)
    
    # Mostrar resultados
    print("\n--- MATRIZ DE CONFUSIÓN ---")
    print(f"Verdaderos Positivos (TP): {cm['TP']}")
    print(f"Verdaderos Negativos (TN): {cm['TN']}")
    print(f"Falsos Positivos (FP):     {cm['FP']}")
    print(f"Falsos Negativos (FN):     {cm['FN']}")
    
    print("\n--- MÉTRICAS DE RENDIMIENTO (IMPLEMENTACIÓN MANUAL) ---")
    print(f"Accuracy (Exactitud):  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision:             {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall (Sensibilidad): {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1-Score:              {f1:.4f} ({f1*100:.2f}%)")
    
    print("\n--- INTERPRETACIÓN DE MÉTRICAS ---")
    print(f"• ACCURACY: Mide el porcentaje total de predicciones correctas.")
    print(f"  - {accuracy*100:.1f}% de las predicciones fueron correctas.")
    print(f"  - Útil cuando las clases están balanceadas.")
    
    print(f"\n• F1-SCORE: Media armónica entre Precision y Recall.")
    print(f"  - Valor de {f1:.4f} indica el balance entre detectar positivos correctamente")
    print(f"    y evitar falsos positivos.")
    print(f"  - Especialmente útil con clases desbalanceadas.")
    
    print(f"\n• PRECISION: De las predicciones positivas, ¿cuántas son correctas?")
    print(f"  - {precision*100:.1f}% de los casos predichos como positivos son realmente positivos.")
    
    print(f"\n• RECALL: De los casos positivos reales, ¿cuántos detectamos?")
    print(f"  - Detectamos {recall*100:.1f}% de todos los casos positivos reales.")
    
    # Graficar matriz de confusión
    nombre_archivo = nombre_modelo.lower().replace(' ', '_')
    graficar_matriz_confusion(cm, 
                              titulo=f"Matriz de Confusión - {nombre_modelo}",
                              guardar_como=f"matriz_confusion_{nombre_archivo}.png")
    
    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "TP": cm["TP"],
        "TN": cm["TN"],
        "FP": cm["FP"],
        "FN": cm["FN"]
    }


# =====================================================
# PIPELINE COMPLETO DE EVALUACIÓN
# =====================================================

def main():
    """
    Pipeline completo de evaluación de modelos con métricas manuales.
    """
    print("="*60)
    print("EVALUACIÓN DE RENDIMIENTO CON MÉTRICAS MANUALES")
    print("="*60)
    print("\nEste script implementa desde cero (sin sklearn.metrics):")
    print("  ✓ Accuracy (Exactitud)")
    print("  ✓ F1-Score")
    print("  ✓ Precision")
    print("  ✓ Recall")
    print("  ✓ Matriz de Confusión")
    print("\nSolo usa librerías de visualización (matplotlib).")
    print("="*60)
    
    # 1. CARGAR Y PREPARAR DATOS
    print("\n[1/5] Cargando dataset...")
    try:
        df = pd.read_csv('dataset_rendimiento_academico.csv')
        print(f"✓ Dataset cargado: {len(df)} registros")
    except FileNotFoundError:
        print("✗ Error: No se encontró 'dataset_rendimiento_academico.csv'")
        return
    
    # 2. PREPARACIÓN DE DATOS
    print("\n[2/5] Preparando datos para modelado...")
    
    # Crear variable objetivo binaria
    df['Estado_Final_Num'] = df['Estado_Final'].map({'Aprobado': 1, 'Reprobado': 0})
    
    # Features
    features = [
        'Actividad_Foros', 
        'Calidad_Foros', 
        'Tareas_Entregadas', 
        'Tiempo_Promedio_Respuesta', 
        'Interacciones_Semanales'
    ]
    
    X = df[features]
    y = df['Estado_Final_Num']
    
    print(f"✓ Features: {', '.join(features)}")
    print(f"✓ Target: Estado_Final (1=Aprobado, 0=Reprobado)")
    print(f"✓ Distribución de clases:")
    print(f"  - Aprobados: {sum(y == 1)} ({sum(y == 1)/len(y)*100:.1f}%)")
    print(f"  - Reprobados: {sum(y == 0)} ({sum(y == 0)/len(y)*100:.1f}%)")
    
    # Manejar desbalance si es necesario
    if sum(y == 0) < 5:  # Si hay muy pocos reprobados
        print("\n⚠ Advertencia: Clase minoritaria muy pequeña. Realizando sobre-muestreo...")
        # Sobre-muestreo simple
        df_minority = df[df['Estado_Final_Num'] == 0]
        df_majority = df[df['Estado_Final_Num'] == 1]
        
        n_samples = max(5, len(df_minority))
        df_minority_upsampled = df_minority.sample(n=n_samples, replace=True, random_state=42)
        
        df_balanced = pd.concat([df_majority, df_minority_upsampled])
        X = df_balanced[features]
        y = df_balanced['Estado_Final_Num']
        
        print(f"✓ Nueva distribución:")
        print(f"  - Aprobados: {sum(y == 1)}")
        print(f"  - Reprobados: {sum(y == 0)}")
    
    # 3. DIVISIÓN Y ESCALADO
    print("\n[3/5] Dividiendo datos y escalando...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"✓ Train set: {len(X_train)} muestras")
    print(f"✓ Test set:  {len(X_test)} muestras")
    print(f"✓ Datos escalados con StandardScaler")
    
    # 4. ENTRENAR MODELOS
    print("\n[4/5] Entrenando modelos...")
    
    # Modelo 1: Regresión Logística
    print("\n  → Entrenando Regresión Logística...")
    modelo_lr = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
    modelo_lr.fit(X_train_scaled, y_train)
    print("    ✓ Modelo entrenado")
    
    # Modelo 2: Árbol de Decisión
    print("\n  → Entrenando Árbol de Decisión...")
    modelo_dt = DecisionTreeClassifier(
        max_depth=4,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        class_weight='balanced'
    )
    modelo_dt.fit(X_train_scaled, y_train)
    print("    ✓ Modelo entrenado")
    
    # Modelo 3: KNN con solo Actividad_Foros y Calificacion_Promedio
    print("\n  → Entrenando KNN (K-Nearest Neighbors)...")
    print("    Usando solo: Actividad_Foros y Calificacion_Promedio")
    
    # Crear dataset con solo estas dos features
    features_knn = ['Actividad_Foros', 'Calificacion_Promedio']
    X_knn = df_balanced[features_knn]
    y_knn = df_balanced['Estado_Final_Num']
    
    # Split para KNN
    X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(
        X_knn, y_knn, test_size=0.3, random_state=42, stratify=y_knn
    )
    
    # Escalar
    scaler_knn = StandardScaler()
    X_train_knn_scaled = scaler_knn.fit_transform(X_train_knn)
    X_test_knn_scaled = scaler_knn.transform(X_test_knn)
    
    # Entrenar KNN
    # Probar con k=5 (un valor común)
    modelo_knn = KNeighborsClassifier(n_neighbors=5, weights='distance', metric='euclidean')
    modelo_knn.fit(X_train_knn_scaled, y_train_knn)
    print(f"    ✓ Modelo KNN entrenado (k=5, metric='euclidean', weights='distance')")
    
    # 5. EVALUAR MODELOS
    print("\n[5/6] Evaluando modelos con métricas manuales...")
    
    # Evaluar Regresión Logística
    metricas_lr = evaluar_modelo_con_metricas_manuales(
        modelo_lr, X_test_scaled, y_test, 
        nombre_modelo="Regresión Logística"
    )
    
    # Evaluar Árbol de Decisión
    metricas_dt = evaluar_modelo_con_metricas_manuales(
        modelo_dt, X_test_scaled, y_test,
        nombre_modelo="Árbol de Decisión"
    )
    
    # Evaluar KNN
    metricas_knn = evaluar_modelo_con_metricas_manuales(
        modelo_knn, X_test_knn_scaled, y_test_knn,
        nombre_modelo="KNN (Actividad Foros + Calificación)"
    )
    
    # Visualizar regiones de decisión del KNN
    print("\n[6/6] Generando visualización de regiones de decisión KNN...")
    graficar_regiones_decision_knn(
        X_train_knn_scaled, y_train_knn, modelo_knn, 
        features_knn,
        guardar_como="knn_regiones_decision.png"
    )
    
    # 6. COMPARACIÓN DE MODELOS
    print(f"\n{'='*70}")
    print("COMPARACIÓN DE MODELOS")
    print(f"{'='*70}")
    
    metricas_comparacion = {
        "Regresión Logística": metricas_lr,
        "Árbol de Decisión": metricas_dt,
        "KNN (Foros+Calif)": metricas_knn
    }
    
    # Tabla comparativa
    print("\n{:<25} {:<15} {:<15} {:<15}".format("Métrica", "Reg. Logística", "Árbol Decisión", "KNN"))
    print("-" * 70)
    for metrica in ["Accuracy", "F1-Score", "Precision", "Recall"]:
        val_lr = metricas_lr[metrica]
        val_dt = metricas_dt[metrica]
        val_knn = metricas_knn[metrica]
        
        # Determinar el mejor
        mejor_val = max(val_lr, val_dt, val_knn)
        marca_lr = "★" if val_lr == mejor_val else " "
        marca_dt = "★" if val_dt == mejor_val else " "
        marca_knn = "★" if val_knn == mejor_val else " "
        
        print("{:<25} {:<14.4f}{} {:<14.4f}{} {:<14.4f}{}".format(
            metrica, val_lr, marca_lr, val_dt, marca_dt, val_knn, marca_knn
        ))
    
    # Graficar comparación
    graficar_comparacion_metricas(
        metricas_comparacion,
        guardar_como="comparacion_metricas_modelos.png"
    )
    
    # 7. CONCLUSIONES
    print(f"\n{'='*70}")
    print("CONCLUSIONES")
    print(f"{'='*70}")
    
    # Determinar el mejor modelo para cada métrica
    modelos_nombres = ["Regresión Logística", "Árbol de Decisión", "KNN (Foros+Calif)"]
    metricas_lista = [metricas_lr, metricas_dt, metricas_knn]
    
    mejor_accuracy_idx = max(range(3), key=lambda i: metricas_lista[i]['Accuracy'])
    mejor_f1_idx = max(range(3), key=lambda i: metricas_lista[i]['F1-Score'])
    
    print(f"\n✓ Mejor ACCURACY: {modelos_nombres[mejor_accuracy_idx]} "
          f"({metricas_lista[mejor_accuracy_idx]['Accuracy']:.4f})")
    print(f"✓ Mejor F1-SCORE: {modelos_nombres[mejor_f1_idx]} "
          f"({metricas_lista[mejor_f1_idx]['F1-Score']:.4f})")
    
    print("\n--- Observaciones sobre KNN ---")
    print("• El modelo KNN usa SOLO dos features: Actividad_Foros y Calificacion_Promedio")
    print("• KNN clasifica basándose en la similitud con los k vecinos más cercanos")
    print(f"• Con k=5, el modelo examina los 5 estudiantes más similares para predecir")
    print("• El gráfico 'knn_regiones_decision.png' muestra cómo KNN divide el espacio")
    print("  de decisión basándose en estas dos variables.")
    
    print("\n--- ¿Cuál métrica usar? ---")
    print("• ACCURACY: Útil cuando las clases están balanceadas.")
    print("            Responde: ¿Qué % de predicciones fueron correctas?")
    print("\n• F1-SCORE: Mejor con clases desbalanceadas.")
    print("            Responde: ¿Qué tan bien balanceamos detectar positivos")
    print("            correctamente vs evitar falsos positivos?")
    
    print("\n--- Archivos generados ---")
    print("  ✓ matriz_confusion_regresion_logistica.png")
    print("  ✓ matriz_confusion_arbol_de_decision.png")
    print("  ✓ matriz_confusion_knn_(actividad_foros_+_calificación).png")
    print("  ✓ knn_regiones_decision.png")
    print("  ✓ comparacion_metricas_modelos.png")
    
    print(f"\n{'='*70}")
    print("EVALUACIÓN COMPLETADA")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
