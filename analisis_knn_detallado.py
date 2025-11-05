"""
Análisis detallado del modelo KNN usando Actividad_Foros y Calificacion_Promedio.

Este script explora diferentes valores de k para encontrar el óptimo
y visualiza cómo cambian las métricas.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from typing import List, Dict


# =====================================================
# MÉTRICAS MANUALES (sin sklearn.metrics)
# =====================================================

def calcular_matriz_confusion(y_real: List[int], y_pred: List[int]) -> Dict[str, int]:
    """Calcula matriz de confusión manualmente."""
    tp = tn = fp = fn = 0
    for real, pred in zip(y_real, y_pred):
        if real == 1 and pred == 1:
            tp += 1
        elif real == 0 and pred == 0:
            tn += 1
        elif real == 0 and pred == 1:
            fp += 1
        elif real == 1 and pred == 0:
            fn += 1
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}


def calcular_accuracy(y_real: List[int], y_pred: List[int]) -> float:
    """Calcula accuracy manualmente."""
    if len(y_real) == 0:
        return 0.0
    aciertos = sum(1 for real, pred in zip(y_real, y_pred) if real == pred)
    return aciertos / len(y_real)


def calcular_f1_score(y_real: List[int], y_pred: List[int]) -> float:
    """Calcula F1-Score manualmente."""
    cm = calcular_matriz_confusion(y_real, y_pred)
    tp, fp, fn = cm["TP"], cm["FP"], cm["FN"]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return f1


# =====================================================
# ANÁLISIS DE KNN
# =====================================================

def evaluar_knn_diferentes_k(X_train, X_test, y_train, y_test, k_values: List[int]):
    """
    Evalúa KNN con diferentes valores de k y retorna las métricas.
    """
    resultados = []
    
    for k in k_values:
        # Entrenar modelo
        modelo = KNeighborsClassifier(n_neighbors=k, weights='distance', metric='euclidean')
        modelo.fit(X_train, y_train)
        
        # Predecir
        y_pred = modelo.predict(X_test)
        
        # Calcular métricas manualmente
        y_test_list = y_test.tolist() if hasattr(y_test, 'tolist') else list(y_test)
        y_pred_list = y_pred.tolist() if hasattr(y_pred, 'tolist') else list(y_pred)
        
        accuracy = calcular_accuracy(y_test_list, y_pred_list)
        f1 = calcular_f1_score(y_test_list, y_pred_list)
        
        resultados.append({
            'k': k,
            'accuracy': accuracy,
            'f1_score': f1
        })
        
    return resultados


def graficar_metricas_vs_k(resultados: List[Dict], guardar_como: str = None):
    """
    Grafica cómo varían las métricas con diferentes valores de k.
    """
    k_values = [r['k'] for r in resultados]
    accuracies = [r['accuracy'] for r in resultados]
    f1_scores = [r['f1_score'] for r in resultados]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Graficar métricas
    ax.plot(k_values, accuracies, marker='o', linewidth=2, markersize=8,
            label='Accuracy', color='#2E86AB')
    ax.plot(k_values, f1_scores, marker='s', linewidth=2, markersize=8,
            label='F1-Score', color='#A23B72')
    
    # Encontrar el mejor k para cada métrica
    mejor_k_acc = resultados[np.argmax(accuracies)]['k']
    mejor_k_f1 = resultados[np.argmax(f1_scores)]['k']
    
    # Marcar los mejores valores
    ax.axvline(x=mejor_k_acc, color='#2E86AB', linestyle='--', alpha=0.3, 
               label=f'Mejor k (Accuracy): {mejor_k_acc}')
    ax.axvline(x=mejor_k_f1, color='#A23B72', linestyle='--', alpha=0.3,
               label=f'Mejor k (F1): {mejor_k_f1}')
    
    ax.set_xlabel('Valor de k (Número de Vecinos)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Valor de Métrica', fontsize=12, fontweight='bold')
    ax.set_title('Rendimiento de KNN con Diferentes Valores de k\n'
                 'Features: Actividad_Foros + Calificacion_Promedio',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([0.85, 1.05])
    
    # Añadir anotaciones para los mejores valores
    max_acc = max(accuracies)
    max_f1 = max(f1_scores)
    
    idx_max_acc = accuracies.index(max_acc)
    idx_max_f1 = f1_scores.index(max_f1)
    
    ax.annotate(f'{max_acc:.4f}', 
                xy=(k_values[idx_max_acc], max_acc),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='#2E86AB', alpha=0.7),
                color='white', fontweight='bold', fontsize=10)
    
    ax.annotate(f'{max_f1:.4f}', 
                xy=(k_values[idx_max_f1], max_f1),
                xytext=(10, -20), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='#A23B72', alpha=0.7),
                color='white', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    
    if guardar_como:
        plt.savefig(guardar_como, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico guardado en: {guardar_como}")
    
    plt.close()


def visualizar_distribucion_datos(X, y, feature_names: List[str], guardar_como: str = None):
    """
    Visualiza la distribución de los datos en el espacio 2D.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Separar por clase
    X_clase_0 = X[y == 0]
    X_clase_1 = X[y == 1]
    
    # Graficar
    scatter_0 = ax.scatter(X_clase_0[:, 0], X_clase_0[:, 1], 
                          c='#E63946', s=100, alpha=0.6, 
                          edgecolors='black', linewidths=1.5,
                          label='Reprobado (0)', marker='x')
    
    scatter_1 = ax.scatter(X_clase_1[:, 0], X_clase_1[:, 1], 
                          c='#06D6A0', s=100, alpha=0.6,
                          edgecolors='black', linewidths=1.5,
                          label='Aprobado (1)', marker='o')
    
    ax.set_xlabel(f'{feature_names[0]} (Escalado)', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{feature_names[1]} (Escalado)', fontsize=12, fontweight='bold')
    ax.set_title('Distribución de Estudiantes en el Espacio de Features\n'
                 f'{feature_names[0]} vs {feature_names[1]}',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Añadir información de conteo
    n_clase_0 = len(X_clase_0)
    n_clase_1 = len(X_clase_1)
    ax.text(0.02, 0.98, f'Reprobados: {n_clase_0}\nAprobados: {n_clase_1}',
            transform=ax.transAxes, fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if guardar_como:
        plt.savefig(guardar_como, dpi=300, bbox_inches='tight')
        print(f"✓ Distribución guardada en: {guardar_como}")
    
    plt.close()


# =====================================================
# FUNCIÓN PRINCIPAL
# =====================================================

def main():
    """
    Pipeline completo de análisis KNN.
    """
    print("="*70)
    print("ANÁLISIS DETALLADO DE KNN")
    print("Features: Actividad_Foros + Calificacion_Promedio")
    print("="*70)
    
    # 1. CARGAR DATOS
    print("\n[1/4] Cargando dataset...")
    try:
        df = pd.read_csv('dataset_rendimiento_academico.csv')
        print(f"✓ Dataset cargado: {len(df)} registros")
    except FileNotFoundError:
        print("✗ Error: No se encontró 'dataset_rendimiento_academico.csv'")
        return
    
    # 2. PREPARAR DATOS
    print("\n[2/4] Preparando datos...")
    
    df['Estado_Final_Num'] = df['Estado_Final'].map({'Aprobado': 1, 'Reprobado': 0})
    
    # Usar solo Actividad_Foros y Calificacion_Promedio
    features = ['Actividad_Foros', 'Calificacion_Promedio']
    X = df[features]
    y = df['Estado_Final_Num']
    
    print(f"✓ Features seleccionadas: {', '.join(features)}")
    print(f"✓ Distribución original:")
    print(f"  - Aprobados: {sum(y == 1)} ({sum(y == 1)/len(y)*100:.1f}%)")
    print(f"  - Reprobados: {sum(y == 0)} ({sum(y == 0)/len(y)*100:.1f}%)")
    
    # Manejar desbalance
    if sum(y == 0) < 5:
        print("\n⚠ Sobre-muestreo de clase minoritaria...")
        df_minority = df[df['Estado_Final_Num'] == 0]
        df_majority = df[df['Estado_Final_Num'] == 1]
        
        df_minority_upsampled = df_minority.sample(n=5, replace=True, random_state=42)
        df_balanced = pd.concat([df_majority, df_minority_upsampled])
        
        X = df_balanced[features]
        y = df_balanced['Estado_Final_Num']
        
        print(f"✓ Nueva distribución:")
        print(f"  - Aprobados: {sum(y == 1)}")
        print(f"  - Reprobados: {sum(y == 0)}")
    
    # Split y escalado
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"✓ Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"✓ Datos escalados con StandardScaler")
    
    # 3. EVALUAR DIFERENTES VALORES DE K
    print("\n[3/4] Evaluando KNN con diferentes valores de k...")
    
    k_values = list(range(1, 21))  # Probar k de 1 a 20
    print(f"✓ Probando k valores: {min(k_values)} a {max(k_values)}")
    
    resultados = evaluar_knn_diferentes_k(
        X_train_scaled, X_test_scaled, y_train, y_test, k_values
    )
    
    # Mostrar tabla de resultados
    print("\n--- Resultados por valor de k ---")
    print(f"{'k':<5} {'Accuracy':<12} {'F1-Score':<12}")
    print("-" * 30)
    for r in resultados:
        print(f"{r['k']:<5} {r['accuracy']:<12.4f} {r['f1_score']:<12.4f}")
    
    # Encontrar mejores valores
    mejor_idx_acc = np.argmax([r['accuracy'] for r in resultados])
    mejor_idx_f1 = np.argmax([r['f1_score'] for r in resultados])
    
    print("\n--- Mejores valores ---")
    print(f"✓ Mejor k para Accuracy: {resultados[mejor_idx_acc]['k']} "
          f"(Accuracy: {resultados[mejor_idx_acc]['accuracy']:.4f})")
    print(f"✓ Mejor k para F1-Score: {resultados[mejor_idx_f1]['k']} "
          f"(F1: {resultados[mejor_idx_f1]['f1_score']:.4f})")
    
    # 4. GENERAR VISUALIZACIONES
    print("\n[4/4] Generando visualizaciones...")
    
    # Gráfico de métricas vs k
    graficar_metricas_vs_k(resultados, guardar_como="knn_metricas_vs_k.png")
    
    # Distribución de datos
    visualizar_distribucion_datos(
        X_train_scaled, y_train.values, features,
        guardar_como="knn_distribucion_datos.png"
    )
    
    print("\n--- Interpretación de KNN ---")
    print("• KNN es un algoritmo 'lazy' (perezoso): no aprende un modelo explícito.")
    print("• Para clasificar un nuevo punto, busca los k vecinos más cercanos.")
    print("• La distancia se calcula en el espacio euclidiano 2D (Foros, Calificación).")
    print("• Con weights='distance', los vecinos más cercanos tienen más influencia.")
    print("\n• Valores pequeños de k (ej. k=1):")
    print("  - Más sensible al ruido y outliers")
    print("  - Fronteras de decisión más complejas")
    print("\n• Valores grandes de k (ej. k=15):")
    print("  - Más robusto al ruido")
    print("  - Fronteras de decisión más suaves")
    print("  - Puede perder detalles importantes")
    
    print("\n--- Archivos generados ---")
    print("  ✓ knn_metricas_vs_k.png - Rendimiento con diferentes k")
    print("  ✓ knn_distribucion_datos.png - Distribución de puntos en 2D")
    
    print(f"\n{'='*70}")
    print("ANÁLISIS COMPLETADO")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
