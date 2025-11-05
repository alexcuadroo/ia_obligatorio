"""
Demo independiente de métricas puras.
- No usa pandas ni sklearn para calcular métricas.
- Solo usa matplotlib (opcional) para graficar si está disponible.

Además: lee el dataset 'dataset_rendimiento_academico.csv' para calcular
métricas de clasificación y regresión con predictores base (naive) y guarda
gráficos a archivos.
"""
from metrics_puros import (
    accuracy,
    precision_recall_f1_binary,
    confusion_matrix_binary,
    mae,
    rmse,
    r2,
    plot_confusion_matrix,
    plot_regression_diagnostics,
)
import csv
import os


def demo_con_dataset_csv():
    print("\n=== Métricas usando dataset CSV ===")
    script_dir = os.path.dirname(__file__)
    dataset_path = os.path.join(script_dir, 'dataset_rendimiento_academico.csv')
    if not os.path.exists(dataset_path):
        print(f"No se encontró el archivo: {dataset_path}")
        return

    rows = []
    with open(dataset_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    n = len(rows)
    print(f"Filas cargadas: {n}")
    if n == 0:
        print("Dataset vacío.")
        return

    # --- Clasificación binaria: Estado_Final (Aprobado=1, Reprobado=0) ---
    y_true_estado = [1 if r.get('Estado_Final') == 'Aprobado' else 0 for r in rows]
    # Baseline ingenuo: predecir siempre Aprobado
    y_pred_estado = [1] * n
    acc = accuracy(y_true_estado, y_pred_estado)
    prec, rec, f1 = precision_recall_f1_binary(y_true_estado, y_pred_estado)
    cm = confusion_matrix_binary(y_true_estado, y_pred_estado)
    print("\n[Clasificación - Estado_Final | baseline 'siempre Aprobado']")
    print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
    print(f"Matriz (TP,TN,FP,FN): {cm}")
    # Guardar matriz multiclase equivalente por conveniencia
    labels_mc = [0, 1]
    mat_mc = [[cm['TN'], cm['FP']], [cm['FN'], cm['TP']]]
    try:
        plot_confusion_matrix(labels_mc, mat_mc, title="Matriz Estado_Final (baseline)", save_path="csv_confusion_estado_baseline.png", show=False)
    except Exception as e:
        print(f"No se pudo graficar matriz de confusión: {e}")

    # --- Clasificación alternativa: Rendimiento_Bajo (Regular o Reprobado) ---
    y_true_bajo = [1 if (r.get('Nivel_Rendimiento') == 'Regular' or r.get('Estado_Final') == 'Reprobado') else 0 for r in rows]
    # Regla ingenua: bajo si Interacciones_Semanales == 0.0 o Tiempo_Promedio_Respuesta >= 10
    from typing import Any
    def to_float(v: Any, default: float = 0.0) -> float:
        try:
            return float(v)
        except Exception:
            return default
    y_pred_bajo = [1 if (to_float(r.get('Interacciones_Semanales', 0.0)) == 0.0 or to_float(r.get('Tiempo_Promedio_Respuesta', 0.0)) >= 10.0) else 0 for r in rows]
    acc_b = accuracy(y_true_bajo, y_pred_bajo)
    prec_b, rec_b, f1_b = precision_recall_f1_binary(y_true_bajo, y_pred_bajo)
    cm_b = confusion_matrix_binary(y_true_bajo, y_pred_bajo)
    print("\n[Clasificación - Rendimiento_Bajo (regla ingenua)]")
    print(f"Accuracy: {acc_b:.4f} | Precision: {prec_b:.4f} | Recall: {rec_b:.4f} | F1: {f1_b:.4f}")
    print(f"Matriz (TP,TN,FP,FN): {cm_b}")
    labels_b = [0, 1]
    mat_b = [[cm_b['TN'], cm_b['FP']], [cm_b['FN'], cm_b['TP']]]
    try:
        plot_confusion_matrix(labels_b, mat_b, title="Matriz Rendimiento_Bajo (regla)", save_path="csv_confusion_rend_bajo_regla.png", show=False)
    except Exception as e:
        print(f"No se pudo graficar matriz de confusión: {e}")

    # --- Regresión: Calificacion_Promedio con predictor media ---
    y_true_reg = [to_float(r.get('Calificacion_Promedio', 0.0)) for r in rows]
    mean_y = sum(y_true_reg) / len(y_true_reg)
    y_pred_reg = [mean_y] * n
    print("\n[Regresión - Calificacion_Promedio | baseline media]")
    print(f"MAE: {mae(y_true_reg, y_pred_reg):.4f}")
    print(f"RMSE: {rmse(y_true_reg, y_pred_reg):.4f}")
    print(f"R^2: {r2(y_true_reg, y_pred_reg):.4f}")
    try:
        plot_regression_diagnostics(y_true_reg, y_pred_reg, save_prefix="csv_regresion_media", show=False)
    except Exception as e:
        print(f"No se pudieron graficar diagnósticos de regresión: {e}")


if __name__ == "__main__":
    # Solo usamos el dataset CSV, sin datos de demostración
    demo_con_dataset_csv()
