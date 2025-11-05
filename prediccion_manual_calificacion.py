"""
Predicción de Calificación Promedio - Implementación Manual
============================================================

Este script implementa regresión lineal múltiple COMPLETAMENTE DESDE CERO
para predecir la calificación promedio de estudiantes.

NO SE USAN:
- sklearn (ningún modelo)
- numpy para cálculos (solo para estructuras de datos básicas)
- pandas para cálculos (solo para cargar datos)

SOLO SE USA:
- pandas: Para cargar el CSV
- matplotlib: Para visualizar resultados
- math: Para operaciones matemáticas básicas
"""

import pandas as pd
import matplotlib.pyplot as plt
import math


# ============================================================================
# FUNCIONES MATEMÁTICAS MANUALES
# ============================================================================

def calcular_media(datos):
    """Calcula la media (promedio) de una lista."""
    if len(datos) == 0:
        return 0.0
    return sum(datos) / len(datos)


def calcular_desviacion_estandar(datos):
    """Calcula la desviación estándar de una lista."""
    if len(datos) <= 1:
        return 0.0
    
    media = calcular_media(datos)
    suma_cuadrados = sum((x - media) ** 2 for x in datos)
    varianza = suma_cuadrados / (len(datos) - 1)
    return math.sqrt(varianza)


def estandarizar_datos(datos):
    """
    Estandariza datos: z = (x - media) / desviacion_estandar
    Retorna: (datos_estandarizados, media, desviacion)
    """
    media = calcular_media(datos)
    desv = calcular_desviacion_estandar(datos)
    
    if desv == 0:
        return [0.0] * len(datos), media, desv
    
    datos_std = [(x - media) / desv for x in datos]
    return datos_std, media, desv


def multiplicar_matrices(A, B):
    """
    Multiplica dos matrices A (m×n) y B (n×p).
    A y B son listas de listas.
    Retorna matriz resultado (m×p).
    """
    filas_A = len(A)
    cols_A = len(A[0]) if A else 0
    cols_B = len(B[0]) if B else 0
    
    resultado = []
    for i in range(filas_A):
        fila = []
        for j in range(cols_B):
            suma = 0.0
            for k in range(cols_A):
                suma += A[i][k] * B[k][j]
            fila.append(suma)
        resultado.append(fila)
    
    return resultado


def transponer_matriz(matriz):
    """Transpone una matriz (convierte filas en columnas)."""
    if not matriz:
        return []
    
    filas = len(matriz)
    cols = len(matriz[0])
    
    resultado = []
    for j in range(cols):
        fila = []
        for i in range(filas):
            fila.append(matriz[i][j])
        resultado.append(fila)
    
    return resultado


def invertir_matriz(matriz):
    """
    Invierte una matriz cuadrada usando eliminación de Gauss-Jordan.
    Retorna None si la matriz es singular (no invertible).
    """
    n = len(matriz)
    
    # Crear copia de la matriz y matriz identidad
    A = [fila[:] for fila in matriz]  # Copia profunda
    I = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    
    # Eliminación de Gauss-Jordan
    for i in range(n):
        # Buscar el pivote máximo
        max_fila = i
        for k in range(i + 1, n):
            if abs(A[k][i]) > abs(A[max_fila][i]):
                max_fila = k
        
        # Intercambiar filas
        A[i], A[max_fila] = A[max_fila], A[i]
        I[i], I[max_fila] = I[max_fila], I[i]
        
        # Si el pivote es cero, la matriz es singular
        if abs(A[i][i]) < 1e-10:
            return None
        
        # Escalar la fila del pivote
        pivote = A[i][i]
        for j in range(n):
            A[i][j] /= pivote
            I[i][j] /= pivote
        
        # Eliminar la columna del pivote en otras filas
        for k in range(n):
            if k != i:
                factor = A[k][i]
                for j in range(n):
                    A[k][j] -= factor * A[i][j]
                    I[k][j] -= factor * I[i][j]
    
    return I


# ============================================================================
# REGRESIÓN LINEAL MÚLTIPLE MANUAL
# ============================================================================

class RegresionLinealManual:
    """
    Implementación manual de Regresión Lineal Múltiple.
    
    Usa la fórmula de mínimos cuadrados ordinarios:
    β = (X^T X)^(-1) X^T y
    
    Donde:
    - X: matriz de features (con columna de 1's para el intercepto)
    - y: vector de valores objetivo
    - β: coeficientes (intercepto + pesos)
    """
    
    def __init__(self):
        self.coeficientes = None
        self.intercepto = None
        self.nombres_features = None
        self.medias_X = None
        self.desv_X = None
        self.media_y = None
        self.desv_y = None
    
    def entrenar(self, X, y, nombres_features=None, estandarizar=True):
        """
        Entrena el modelo de regresión lineal.
        
        Args:
            X: Lista de listas (filas=observaciones, columnas=features)
            y: Lista de valores objetivo
            nombres_features: Lista con nombres de las features
            estandarizar: Si True, estandariza los datos antes de entrenar
        """
        n_muestras = len(X)
        n_features = len(X[0])
        
        self.nombres_features = nombres_features or [f"X{i}" for i in range(n_features)]
        
        # Estandarizar datos si se solicita
        if estandarizar:
            print("Estandarizando datos...")
            X_std = []
            self.medias_X = []
            self.desv_X = []
            
            # Estandarizar cada feature
            for j in range(n_features):
                columna = [X[i][j] for i in range(n_muestras)]
                col_std, media, desv = estandarizar_datos(columna)
                
                if j == 0:
                    X_std = [[val] for val in col_std]
                else:
                    for i in range(n_muestras):
                        X_std[i].append(col_std[i])
                
                self.medias_X.append(media)
                self.desv_X.append(desv)
            
            # Estandarizar y
            y_std, self.media_y, self.desv_y = estandarizar_datos(y)
            
            X_trabajo = X_std
            y_trabajo = y_std
            print("  ✓ Datos estandarizados")
        else:
            X_trabajo = X
            y_trabajo = y
        
        # Añadir columna de 1's para el intercepto
        X_con_intercepto = []
        for i in range(n_muestras):
            fila = [1.0] + X_trabajo[i]
            X_con_intercepto.append(fila)
        
        print("\nCalculando coeficientes usando (X^T X)^(-1) X^T y...")
        
        # Calcular X^T (transpuesta)
        X_T = transponer_matriz(X_con_intercepto)
        
        # Calcular X^T X
        XTX = multiplicar_matrices(X_T, X_con_intercepto)
        
        # Calcular (X^T X)^(-1)
        XTX_inv = invertir_matriz(XTX)
        
        if XTX_inv is None:
            print("ERROR: La matriz X^T X es singular (no invertible)")
            print("Esto puede ocurrir si hay features perfectamente correlacionadas.")
            return False
        
        # Calcular X^T y
        y_matriz = [[val] for val in y_trabajo]  # Convertir a matriz columna
        XTy = multiplicar_matrices(X_T, y_matriz)
        
        # Calcular β = (X^T X)^(-1) X^T y
        beta = multiplicar_matrices(XTX_inv, XTy)
        
        # Extraer coeficientes
        self.intercepto = beta[0][0]
        self.coeficientes = [beta[i][0] for i in range(1, len(beta))]
        
        print("  ✓ Coeficientes calculados exitosamente")
        return True
    
    def predecir(self, X):
        """
        Realiza predicciones para nuevos datos.
        
        Args:
            X: Lista de listas con features
        
        Returns:
            Lista con predicciones
        """
        if self.coeficientes is None:
            print("ERROR: El modelo no ha sido entrenado")
            return None
        
        predicciones = []
        
        for muestra in X:
            # Estandarizar si el modelo fue entrenado con estandarización
            if self.medias_X is not None:
                muestra_std = []
                for j, val in enumerate(muestra):
                    if self.desv_X[j] == 0:
                        muestra_std.append(0.0)
                    else:
                        muestra_std.append((val - self.medias_X[j]) / self.desv_X[j])
                muestra_trabajo = muestra_std
            else:
                muestra_trabajo = muestra
            
            # Calcular predicción: y = β0 + β1*x1 + β2*x2 + ...
            pred = self.intercepto
            for i, coef in enumerate(self.coeficientes):
                pred += coef * muestra_trabajo[i]
            
            # Des-estandarizar predicción si es necesario
            if self.media_y is not None and self.desv_y != 0:
                pred = pred * self.desv_y + self.media_y
            
            predicciones.append(pred)
        
        return predicciones
    
    def mostrar_coeficientes(self):
        """Muestra los coeficientes del modelo de forma legible."""
        if self.coeficientes is None:
            print("El modelo no ha sido entrenado")
            return
        
        print("\n" + "="*70)
        print("COEFICIENTES DEL MODELO")
        print("="*70)
        print(f"\nIntercepto (β0): {self.intercepto:.6f}")
        print("\nCoeficientes de las features:")
        print("-" * 50)
        
        for i, (nombre, coef) in enumerate(zip(self.nombres_features, self.coeficientes)):
            signo = "+" if coef >= 0 else ""
            print(f"  {nombre:<30} (β{i+1}): {signo}{coef:.6f}")
        
        print("\nEcuación del modelo:")
        ecuacion = f"y = {self.intercepto:.4f}"
        for nombre, coef in zip(self.nombres_features, self.coeficientes):
            signo = "+" if coef >= 0 else ""
            ecuacion += f" {signo} {coef:.4f}*{nombre}"
        print(f"  {ecuacion}")
        print("="*70)


# ============================================================================
# MÉTRICAS DE EVALUACIÓN MANUALES
# ============================================================================

def calcular_mae(y_real, y_pred):
    """Mean Absolute Error: promedio de errores absolutos."""
    errores = [abs(y_real[i] - y_pred[i]) for i in range(len(y_real))]
    return calcular_media(errores)


def calcular_rmse(y_real, y_pred):
    """Root Mean Squared Error: raíz del promedio de errores al cuadrado."""
    errores_cuadrados = [(y_real[i] - y_pred[i]) ** 2 for i in range(len(y_real))]
    mse = calcular_media(errores_cuadrados)
    return math.sqrt(mse)


def calcular_r2(y_real, y_pred):
    """
    R² (coeficiente de determinación): proporción de varianza explicada.
    R² = 1 - (SS_res / SS_tot)
    """
    media_y = calcular_media(y_real)
    
    # SS_res: suma de cuadrados residuales
    ss_res = sum((y_real[i] - y_pred[i]) ** 2 for i in range(len(y_real)))
    
    # SS_tot: suma total de cuadrados
    ss_tot = sum((y_real[i] - media_y) ** 2 for i in range(len(y_real)))
    
    if ss_tot == 0:
        return 0.0
    
    return 1.0 - (ss_res / ss_tot)


# ============================================================================
# DIVISIÓN DE DATOS MANUAL (TRAIN/TEST SPLIT)
# ============================================================================

def dividir_datos(X, y, proporcion_test=0.3, semilla=42):
    """
    Divide datos en conjunto de entrenamiento y prueba.
    
    Args:
        X: Lista de listas (features)
        y: Lista (valores objetivo)
        proporcion_test: Proporción para el conjunto de prueba (0.0 a 1.0)
        semilla: Semilla para reproducibilidad
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    import random
    random.seed(semilla)
    
    n = len(X)
    indices = list(range(n))
    random.shuffle(indices)
    
    n_test = int(n * proporcion_test)
    n_train = n - n_test
    
    indices_train = indices[:n_train]
    indices_test = indices[n_train:]
    
    X_train = [X[i] for i in indices_train]
    X_test = [X[i] for i in indices_test]
    y_train = [y[i] for i in indices_train]
    y_test = [y[i] for i in indices_test]
    
    return X_train, X_test, y_train, y_test


# ============================================================================
# PROGRAMA PRINCIPAL
# ============================================================================

def main():
    print("="*80)
    print("PREDICCIÓN MANUAL DE CALIFICACIÓN PROMEDIO")
    print("Regresión Lineal Múltiple - Implementación desde cero")
    print("="*80)
    
    # Cargar datos
    print("\n[1] Cargando datos...")
    df = pd.read_csv('dataset_rendimiento_academico.csv')
    print(f"  ✓ Datos cargados: {len(df)} registros")
    
    # Seleccionar features y target
    features_nombres = [
        'Tiempo_Promedio_Respuesta'
    ]
    target_nombre = 'Calificacion_Promedio'
    
    print(f"\n[2] Features seleccionadas:")
    for i, f in enumerate(features_nombres, 1):
        print(f"  {i}. {f}")
    print(f"\nTarget: {target_nombre}")
    
    # Convertir a listas (sin pandas)
    X = []
    for _, fila in df.iterrows():
        muestra = [fila[f] for f in features_nombres]
        X.append(muestra)
    
    y = df[target_nombre].tolist()
    
    # Dividir en train/test
    print(f"\n[3] Dividiendo datos (70% entrenamiento, 30% prueba)...")
    X_train, X_test, y_train, y_test = dividir_datos(X, y, proporcion_test=0.3, semilla=42)
    print(f"  ✓ Entrenamiento: {len(X_train)} muestras")
    print(f"  ✓ Prueba: {len(X_test)} muestras")
    
    # Entrenar modelo
    print(f"\n[4] Entrenando modelo de regresión lineal múltiple...")
    modelo = RegresionLinealManual()
    exito = modelo.entrenar(X_train, y_train, nombres_features=features_nombres, estandarizar=True)
    
    if not exito:
        print("ERROR: No se pudo entrenar el modelo")
        return
    
    # Mostrar coeficientes
    modelo.mostrar_coeficientes()
    
    # Realizar predicciones
    print(f"\n[5] Realizando predicciones en conjunto de prueba...")
    y_pred_test = modelo.predecir(X_test)
    y_pred_train = modelo.predecir(X_train)
    print(f"  ✓ Predicciones completadas")
    
    # Evaluar modelo
    print(f"\n[6] Evaluando modelo...")
    print("\n" + "="*70)
    print("RESULTADOS - CONJUNTO DE ENTRENAMIENTO")
    print("="*70)
    mae_train = calcular_mae(y_train, y_pred_train)
    rmse_train = calcular_rmse(y_train, y_pred_train)
    r2_train = calcular_r2(y_train, y_pred_train)
    print(f"MAE (Error Absoluto Medio):     {mae_train:.4f}")
    print(f"RMSE (Raíz del Error Cuadrático): {rmse_train:.4f}")
    print(f"R² (Coef. Determinación):       {r2_train:.4f}")
    
    print("\n" + "="*70)
    print("RESULTADOS - CONJUNTO DE PRUEBA")
    print("="*70)
    mae_test = calcular_mae(y_test, y_pred_test)
    rmse_test = calcular_rmse(y_test, y_pred_test)
    r2_test = calcular_r2(y_test, y_pred_test)
    print(f"MAE (Error Absoluto Medio):     {mae_test:.4f}")
    print(f"RMSE (Raíz del Error Cuadrático): {rmse_test:.4f}")
    print(f"R² (Coef. Determinación):       {r2_test:.4f}")
    print("="*70)
    
    # Interpretación de R²
    print("\n[7] Interpretación de resultados:")
    print(f"\nR² = {r2_test:.4f} significa que el modelo explica el {r2_test*100:.2f}%")
    print("de la variabilidad en las calificaciones.")
    
    if r2_test < 0.3:
        print("→ Ajuste DÉBIL: Muchos otros factores influyen en la calificación")
    elif r2_test < 0.7:
        print("→ Ajuste MODERADO: El modelo captura algunos patrones importantes")
    else:
        print("→ Ajuste FUERTE: El modelo predice bien las calificaciones")
    
    # Visualizaciones
    print(f"\n[8] Generando visualizaciones...")
    
    # Gráfico 1: Predicciones vs Valores Reales
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Scatter plot
    ax1.scatter(y_test, y_pred_test, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    
    # Línea diagonal (predicción perfecta)
    min_val = min(min(y_test), min(y_pred_test))
    max_val = max(max(y_test), max(y_pred_test))
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Predicción Perfecta')
    
    ax1.set_xlabel('Calificación Real', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Calificación Predicha', fontsize=11, fontweight='bold')
    ax1.set_title(f'Predicciones vs Reales\n(R² = {r2_test:.3f})', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend()
    
    # Gráfico 2: Distribución de errores
    errores = [y_test[i] - y_pred_test[i] for i in range(len(y_test))]
    ax2.hist(errores, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Error = 0')
    ax2.set_xlabel('Error (Real - Predicho)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Frecuencia', fontsize=11, fontweight='bold')
    ax2.set_title(f'Distribución de Errores\n(MAE = {mae_test:.3f})', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('prediccion_manual_resultados.png', dpi=150, bbox_inches='tight')
    print("  ✓ Gráfico guardado: 'prediccion_manual_resultados.png'")
    
    # Gráfico 3: Importancia de features (valor absoluto de coeficientes)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    coefs_abs = [abs(c) for c in modelo.coeficientes]
    colores = ['green' if c > 0 else 'red' for c in modelo.coeficientes]
    
    barras = ax.barh(features_nombres, coefs_abs, color=colores, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Magnitud del Coeficiente (Estandarizado)', fontsize=11, fontweight='bold')
    ax.set_title('Importancia de Features en la Predicción\n(Verde=Positivo, Rojo=Negativo)', 
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', axis='x')
    
    plt.tight_layout()
    plt.savefig('prediccion_manual_importancia.png', dpi=150, bbox_inches='tight')
    print("  ✓ Gráfico guardado: 'prediccion_manual_importancia.png'")
    
    # Ejemplo de predicción manual
    print("\n" + "="*70)
    print("[9] EJEMPLO DE PREDICCIÓN MANUAL")
    print("="*70)
    print("\nSupongamos un estudiante con estas características:")
    ejemplo = {
        'Actividad_Foros': 15,
        'Calidad_Foros': 3.5,
        'Tareas_Entregadas': 8,
        'Tiempo_Promedio_Respuesta': 2.5,
        'Interacciones_Semanales': 5.0
    }
    
    for feat, val in ejemplo.items():
        print(f"  • {feat}: {val}")
    
    X_ejemplo = [[ejemplo[f] for f in features_nombres]]
    pred_ejemplo = modelo.predecir(X_ejemplo)
    
    print(f"\nCalificación predicha: {pred_ejemplo[0]:.2f}")
    print("="*70)
    
    print("\n✓ ANÁLISIS COMPLETO FINALIZADO")
    print("  Archivos generados:")
    print("  - prediccion_manual_resultados.png")
    print("  - prediccion_manual_importancia.png")


if __name__ == "__main__":
    main()
