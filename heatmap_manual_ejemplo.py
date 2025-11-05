"""
Ejemplo de cómo crear un heatmap de correlación completamente manual
(sin seaborn ni pandas.corr())
"""

import numpy as np
import matplotlib.pyplot as plt

def calcular_correlacion_pearson(x, y):
    """
    Calcula el coeficiente de correlación de Pearson entre dos arrays.
    
    Fórmula: r = Σ((xi - x̄)(yi - ȳ)) / sqrt(Σ(xi - x̄)² * Σ(yi - ȳ)²)
    """
    # Verificar que tengan la misma longitud
    assert len(x) == len(y), "Los arrays deben tener la misma longitud"
    
    n = len(x)
    if n == 0:
        return 0.0
    
    # Calcular medias
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    
    # Calcular desviaciones
    diff_x = [xi - mean_x for xi in x]
    diff_y = [yi - mean_y for yi in y]
    
    # Calcular productos y sumas
    numerador = sum(dx * dy for dx, dy in zip(diff_x, diff_y))
    suma_cuadrados_x = sum(dx * dx for dx in diff_x)
    suma_cuadrados_y = sum(dy * dy for dy in diff_y)
    
    denominador = (suma_cuadrados_x * suma_cuadrados_y) ** 0.5
    
    # Evitar división por cero
    if denominador == 0:
        return 0.0
    
    return numerador / denominador


def crear_matriz_correlacion(datos, nombres_columnas):
    """
    Crea una matriz de correlación completa desde datos en formato lista de listas.
    
    Args:
        datos: Lista de listas donde cada sublista es una columna
        nombres_columnas: Lista con los nombres de las columnas
    
    Returns:
        matriz: Array numpy con las correlaciones
        nombres: Lista con los nombres de las columnas
    """
    n_vars = len(datos)
    matriz = np.zeros((n_vars, n_vars))
    
    # Calcular correlación para cada par de variables
    for i in range(n_vars):
        for j in range(n_vars):
            if i == j:
                # Correlación de una variable consigo misma es 1
                matriz[i, j] = 1.0
            else:
                # Calcular correlación entre variable i y variable j
                matriz[i, j] = calcular_correlacion_pearson(datos[i], datos[j])
    
    return matriz, nombres_columnas


def dibujar_heatmap_manual(matriz, nombres, titulo='Heatmap de Correlación'):
    """
    Dibuja un heatmap manualmente sin usar seaborn.
    
    Args:
        matriz: Matriz de correlación (numpy array)
        nombres: Lista con nombres de variables
        titulo: Título del gráfico
    """
    n_vars = len(nombres)
    
    # Crear figura y ejes
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Dibujar el mapa de calor usando imshow
    # imshow es una función primitiva de matplotlib que muestra una matriz como imagen
    im = ax.imshow(matriz, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    
    # Configurar etiquetas de los ejes
    ax.set_xticks(np.arange(n_vars))
    ax.set_yticks(np.arange(n_vars))
    ax.set_xticklabels(nombres, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(nombres, fontsize=9)
    
    # Añadir grid (líneas entre celdas)
    ax.set_xticks(np.arange(n_vars + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_vars + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    
    # Añadir los valores numéricos en cada celda
    for i in range(n_vars):
        for j in range(n_vars):
            valor = matriz[i, j]
            # Elegir color de texto según intensidad del fondo
            color_texto = 'white' if abs(valor) > 0.5 else 'black'
            ax.text(j, i, f'{valor:.2f}', 
                   ha='center', va='center', 
                   color=color_texto, fontsize=9, fontweight='bold')
    
    # Añadir barra de color (colorbar)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Coeficiente de Correlación', rotation=270, labelpad=20)
    
    # Título
    ax.set_title(titulo, fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig, ax


# --- EJEMPLO DE USO ---
if __name__ == "__main__":
    # Datos de ejemplo (cada lista es una columna/variable)
    # Por ejemplo: datos de estudiantes
    actividad_foros = [5, 10, 15, 20, 8, 12, 18, 7, 14, 19]
    calidad_foros = [2.5, 3.0, 4.0, 4.5, 2.8, 3.2, 4.2, 2.6, 3.8, 4.3]
    tareas_entregadas = [6, 8, 9, 10, 7, 8, 9, 6, 9, 10]
    calificacion = [5.5, 6.5, 7.5, 8.5, 6.0, 6.8, 8.0, 5.8, 7.2, 8.2]
    
    # Organizar datos
    datos = [
        actividad_foros,
        calidad_foros,
        tareas_entregadas,
        calificacion
    ]
    
    nombres = [
        'Actividad_Foros',
        'Calidad_Foros',
        'Tareas_Entregadas',
        'Calificacion'
    ]
    
    # Crear matriz de correlación manualmente
    print("Calculando matriz de correlación manualmente...")
    matriz_corr, nombres_vars = crear_matriz_correlacion(datos, nombres)
    
    print("\nMatriz de Correlación:")
    print(matriz_corr)
    
    # Dibujar heatmap manual
    print("\nGenerando heatmap manual...")
    fig, ax = dibujar_heatmap_manual(matriz_corr, nombres_vars, 
                                      'Heatmap de Correlación (Implementación Manual)')
    
    plt.savefig('heatmap_manual_ejemplo.png', dpi=150, bbox_inches='tight')
    print("Heatmap guardado en 'heatmap_manual_ejemplo.png'")
    
    # Mostrar (opcional)
    # plt.show()
    
    print("\n--- Explicación de la implementación ---")
    print("1. calcular_correlacion_pearson(): Implementa la fórmula de Pearson desde cero")
    print("2. crear_matriz_correlacion(): Calcula todas las correlaciones entre pares de variables")
    print("3. dibujar_heatmap_manual(): Usa plt.imshow() para crear el mapa de calor")
    print("   - imshow() dibuja una matriz como imagen coloreada")
    print("   - Añadimos manualmente los textos con los valores")
    print("   - Configuramos los ticks, labels y colorbar")
