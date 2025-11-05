"""
Verificación: Comparar correlación manual vs pandas.corr() con el dataset real
Y generar scatter plot manual de Actividad_Foros vs Calificacion_Promedio
"""

import pandas as pd
import matplotlib.pyplot as plt

def calcular_correlacion_pearson(x, y):
    """
    Calcula el coeficiente de correlación de Pearson entre dos arrays.
    Fórmula: r = Σ((xi - x̄)(yi - ȳ)) / sqrt(Σ(xi - x̄)² * Σ(yi - ȳ)²)
    """
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
    
    if denominador == 0:
        return 0.0
    
    return numerador / denominador


# Cargar el dataset real
print("Cargando dataset real...")
df = pd.read_csv('dataset_rendimiento_academico.csv')

# Variables a comparar
variables = [
    'Actividad_Foros',
    'Calidad_Foros', 
    'Tareas_Entregadas',
    'Tiempo_Promedio_Respuesta',
    'Interacciones_Semanales'
]

target = 'Calificacion_Promedio'

print(f"\n{'='*70}")
print(f"COMPARACIÓN: Correlación Manual vs Pandas.corr()")
print(f"{'='*70}")
print(f"\n{'Variable':<30} {'Manual':<15} {'Pandas':<15} {'Diferencia':<15}")
print(f"{'-'*70}")

for var in variables:
    # Correlación MANUAL (nuestra implementación)
    x = df[var].tolist()
    y = df[target].tolist()
    corr_manual = calcular_correlacion_pearson(x, y)
    
    # Correlación PANDAS (referencia)
    corr_pandas = df[var].corr(df[target])
    
    # Diferencia
    diferencia = abs(corr_manual - corr_pandas)
    
    print(f"{var:<30} {corr_manual:>8.4f}      {corr_pandas:>8.4f}      {diferencia:>8.6f}")

print(f"{'-'*70}")

# Verificar que los valores coincidan con tu amigo
print(f"\n{'='*70}")
print("VALORES ESPERADOS (de tu amigo):")
print(f"{'='*70}")
valores_esperados = {
    'Actividad_Foros': 0.19,
    'Calidad_Foros': 0.18,
    'Tareas_Entregadas': 0.30,
    'Tiempo_Promedio_Respuesta': -0.16,
    'Interacciones_Semanales': 0.08
}

print(f"\n{'Variable':<30} {'Tu amigo':<15} {'Pandas':<15} {'Diferencia':<15}")
print(f"{'-'*70}")

for var, valor_esperado in valores_esperados.items():
    corr_pandas = df[var].corr(df[target])
    diferencia = abs(valor_esperado - corr_pandas)
    print(f"{var:<30} {valor_esperado:>8.2f}      {corr_pandas:>8.4f}      {diferencia:>8.4f}")

print(f"\n{'='*70}")
print("CONCLUSIÓN:")
print("- Si 'Manual' ≈ 'Pandas': El código funciona correctamente ✓")
print("- Si 'Pandas' ≈ 'Tu amigo': Tu amigo tiene razón ✓")
print("- Los datos de ejemplo eran artificiales, por eso correlaciones ~0.99")
print(f"{'='*70}")

# ===== GENERAR SCATTER PLOT MANUAL: Actividad_Foros vs Calificacion_Promedio =====
print("\n\nGenerando scatter plot manual: Actividad_Foros vs Calificacion_Promedio...")

# Extraer datos manualmente (sin pandas helpers)
x_data = df['Actividad_Foros'].tolist()
y_data = df['Calificacion_Promedio'].tolist()
estado_final = df['Estado_Final'].tolist()

# Separar datos por Estado_Final manualmente
x_aprobados = []
y_aprobados = []
x_reprobados = []
y_reprobados = []

for i in range(len(x_data)):
    if estado_final[i] == 'Aprobado':
        x_aprobados.append(x_data[i])
        y_aprobados.append(y_data[i])
    else:  # Reprobado
        x_reprobados.append(x_data[i])
        y_reprobados.append(y_data[i])

print(f"  - Estudiantes Aprobados: {len(x_aprobados)}")
print(f"  - Estudiantes Reprobados: {len(x_reprobados)}")

# Calcular correlación manual
corr_manual = calcular_correlacion_pearson(x_data, y_data)
print(f"  - Correlación (Pearson): {corr_manual:.4f}")

# Crear figura
fig, ax = plt.subplots(figsize=(10, 7))

# Dibujar puntos manualmente usando scatter (función básica de matplotlib)
# Aprobados en verde
ax.scatter(x_aprobados, y_aprobados, 
          c='green', alpha=0.6, s=80, 
          label='Aprobado', edgecolors='darkgreen', linewidth=0.5)

# Reprobados en rojo
ax.scatter(x_reprobados, y_reprobados, 
          c='red', alpha=0.6, s=80, 
          label='Reprobado', edgecolors='darkred', linewidth=0.5)

# Calcular línea de tendencia manual (regresión lineal simple)
# Fórmula: y = mx + b
# m = Σ((xi - x̄)(yi - ȳ)) / Σ(xi - x̄)²
# b = ȳ - m * x̄

n = len(x_data)
mean_x = sum(x_data) / n
mean_y = sum(y_data) / n

numerador = sum((x_data[i] - mean_x) * (y_data[i] - mean_y) for i in range(n))
denominador = sum((x_data[i] - mean_x) ** 2 for i in range(n))

if denominador != 0:
    m = numerador / denominador
    b = mean_y - m * mean_x
    
    # Calcular puntos de la línea de tendencia
    x_min = min(x_data)
    x_max = max(x_data)
    y_min_linea = m * x_min + b
    y_max_linea = m * x_max + b
    
    # Dibujar línea de tendencia
    ax.plot([x_min, x_max], [y_min_linea, y_max_linea], 
           'b--', linewidth=2, alpha=0.7, 
           label=f'Tendencia (y={m:.3f}x+{b:.2f})')
    
    print(f"  - Ecuación de tendencia: y = {m:.3f}x + {b:.2f}")

# Configurar etiquetas y título
ax.set_xlabel('Actividad en Foros (Cantidad de Participaciones)', fontsize=11, fontweight='bold')
ax.set_ylabel('Calificación Promedio', fontsize=11, fontweight='bold')
ax.set_title(f'Scatter Plot Manual: Actividad en Foros vs Calificación\n(Correlación = {corr_manual:.3f})', 
            fontsize=13, fontweight='bold', pad=15)

# Grid manual
ax.grid(True, linestyle='--', alpha=0.4, color='gray', linewidth=0.5)

# Leyenda
ax.legend(loc='best', fontsize=10, framealpha=0.9)

# Ajustar límites de los ejes manualmente
ax.set_xlim(min(x_data) - 1, max(x_data) + 1)
ax.set_ylim(min(y_data) - 0.5, max(y_data) + 0.5)

plt.tight_layout()
plt.savefig('scatter_manual_foros_vs_calificacion.png', dpi=150, bbox_inches='tight')
print("✓ Scatter plot guardado en: 'scatter_manual_foros_vs_calificacion.png'")

print("\n--- Interpretación ---")
print(f"La correlación de {corr_manual:.3f} indica una relación positiva DÉBIL.")
print("Esto significa que:")
print("  • Mayor actividad en foros tiende ligeramente a mejores calificaciones")
print("  • PERO hay mucha variabilidad (estudiantes activos con bajas notas y viceversa)")
print("  • La participación en foros NO es el único factor determinante del rendimiento")
