# modules/optimizer.py

import pandas as pd
import numpy as np
from scipy.optimize import minimize

# =============================
# Función principal de optimización ALM
# =============================
def run_optimization(df, tasa_objetivo, liquidez_minima):
    # Separar el DataFrame en activos y pasivos
    activos = df[df['Tipo'] == 'Activo'].copy()
    pasivos = df[df['Tipo'] == 'Pasivo'].copy()

    # Crear vector inicial de proporciones igualadas
    n = len(df)
    x0 = np.ones(n) / n

    # Limitar todas las proporciones entre 0% y 100%
    bounds = [(0, 1) for _ in range(n)]

    # --------------------------------------------
    # Función objetivo: minimizar diferencia de duración entre activos y pasivos
    # --------------------------------------------
    def objective(x):
        activos_sel = x[:len(activos)]
        pasivos_sel = x[len(activos):]

        dur_act = np.sum(activos_sel * activos['Monto (USD M)'] * activos['Duración (años)']) / np.sum(activos_sel * activos['Monto (USD M)'])
        dur_pas = np.sum(pasivos_sel * pasivos['Monto (USD M)'] * pasivos['Duración (años)']) / np.sum(pasivos_sel * pasivos['Monto (USD M)'])
        return abs(dur_act - dur_pas)

    # --------------------------------------------
    # Restricción 1: balance entre activos y pasivos
    # --------------------------------------------
    def constraint_balance(x):
        activos_sel = x[:len(activos)]
        pasivos_sel = x[len(activos):]
        return np.sum(activos_sel * activos['Monto (USD M)']) - np.sum(pasivos_sel * pasivos['Monto (USD M)'])

    # --------------------------------------------
    # Restricción 2: liquidez mínima en efectivo
    # --------------------------------------------
    def constraint_liquidez(x):
        activos_sel = x[:len(activos)]
        efectivo_idx = activos[activos['Categoría'] == 'Efectivo'].index[0]
        return x[efectivo_idx] * activos.loc[efectivo_idx, 'Monto (USD M)'] - liquidez_minima

    # --------------------------------------------
    # Restricción 3: tasa objetivo del portafolio
    # --------------------------------------------
    def constraint_tasa(x):
        activos_sel = x[:len(activos)]
        tasa_port = np.sum(activos_sel * activos['Monto (USD M)'] * activos['Tasa (%)']) / np.sum(activos_sel * activos['Monto (USD M)'])
        return tasa_port - tasa_objetivo

    # Agrupación de todas las restricciones
    constraints = [
        {'type': 'eq', 'fun': constraint_balance},
        {'type': 'ineq', 'fun': constraint_liquidez},
        {'type': 'eq', 'fun': constraint_tasa}
    ]

    # Resolver el problema de optimización
    result = minimize(objective, x0, bounds=bounds, constraints=constraints)

    # --------------------------------------------
    # Resultado exitoso: construir DataFrame de salida
    # --------------------------------------------
    if result.success:
        x_opt = result.x
        df_resultado = df.copy()

        df_resultado['Asignación Óptima (%)'] = x_opt * 100
        df_resultado['Valor Asignado (USD M)'] = df_resultado['Asignación Óptima (%)'] * df_resultado['Monto (USD M)'] / 100

        # Crear resumen con métricas clave
        resumen = {
            'Duración Activos': round(objective(x_opt), 2),
            'Liquidez Disponible (USD M)': round(x_opt[activos[activos['Categoría'] == 'Efectivo'].index[0]] * activos[activos['Categoría'] == 'Efectivo']['Monto (USD M)'].values[0], 2),
            'Tasa Promedio del Portafolio (%)': round(np.sum(x_opt[:len(activos)] * activos['Monto (USD M)'] * activos['Tasa (%)']) / np.sum(x_opt[:len(activos)] * activos['Monto (USD M)']), 2),
            'Duración Promedio Activos (años)': round(np.sum(x_opt[:len(activos)] * activos['Monto (USD M)'] * activos['Duración (años)']) / np.sum(x_opt[:len(activos)] * activos['Monto (USD M)']), 2),
            'Duración Promedio Pasivos (años)': round(np.sum(x_opt[len(activos):] * pasivos['Monto (USD M)'] * pasivos['Duración (años)']) / np.sum(x_opt[len(activos):] * pasivos['Monto (USD M)']), 2)
        }

        return df_resultado, resumen

    # --------------------------------------------
    # En caso de fallo: devolver mensaje de error
    # --------------------------------------------
    else:
        return df, {"error": "Optimización no exitosa. Ajusta las restricciones."}


# =============================
# Función para simular escenarios de tasas de interés
# =============================
def simular_escenario(df, cambio_tasa):
    df_simulado = df.copy()
    cols = df_simulado.columns

    # Determinar base de cálculo: monto original o valor asignado
    if 'Monto (USD M)' in cols:
        df_simulado['Tasa Simulada (%)'] = df_simulado['Tasa (%)'] + cambio_tasa
        df_simulado['Interés Estimado (USD M)'] = df_simulado['Monto (USD M)'] * df_simulado['Tasa Simulada (%)'] / 100
    elif 'Valor Asignado (USD M)' in cols:
        df_simulado['Tasa Simulada (%)'] = df_simulado['Tasa (%)'] + cambio_tasa
        df_simulado['Interés Estimado (USD M)'] = df_simulado['Valor Asignado (USD M)'] * df_simulado['Tasa Simulada (%)'] / 100

    # Asegurar existencia de columna 'Categoría'
    if 'Categoría' not in df_simulado.columns:
        df_simulado['Categoría'] = df_simulado.index.astype(str)

    # Crear columnas vacías si no existen (para evitar errores en el front-end)
    for col in ['Monto (USD M)', 'Tasa (%)', 'Tasa Simulada (%)', 'Interés Estimado (USD M)']:
        if col not in df_simulado.columns:
            df_simulado[col] = np.nan

    return df_simulado
