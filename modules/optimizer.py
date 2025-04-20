import pandas as pd
import numpy as np
from scipy.optimize import minimize

# =============================
# Función principal de optimización ALM
# =============================
def run_optimization(df, tasa_objetivo, liquidez_minima):
    activos = df[df['Tipo'] == 'Activo'].copy()
    pasivos = df[df['Tipo'] == 'Pasivo'].copy()
    n = len(df)
    x0 = np.ones(n) / n
    bounds = [(0, 1) for _ in range(n)]

    def objective(x):
        activos_sel = x[:len(activos)]
        pasivos_sel = x[len(activos):]
        dur_act = np.sum(activos_sel * activos['Monto (USD M)'] * activos['Duración (años)']) / np.sum(activos_sel * activos['Monto (USD M)'])
        dur_pas = np.sum(pasivos_sel * pasivos['Monto (USD M)'] * pasivos['Duración (años)']) / np.sum(pasivos_sel * pasivos['Monto (USD M)'])
        return abs(dur_act - dur_pas)

    def constraint_balance(x):
        activos_sel = x[:len(activos)]
        pasivos_sel = x[len(activos):]
        return np.sum(activos_sel * activos['Monto (USD M)']) - np.sum(pasivos_sel * pasivos['Monto (USD M)'])

    def constraint_liquidez(x):
        activos_sel = x[:len(activos)]
        efectivo_idx = activos[activos['Categoría'] == 'Efectivo'].index[0]
        return x[efectivo_idx] * activos.loc[efectivo_idx, 'Monto (USD M)'] - liquidez_minima

    def constraint_tasa(x):
        activos_sel = x[:len(activos)]
        tasa_port = np.sum(activos_sel * activos['Monto (USD M)'] * activos['Tasa (%)']) / np.sum(activos_sel * activos['Monto (USD M)'])
        return tasa_port - tasa_objetivo

    constraints = [
        {'type': 'eq', 'fun': constraint_balance},
        {'type': 'ineq', 'fun': constraint_liquidez},
        {'type': 'eq', 'fun': constraint_tasa}
    ]

    result = minimize(objective, x0, bounds=bounds, constraints=constraints)

    if result.success:
        x_opt = result.x
        df_resultado = df.copy()
        df_resultado['Asignación Óptima (%)'] = x_opt * 100
        df_resultado['Valor Asignado (USD M)'] = df_resultado['Asignación Óptima (%)'] * df_resultado['Monto (USD M)'] / 100

        dur_act = np.sum(x_opt[:len(activos)] * activos['Monto (USD M)'] * activos['Duración (años)']) / np.sum(x_opt[:len(activos)] * activos['Monto (USD M)'])
        dur_pas = np.sum(x_opt[len(activos):] * pasivos['Monto (USD M)'] * pasivos['Duración (años)']) / np.sum(x_opt[len(activos):] * pasivos['Monto (USD M)'])
        ibo = 1 - abs(dur_act - dur_pas) / dur_act

        # Cálculo del VaR simplificado (95% de confianza, distribución normal)
        rendimientos = activos['Tasa (%)'] / 100
        var_portafolio = np.std(rendimientos) * 1.65 * np.sum(x_opt[:len(activos)] * activos['Monto (USD M)'])

        resumen = {
            'Duración Activos': round(abs(dur_act - dur_pas), 2),
            'Liquidez Disponible (USD M)': round(x_opt[activos[activos['Categoría'] == 'Efectivo'].index[0]] * activos[activos['Categoría'] == 'Efectivo']['Monto (USD M)'].values[0], 2),
            'Tasa Promedio del Portafolio (%)': round(np.sum(x_opt[:len(activos)] * activos['Monto (USD M)'] * activos['Tasa (%)']) / np.sum(x_opt[:len(activos)] * activos['Monto (USD M)']), 2),
            'Duración Promedio Activos (años)': round(dur_act, 2),
            'Duración Promedio Pasivos (años)': round(dur_pas, 2),
            'Índice de Balance Óptimo (IBO)': round(ibo, 4),
            'Valor en Riesgo (VaR 95%) USD M': round(var_portafolio, 2)
        }

        return df_resultado, resumen
    else:
        return df, {"error": "Optimización no exitosa. Ajusta las restricciones."}


# =============================
# Simulación de tasas
# =============================
def simular_escenario(df, cambio_tasa):
    df_simulado = df.copy()
    cols = df_simulado.columns
    if 'Monto (USD M)' in cols:
        df_simulado['Tasa Simulada (%)'] = df_simulado['Tasa (%)'] + cambio_tasa
        df_simulado['Interés Estimado (USD M)'] = df_simulado['Monto (USD M)'] * df_simulado['Tasa Simulada (%)'] / 100
    elif 'Valor Asignado (USD M)' in cols:
        df_simulado['Tasa Simulada (%)'] = df_simulado['Tasa (%)'] + cambio_tasa
        df_simulado['Interés Estimado (USD M)'] = df_simulado['Valor Asignado (USD M)'] * df_simulado['Tasa Simulada (%)'] / 100

    if 'Categoría' not in df_simulado.columns:
        df_simulado['Categoría'] = df_simulado.index.astype(str)

    for col in ['Monto (USD M)', 'Tasa (%)', 'Tasa Simulada (%)', 'Interés Estimado (USD M)']:
        if col not in df_simulado.columns:
            df_simulado[col] = np.nan

    return df_simulado


# =============================
# Explicación de parámetros
# =============================
PARAM_DESCRIPTION = {
    "tasa_objetivo": "Esta es la tasa promedio que deseas alcanzar en tu portafolio de activos. Debe estar dentro del rango de tasas disponibles.",
    "liquidez_minima": "Monto mínimo de efectivo que quieres mantener disponible como colchón de liquidez."
}
