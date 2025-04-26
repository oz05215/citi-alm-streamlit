import pandas as pd
import numpy as np
from scipy.optimize import minimize

def run_optimization(
    df,
    tasa_objetivo,
    porcentaje_liquidez_objetivo,
    tolerancia_duracion=5.0,
    tolerancia_monto=5.0,
    tolerancias_categorias=None,
    optimizar_hacia_arriba=False,
    optimizar_hacia_abajo=False,
    max_multiplicador_categoria=2.0,
    max_concentracion_categoria=0.30,
    min_diversificacion_categoria=0.01
):
    activos = df[df['Tipo'] == 'Activo'].copy()
    pasivos = df[df['Tipo'] == 'Pasivo'].copy()
    total_original = df['Monto (USD M)'].sum()
    monto_liquidez_objetivo = porcentaje_liquidez_objetivo / 100 * activos['Monto (USD M)'].sum()

    n = len(df)

    # Configurar bounds dinámicos
    if optimizar_hacia_arriba:
        bounds = [(0, max_multiplicador_categoria) for _ in range(n)]
    else:
        bounds = [(0, 1) for _ in range(n)]

    x0 = np.ones(n) / n

    def objective(x):
        activos_sel = x[:len(activos)]
        pasivos_sel = x[len(activos):]

        dur_act = np.sum(activos_sel * activos['Monto (USD M)'] * activos['Duración (años)']) / np.sum(activos_sel * activos['Monto (USD M)'])
        dur_pas = np.sum(pasivos_sel * pasivos['Monto (USD M)'] * pasivos['Duración (años)']) / np.sum(pasivos_sel * pasivos['Monto (USD M)'])

        penalty_concentracion = 0
        penalty_diversificacion = 0

        pesos = (x * df['Monto (USD M)']) / np.sum(x * df['Monto (USD M)'])
        for peso in pesos:
            if peso > max_concentracion_categoria:
                penalty_concentracion += (peso - max_concentracion_categoria) ** 2
            if peso < min_diversificacion_categoria:
                penalty_diversificacion += (min_diversificacion_categoria - peso) ** 2

        return abs(dur_act - dur_pas) + 10 * penalty_concentracion + 5 * penalty_diversificacion

    constraints = []

    # Restricción de duración tolerada
    def constraint_duracion(x):
        activos_sel = x[:len(activos)]
        pasivos_sel = x[len(activos):]
        dur_act = np.sum(activos_sel * activos['Monto (USD M)'] * activos['Duración (años)']) / np.sum(activos_sel * activos['Monto (USD M)'])
        dur_pas = np.sum(pasivos_sel * pasivos['Monto (USD M)'] * pasivos['Duración (años)']) / np.sum(pasivos_sel * pasivos['Monto (USD M)'])
        return (tolerancia_duracion / 100) * dur_act - abs(dur_act - dur_pas)

    constraints.append({'type': 'ineq', 'fun': constraint_duracion})

    # Restricción de liquidez mínima
    def constraint_liquidez(x):
        efectivo_idx = activos[activos['Categoría'] == 'Efectivo'].index[0]
        return (x[efectivo_idx] * activos.loc[efectivo_idx, 'Monto (USD M)']) - monto_liquidez_objetivo

    constraints.append({'type': 'ineq', 'fun': constraint_liquidez})

    # Restricción de tasa promedio objetivo
    def constraint_tasa(x):
        activos_sel = x[:len(activos)]
        tasa_port = np.sum(activos_sel * activos['Monto (USD M)'] * activos['Tasa (%)']) / np.sum(activos_sel * activos['Monto (USD M)'])
        return tasa_port - tasa_objetivo

    constraints.append({'type': 'eq', 'fun': constraint_tasa})

    # Restricción de monto total tolerado
    def constraint_monto(x):
        return (tolerancia_monto / 100 * total_original) - abs(np.sum(x * df['Monto (USD M)']) - total_original)

    constraints.append({'type': 'ineq', 'fun': constraint_monto})

    # Subir total si es permitido
    if optimizar_hacia_arriba:
        def constraint_subir_total(x):
            return np.sum(x * df['Monto (USD M)']) - total_original
        constraints.append({'type': 'ineq', 'fun': constraint_subir_total})

    # Bajar total si es permitido
    if optimizar_hacia_abajo:
        def constraint_bajar_total(x):
            return total_original - np.sum(x * df['Monto (USD M)'])
        constraints.append({'type': 'ineq', 'fun': constraint_bajar_total})

    # Restricciones por categoría (si se dan tolerancias específicas)
    if tolerancias_categorias:
        for i, categoria in enumerate(df['Categoría']):
            if categoria in tolerancias_categorias:
                tol = tolerancias_categorias[categoria] / 100
                monto_original = df.loc[i, 'Monto (USD M)']

                def categoria_max(x, idx=i, tol=tol, monto_original=monto_original):
                    return (tol * monto_original) - abs(x[idx] * monto_original - monto_original)

                constraints.append({'type': 'ineq', 'fun': categoria_max})

    result = minimize(objective, x0, bounds=bounds, constraints=constraints)

    if result.success:
        x_opt = result.x
        df_resultado = df.copy()
        df_resultado['Asignación Óptima (%)'] = x_opt * 100
        df_resultado['Valor Asignado (USD M)'] = df_resultado['Asignación Óptima (%)'] * df_resultado['Monto (USD M)'] / 100

        dur_act = np.sum(x_opt[:len(activos)] * activos['Monto (USD M)'] * activos['Duración (años)']) / np.sum(x_opt[:len(activos)] * activos['Monto (USD M)'])
        dur_pas = np.sum(x_opt[len(activos):] * pasivos['Monto (USD M)'] * pasivos['Duración (años)']) / np.sum(x_opt[len(activos):] * pasivos['Monto (USD M)'])
        ibo = 1 - abs(dur_act - dur_pas) / dur_act

        liquidez_despues = df_resultado[df_resultado['Categoría'] == 'Efectivo']['Valor Asignado (USD M)'].sum()
        total_activos_despues = df_resultado[df_resultado['Tipo'] == 'Activo']['Valor Asignado (USD M)'].sum()
        liquidez_porcentaje_despues = (liquidez_despues / total_activos_despues) * 100 if total_activos_despues != 0 else 0

        rendimientos = activos['Tasa (%)'] / 100
        var_portafolio = np.std(rendimientos) * 1.65 * np.sum(x_opt[:len(activos)] * activos['Monto (USD M)'])

        tasa_prom_opt = np.sum(x_opt[:len(activos)] * activos['Monto (USD M)'] * activos['Tasa (%)']) / np.sum(x_opt[:len(activos)] * activos['Monto (USD M)'])
        ganancia_antes = np.sum(activos['Monto (USD M)']) * activos['Tasa (%)'].mean() / 100
        ganancia_despues = total_activos_despues * tasa_prom_opt / 100

        resumen = {
            'Duración Promedio Activos (años)': round(dur_act, 2),
            'Duración Promedio Pasivos (años)': round(dur_pas, 2),
            'Índice de Balance Óptimo (IBO)': round(ibo, 4),
            'Tasa Promedio del Portafolio (%)': round(tasa_prom_opt, 2),
            'Liquidez Disponible (USD M)': round(liquidez_despues, 2),
            'Liquidez % Después': round(liquidez_porcentaje_despues, 2),
            'Valor en Riesgo (VaR 95%) USD M': round(var_portafolio, 2),
            'Ganancia Estimada Antes (USD M)': round(ganancia_antes, 2),
            'Ganancia Estimada Después (USD M)': round(ganancia_despues, 2)
        }

        return df_resultado, resumen

    else:
        return df, {"error": "Optimización no exitosa. Ajusta restricciones o tolerancias."}


def check_feasibility(*args, **kwargs):
    _, resumen = run_optimization(*args, **kwargs)
    return "error" not in resumen

def simular_escenario(df, cambio_tasa):
    df_simulado = df.copy()
    if 'Monto (USD M)' in df_simulado.columns:
        df_simulado['Tasa Simulada (%)'] = df_simulado['Tasa (%)'] + cambio_tasa
        df_simulado['Interés Estimado (USD M)'] = df_simulado['Monto (USD M)'] * df_simulado['Tasa Simulada (%)'] / 100
    elif 'Valor Asignado (USD M)' in df_simulado.columns:
        df_simulado['Tasa Simulada (%)'] = df_simulado['Tasa (%)'] + cambio_tasa
        df_simulado['Interés Estimado (USD M)'] = df_simulado['Valor Asignado (USD M)'] * df_simulado['Tasa Simulada (%)'] / 100

    if 'Categoría' not in df_simulado.columns:
        df_simulado['Categoría'] = df_simulado.index.astype(str)

    return df_simulado

PARAM_DESCRIPTION = {
    "tasa_objetivo": "Tasa promedio objetivo del portafolio de activos.",
    "liquidez_minima": "Porcentaje mínimo de liquidez deseado respecto al total de activos."
}
