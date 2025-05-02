# optimizer.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize




def calcular_eve(df, tasa_base, shocks, columna_monto='Monto (USD B)'):
    activos = df[df['Tipo'] == 'Activo']
    pasivos = df[df['Tipo'] == 'Pasivo']
    
    resultados = []

    for shock in shocks:
        tasa_chocada = tasa_base + shock
        
        vp_activos = np.sum(activos[columna_monto] / (1 + tasa_chocada) ** activos['Duración (años)'])
        vp_pasivos = np.sum(pasivos[columna_monto] / (1 + tasa_chocada) ** pasivos['Duración (años)'])

        eve = vp_activos - vp_pasivos
        resultados.append({'Shock (%)': shock * 100, 'EVE (USD B)': eve})

    return pd.DataFrame(resultados)




def calcular_convexidad(df, tasas_cambio):
    activos = df[df['Tipo'] == 'Activo']
    convexidad = []

    # Detectar la columna de monto relevante
    if 'Valor Asignado (USD B)' in activos.columns:
        montos = activos['Valor Asignado (USD B)']
    else:
        montos = activos['Monto (USD B)']

    duraciones = activos['Duración (años)']
    
    for delta in tasas_cambio:
        cambio_precio = -duraciones * delta + 0.5 * (duraciones ** 2) * (delta ** 2)
        valor_cambiado = montos * (1 + cambio_precio)
        convexidad.append(valor_cambiado.sum())

    return tasas_cambio, convexidad



def graficar_convexidad(tasas_cambio, convexidad_antes, convexidad_despues=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(tasas_cambio, convexidad_antes, label='Antes de Optimizar', marker='o')
    if convexidad_despues is not None:
        ax.plot(tasas_cambio, convexidad_despues, label='Después de Optimizar', marker='o')

    # Línea horizontal en el valor base (cuando delta = 0)
    base_valor = convexidad_antes[np.where(tasas_cambio == 0)[0][0]]
    ax.axhline(y=base_valor, color='gray', linestyle='--', linewidth=1)
    ax.text(tasas_cambio[-1], base_valor, f"Valor Base ≈ {base_valor:.2f}B", va='bottom', ha='right', fontsize=9, color='gray')

    ax.set_title("Curva de Convexidad del Portafolio de Activos")
    ax.set_xlabel("Cambio en la Tasa (%)")
    ax.set_ylabel("Valor Total Estimado (USD B)")
    ax.legend()
    ax.grid(True)
    return fig



def run_optimization(
    df,
    tasa_objetivo,
    porcentaje_liquidez_objetivo,
    tolerancia_duracion=5.0,
    tolerancia_monto=5.0,
    tolerancias_Categorías=None,
    tolerancias_Subtipos=None,
    optimizar_hacia_arriba=False,
    optimizar_hacia_abajo=False,
    max_multiplicador_Categoría=2.0,
    max_concentracion_Categoría=0.30,
    min_diversificacion_Categoría=0.01,
    peso_concentracion=0.5,
    peso_diversificacion=0.25,
    peso_var=0.1,
    penalizar_concentracion=True,
    penalizar_diversificacion=True,
    considerar_riesgo=False,
    peso_riesgo=0.5,
    var_original=None,  # 👈 nuevo
    porcentaje_var_tolerado=None,  # 👈 nuevo
    cambio_tasa_simulacion=0.0,
):


    activos = df[df['Tipo'] == 'Activo'].copy()
    pasivos = df[df['Tipo'] == 'Pasivo'].copy()

    total_original = df['Monto (USD B)'].sum()
    monto_liquidez_objetivo = porcentaje_liquidez_objetivo / 100 * activos['Monto (USD B)'].sum()
    n = len(df)

    bounds = [(0, max_multiplicador_Categoría) for _ in range(n)] if optimizar_hacia_arriba else [(0, 1) for _ in range(n)]
    x0 = np.ones(n) / n


    var_original = np.std(activos['Tasa (%)'] / 100) * 1.65 * np.sum(activos['Monto (USD B)'])






    def objective(x):
        activos_sel = x[:len(activos)]
        pasivos_sel = x[len(activos):]

        # ──────────────────────────────
        # Cálculo de Duración (para IBO)
        # ──────────────────────────────
        dur_act = np.sum(activos_sel * activos['Monto (USD B)'] * activos['Duración (años)']) / np.sum(activos_sel * activos['Monto (USD B)'])
        dur_pas = np.sum(pasivos_sel * pasivos['Monto (USD B)'] * pasivos['Duración (años)']) / np.sum(pasivos_sel * pasivos['Monto (USD B)'])
        descalce_duracion = 0.5 * abs(dur_act - dur_pas)

        # ──────────────────────────────
        # Penalización Concentración / Diversificación
        # ──────────────────────────────
        penalty_concentracion = 0
        penalty_diversificacion = 0
        if penalizar_concentracion or penalizar_diversificacion:
            pesos = (x * df['Monto (USD B)']) / np.sum(x * df['Monto (USD B)'])
            for peso in pesos:
                if penalizar_concentracion and peso > max_concentracion_Categoría:
                    penalty_concentracion += (peso - max_concentracion_Categoría) ** 2
                if penalizar_diversificacion and peso < min_diversificacion_Categoría:
                    penalty_diversificacion += (min_diversificacion_Categoría - peso) ** 2

        # ──────────────────────────────
        # Penalización Riesgo Promedio
        # ──────────────────────────────
        penalty_riesgo = 0
        if considerar_riesgo and 'Peso de Riesgo' in df.columns:
            riesgo_promedio = np.sum(x * df['Monto (USD B)'] * df['Peso de Riesgo']) / np.sum(x * df['Monto (USD B)'])
            penalty_riesgo = peso_riesgo * riesgo_promedio

        # ──────────────────────────────
        # Penalización por pasivos caros y activos poco defensivos
        # ──────────────────────────────
        indices_pasivos_caros = pasivos[(pasivos['Tasa (%)'] > 4) & (pasivos['Duración (años)'] > 3)].index
        penalidad_pasivos_largos_caros = np.sum([pasivos_sel[i - len(activos)] for i in indices_pasivos_caros]) * 0.5

        indices_activos_debiles = activos[(activos['Duración (años)'] > 5) & (activos['Tasa (%)'] < 2)].index
        penalidad_activos_poco_defensivos = np.sum([activos_sel[i] for i in indices_activos_debiles]) * 0.5

        # ──────────────────────────────
        # Construcción base para simulaciones de tasa
        # ──────────────────────────────
        df_temp = df.copy()
        df_temp['Asignación Óptima (%)'] = x * 100
        df_temp['Valor Asignado (USD B)'] = df_temp['Asignación Óptima (%)'] / 100 * df_temp['Monto (USD B)']
        df_temp['Tasa Simulada (%)'] = df_temp['Tasa (%)']
        df_temp['Interés Estimado Base'] = df_temp['Valor Asignado (USD B)'] * df_temp['Tasa Simulada (%)'] / 100
        df_temp.loc[df_temp['Tipo'] == 'Pasivo', 'Interés Estimado Base'] *= -1
        impacto_base = df_temp['Interés Estimado Base'].sum()

        # ──────────────────────────────
        # Penalización por sensibilidad neta a tasas
        # ──────────────────────────────
        sensibilidad_score = 0
        ganancia_total = 0
        perdida_total = 0

        for delta in [-0.02, -0.01, 0.01, 0.02]:
            df_temp['Tasa Simulada (%)'] = df_temp['Tasa (%)'] + delta * 100
            df_temp['Interés Estimado Sim'] = df_temp['Valor Asignado (USD B)'] * df_temp['Tasa Simulada (%)'] / 100
            df_temp.loc[df_temp['Tipo'] == 'Pasivo', 'Interés Estimado Sim'] *= -1
            impacto_delta = df_temp['Interés Estimado Sim'].sum()

            if delta < 0:
                perdida_total += max(0, impacto_base - impacto_delta)
            else:
                ganancia_total += max(0, impacto_delta - impacto_base)

        sensibilidad_score = perdida_total - 1.25 * ganancia_total

        # ──────────────────────────────
        # Penalización fuerte si el EVE es negativo para cualquier shock
        # ──────────────────────────────
        penalidad_eve = 0
        for delta in [-0.02, -0.01, 0.01, 0.02]:
            tasa_chocada = df_temp['Tasa (%)'] + delta * 100
            df_temp['VP'] = df_temp['Valor Asignado (USD B)'] / (1 + tasa_chocada / 100) ** df_temp['Duración (años)']
            vp_act = df_temp[df_temp['Tipo'] == 'Activo']['VP'].sum()
            vp_pas = df_temp[df_temp['Tipo'] == 'Pasivo']['VP'].sum()
            eve = vp_act - vp_pas
            if eve < 0:
                penalidad_eve += abs(eve)

        # ──────────────────────────────
        # Penalización por impacto general (±bps simulados)
        # ──────────────────────────────
        df_temp['Tasa Simulada (%)'] = df['Tasa (%)'] + cambio_tasa_simulacion
        df_temp['Interés Estimado (USD B)'] = df_temp['Valor Asignado (USD B)'] * df_temp['Tasa Simulada (%)'] / 100
        df_temp.loc[df_temp['Tipo'] == 'Pasivo', 'Interés Estimado (USD B)'] *= -1
        impacto_despues = df_temp['Interés Estimado (USD B)'].sum()

        df_base = df.copy()
        df_base['Tasa Simulada (%)'] = df_base['Tasa (%)'] + cambio_tasa_simulacion
        df_base['Interés Estimado (USD B)'] = df_base['Monto (USD B)'] * df_base['Tasa Simulada (%)'] / 100
        df_base.loc[df_base['Tipo'] == 'Pasivo', 'Interés Estimado (USD B)'] *= -1
        impacto_antes = df_base['Interés Estimado (USD B)'].sum()

        penalidad_impacto = max(0, impacto_antes - impacto_despues)

        # ──────────────────────────────
        # Penalización por VaR
        # ──────────────────────────────
        var_activos = np.std(activos['Tasa (%)'] / 100)
        valor_total_activos = np.sum(activos_sel * activos['Monto (USD B)'])
        var_penalty = var_activos * 1.65 * valor_total_activos

        # ──────────────────────────────
        # Objetivo Total
        # ──────────────────────────────
        return (
            descalce_duracion
            + peso_concentracion * penalty_concentracion
            + peso_diversificacion * penalty_diversificacion
            + penalty_riesgo
            + 2.5 * sensibilidad_score
            + 1.0 * penalidad_eve
            + 1.0 * penalidad_impacto
            + 1.5 * penalidad_activos_poco_defensivos
            + 1.0 * penalidad_pasivos_largos_caros
            + peso_var * var_penalty
        )





    constraints = []

    def constraint_duracion(x):
        activos_sel = x[:len(activos)]
        pasivos_sel = x[len(activos):]
        dur_act = np.sum(activos_sel * activos['Monto (USD B)'] * activos['Duración (años)']) / np.sum(activos_sel * activos['Monto (USD B)'])
        dur_pas = np.sum(pasivos_sel * pasivos['Monto (USD B)'] * pasivos['Duración (años)']) / np.sum(pasivos_sel * pasivos['Monto (USD B)'])
        return (tolerancia_duracion / 100) * dur_act - abs(dur_act - dur_pas)

    constraints.append({'type': 'ineq', 'fun': constraint_duracion})

    def constraint_liquidez(x):
        activos_sel = x[:len(activos)]
        total_activos_opt = np.sum(activos_sel * activos['Monto (USD B)'])

        indices_efectivo = activos[activos['Categoría'] == 'Efectivo'].index
        monto_efectivo_opt = np.sum([x[i] * df.loc[i, 'Monto (USD B)'] for i in indices_efectivo])

        return monto_efectivo_opt - (porcentaje_liquidez_objetivo / 100) * total_activos_opt



    constraints.append({'type': 'ineq', 'fun': constraint_liquidez})

    def constraint_tasa(x):
        activos_sel = x[:len(activos)]
        tasa_port = np.sum(activos_sel * activos['Monto (USD B)'] * activos['Tasa (%)']) / np.sum(activos_sel * activos['Monto (USD B)'])
        return tasa_port - tasa_objetivo

    constraints.append({'type': 'eq', 'fun': constraint_tasa})

    def constraint_monto(x):
        return (tolerancia_monto / 100 * total_original) - abs(np.sum(x * df['Monto (USD B)']) - total_original)

    constraints.append({'type': 'ineq', 'fun': constraint_monto})


    if var_original is not None and porcentaje_var_tolerado is not None:
        def constraint_var(x):
            activos_sel = x[:len(activos)]
            valor_total_activos = np.sum(activos_sel * activos['Monto (USD B)'])
            var_despues = np.std(activos['Tasa (%)'] / 100) * 1.65 * valor_total_activos
            var_min = var_original * (1 + porcentaje_var_tolerado / 100)
            return var_min - var_despues

        constraints.append({'type': 'ineq', 'fun': constraint_var})



    constraints.append({'type': 'ineq', 'fun': constraint_var})


    if optimizar_hacia_arriba:
        def constraint_subir_total(x):
            return np.sum(x * df['Monto (USD B)']) - total_original
        constraints.append({'type': 'ineq', 'fun': constraint_subir_total})

    if optimizar_hacia_abajo:
        def constraint_bajar_total(x):
            return total_original - np.sum(x * df['Monto (USD B)'])
        constraints.append({'type': 'ineq', 'fun': constraint_bajar_total})

    if tolerancias_Categorías:
        for i, Categoría in enumerate(df['Categoría']):
            if Categoría in tolerancias_Categorías:
                tol = tolerancias_Categorías[Categoría] / 100
                monto_original = df.loc[i, 'Monto (USD B)']

                def Categoria_max(x, idx=i, tol=tol, monto_original=monto_original):
                    return (tol * monto_original) - abs(x[idx] * monto_original - monto_original)

                constraints.append({'type': 'ineq', 'fun': Categoria_max})


    if tolerancias_Subtipos:
        for i, Subtipo in enumerate(df['Subtipo']):
            if Subtipo in tolerancias_Subtipos:
                tol = tolerancias_Subtipos[Subtipo] / 100
                monto_original = df.loc[i, 'Monto (USD B)']

                def Subtipo_max(x, idx=i, tol=tol, monto_original=monto_original):
                    return (tol * monto_original) - abs(x[idx] * monto_original - monto_original)

                constraints.append({'type': 'ineq', 'fun': Subtipo_max})



    result = minimize(objective, x0, bounds=bounds, constraints=constraints)


    if result.success:
        x_opt = result.x
        df_resultado = df.copy()
        df_resultado['Asignación Óptima (%)'] = x_opt * 100
        df_resultado['Valor Asignado (USD B)'] = (df_resultado['Asignación Óptima (%)'] / 100) * df_resultado['Monto (USD B)']

        dur_act = np.sum(x_opt[:len(activos)] * activos['Monto (USD B)'] * activos['Duración (años)']) / np.sum(x_opt[:len(activos)] * activos['Monto (USD B)'])
        dur_pas = np.sum(x_opt[len(activos):] * pasivos['Monto (USD B)'] * pasivos['Duración (años)']) / np.sum(x_opt[len(activos):] * pasivos['Monto (USD B)'])
        ibo = 1 - abs(dur_act - dur_pas) / dur_act

        liquidez_despues = df_resultado[df_resultado['Categoría'] == 'Efectivo']['Valor Asignado (USD B)'].sum()
        total_activos_despues = df_resultado[df_resultado['Tipo'] == 'Activo']['Valor Asignado (USD B)'].sum()
        liquidez_porcentaje_despues = (liquidez_despues / total_activos_despues) * 100 if total_activos_despues != 0 else 0

        rendimientos = activos['Tasa (%)'] / 100
        var_portafolio = np.std(rendimientos) * 1.65 * np.sum(x_opt[:len(activos)] * activos['Monto (USD B)'])

        tasa_prom_opt = np.sum(x_opt[:len(activos)] * activos['Monto (USD B)'] * activos['Tasa (%)']) / np.sum(x_opt[:len(activos)] * activos['Monto (USD B)'])
        ganancia_antes = np.sum(activos['Monto (USD B)']) * activos['Tasa (%)'].mean() / 100
        ganancia_despues = total_activos_despues * tasa_prom_opt / 100

        resumen = {
            'Duración Promedio Activos (años)': round(dur_act, 2),
            'Duración Promedio Pasivos (años)': round(dur_pas, 2),
            'Índice de Balance Óptimo (IBO)': round(ibo, 4),
            'Indice de Balance Óptimo (IBO)': round(ibo, 4),
            'Tasa Promedio del Portafolio (%)': round(tasa_prom_opt, 2),
            'Liquidez Disponible (USD B)': round(liquidez_despues, 2),
            'Liquidez % Después': round(liquidez_porcentaje_despues, 2),
            'Valor en Riesgo (VaR 95%) USD B': round(var_portafolio, 2),
            'Ganancia Estimada Antes (USD B)': round(ganancia_antes, 2),
            'Ganancia Estimada Después (USD B)': round(ganancia_despues, 2)
        }

        return df_resultado, resumen

    else:
        return df, {"error": "Optimización no exitosa. Ajusta restricciones o tolerancias."}

def simular_escenario(df_antes, df_despues, cambio_tasa):
    df_sim_antes = df_antes.copy()
    df_sim_despues = df_despues.copy()

    # Simulación antes de la optimización
    df_sim_antes['Tasa Simulada (%)'] = df_sim_antes['Tasa (%)'] + cambio_tasa
    if 'Valor Asignado (USD B)' in df_sim_antes.columns:
        df_sim_antes['Interés Estimado (USD B)'] = (
            df_sim_antes['Valor Asignado (USD B)'] * df_sim_antes['Tasa Simulada (%)'] / 100
        )
    else:
        df_sim_antes['Interés Estimado (USD B)'] = (
            df_sim_antes['Monto (USD B)'] * df_sim_antes['Tasa Simulada (%)'] / 100
        )

    # Simulación después de la optimización
    df_sim_despues['Tasa Simulada (%)'] = df_sim_despues['Tasa (%)'] + cambio_tasa
    if 'Valor Asignado (USD B)' in df_sim_despues.columns:
        df_sim_despues['Interés Estimado (USD B)'] = (
            df_sim_despues['Valor Asignado (USD B)'] * df_sim_despues['Tasa Simulada (%)'] / 100
        )
    else:
        df_sim_despues['Interés Estimado (USD B)'] = (
            df_sim_despues['Monto (USD B)'] * df_sim_despues['Tasa Simulada (%)'] / 100
        )

    for df_sim in [df_sim_antes, df_sim_despues]:
        if 'Categoría' not in df_sim.columns:
            df_sim['Categoría'] = df_sim.index.astype(str)
        if 'Tipo' in df_sim.columns:
            df_sim.loc[df_sim['Tipo'] == 'Pasivo', 'Interés Estimado (USD B)'] *= -1

    # Calcular impacto total como suma neta (Activos - Pasivos)
    impacto_total_antes = df_sim_antes['Interés Estimado (USD B)'].sum()
    impacto_total_despues = df_sim_despues['Interés Estimado (USD B)'].sum()

    resumen_impacto = {
        "Impacto Antes": impacto_total_antes,
        "Impacto Después": impacto_total_despues
    }

    return df_sim_antes, df_sim_despues, resumen_impacto







def check_feasibility(*args, **kwargs):
    _, resumen = run_optimization(*args, **kwargs)
    return "error" not in resumen

PARAM_DESCRIPTION = {
    "tasa_objetivo": "Tasa promedio objetivo del portafolio de activos.",
    "liquidez_minima": "Porcentaje mínimo de liquidez deseado respecto al total de activos."
}




