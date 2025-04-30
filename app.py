import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from modules.optimizer import (
    run_optimization,
    simular_escenario,
    check_feasibility,
    calcular_convexidad,
    graficar_convexidad,
    calcular_eve, 
    PARAM_DESCRIPTION
)


from modules.optimizer import PARAM_DESCRIPTION

st.set_page_config(page_title="Citi ALM Optimizer", layout="wide")
st.title("Optimización y Simulación ALM - Citigroup - Por Oscar Zeledón - ADEN Business School")

st.sidebar.header("📂 Cargar archivo de entrada")
uploaded_file = st.sidebar.file_uploader("Selecciona un archivo CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file, encoding='latin1')
    st.success("Archivo cargado correctamente ✅")

    opt_tab, sim_tab, sens_tab, convex_tab, eve_tab = st.tabs([
        "🔏 Optimización de Portafolio",
        "📈 Simulación de Tasas",
        "📊 Análisis de Sensibilidad",
        "🔁 Curva de Convexidad",
        "💼 EVE (Valor Económico del Capital)"
    ])


    def format_b(val):
        return f"{val:.3f}B"






    with opt_tab:
        st.header("Parámetros de Optimización")

        activos = df[df['Tipo'] == 'Activo']
        pasivos = df[df['Tipo'] == 'Pasivo']
        dur_act_orig = np.sum(activos['Monto (USD B)'] * activos['Duración (años)']) / np.sum(activos['Monto (USD B)'])
        dur_pas_orig = np.sum(pasivos['Monto (USD B)'] * pasivos['Duración (años)']) / np.sum(pasivos['Monto (USD B)'])
        var_orig = np.std(activos['Tasa (%)'] / 100) * 1.65 * np.sum(activos['Monto (USD B)'])
        liquidez_actual = activos[activos['Categoría'] == 'Efectivo']['Monto (USD B)'].sum()
        total_activos = activos['Monto (USD B)'].sum()
        porcentaje_liquidez_actual = (liquidez_actual / total_activos) * 100 if total_activos != 0 else 0
        tasa_promedio_actual = np.sum(activos['Monto (USD B)'] * activos['Tasa (%)']) / np.sum(activos['Monto (USD B)'])

        riesgo_promedio_actual = np.nan
        if 'Peso de Riesgo' in df.columns:
            riesgo_promedio_actual = np.sum(activos['Monto (USD B)'] * activos['Peso de Riesgo']) / np.sum(activos['Monto (USD B)'])

        st.markdown(f"**💵 Liquidez Actual: {porcentaje_liquidez_actual:.2f}% del total de activos**")
        if not np.isnan(riesgo_promedio_actual):
            st.markdown(f"**⚡ Riesgo Promedio Actual: {riesgo_promedio_actual:.2f}**")

        st.markdown("### 📏 Tolerancias Globales (Escribe en %)")
        tolerancia_duracion = st.number_input("Tolerancia de Desbalance de Duración (%)", value=5.0)
        tolerancia_monto = st.number_input("Tolerancia de Variación del Monto Total (%)", value=5.0)

        st.markdown("### 🌟 Parámetros de Optimización Globales")
        tasa_min = float(np.min(activos['Tasa (%)']))
        tasa_max = float(np.max(activos['Tasa (%)']))
        st.markdown(f"**Tasa Promedio Actual del Portafolio: {tasa_promedio_actual:.2f}%**")
        tasa_objetivo = st.number_input("Tasa Objetivo Promedio (%)", min_value=tasa_min, max_value=tasa_max, value=round((tasa_min + tasa_max) / 2, 2))
        st.caption(PARAM_DESCRIPTION.get("tasa_objetivo", ""))

        porcentaje_liquidez_objetivo = st.number_input("Liquidez Objetivo (% del total activos)", 0.0, 100.0, 5.0)

        st.markdown("### ⚖️ Opciones de Penalización")
        penalizar_concentracion = st.checkbox("Penalizar concentración de categorías", value=True)
        penalizar_diversificacion = st.checkbox("Penalizar falta de diversificación", value=True)
        considerar_riesgo = st.checkbox("Minimizar riesgo ponderado del portafolio", value=True)

        opcion_optimizar = st.selectbox("¿Hacia dónde deseas optimizar el monto total?", options=["Cualquiera", "Subir Total", "Bajar Total"])
        optimizar_hacia_arriba = opcion_optimizar == "Subir Total"
        optimizar_hacia_abajo = opcion_optimizar == "Bajar Total"

        st.markdown("### 🛠️ Tolerancia de Variación por Categoría (%)")
        categorias = df['Categoría'].unique()
        tolerancias_categorias = {}

        with st.expander("Configurar Tolerancias Individuales por Categoría"):
            for cat in categorias:
                tolerancia = st.number_input(f"Tolerancia para {cat} (%)", min_value=0.0, max_value=100.0, value=10.0, step=1.0)
                tolerancias_categorias[cat] = tolerancia

        factible = check_feasibility(
            df,
            tasa_objetivo,
            porcentaje_liquidez_objetivo,
            tolerancia_duracion,
            tolerancia_monto,
            tolerancias_categorias,
            optimizar_hacia_arriba,
            optimizar_hacia_abajo
        )

        if not factible:
            st.warning("⚠️ La combinación de restricciones y tolerancias no permite una solución viable.")
        else:











            if st.button("🚀 Ejecutar Optimización"):
                resultado, resumen = run_optimization(
                    df,
                    tasa_objetivo,
                    porcentaje_liquidez_objetivo,
                    tolerancia_duracion,
                    tolerancia_monto,
                    tolerancias_categorias,
                    optimizar_hacia_arriba,
                    optimizar_hacia_abajo,
                    penalizar_concentracion=penalizar_concentracion,
                    penalizar_diversificacion=penalizar_diversificacion,
                    considerar_riesgo=considerar_riesgo,
                    peso_riesgo=0.5
                )

                if "error" in resumen:
                    st.error("❌ Optimización no exitosa.")
                else:
                    # 🔄 Actualizar valores antes de agrupar
                    resultado['Valor Asignado (USD B)'] = (resultado['Asignación Óptima (%)'] / 100) * resultado['Monto (USD B)']

                    # 📊 Agrupar por Tipo, Categoría y Subtipo
                    resultado_agrupado = resultado.groupby(['Tipo', 'Categoría', 'Subtipo'], as_index=False).agg({
                        'Monto (USD B)': 'sum',
                        'Duración (años)': 'mean',
                        'Tasa (%)': 'mean',
                        'Peso de Riesgo': 'mean',
                        'Asignación Óptima (%)': 'sum',
                        'Valor Asignado (USD B)': 'sum'
                    })




                    activos_opt = resultado_agrupado[resultado_agrupado['Tipo'] == 'Activo']
                    rendimientos = activos_opt['Tasa (%)'] / 100
                    var_despues = np.std(rendimientos) * 1.65 * activos_opt['Valor Asignado (USD B)'].sum()

                    resumen['Valor en Riesgo (VaR 95%) USD B'] = var_despues

                    st.success("✅ Optimización exitosa")

                    st.subheader("📋 Datos Comparativos - Original + Optimizado")
                    st.dataframe(resultado_agrupado)



                    # ===========================
                    # 📈 Curva de Convexidad - Antes y Después
                    # ===========================

                    with convex_tab:
                        if resultado is not None:
                            tasas_eval = np.linspace(-0.03, 0.03, 13)
                            _, convexidad_antes = calcular_convexidad(df, tasas_eval)
                            _, convexidad_despues = calcular_convexidad(resultado, tasas_eval)
                            fig_convex = graficar_convexidad(tasas_eval, convexidad_antes, convexidad_despues)

                    # Agregar etiquetas de tasas en puntos base (ej: -300 bps)
                            ax = fig_convex.axes[0]
                            for i, tasa in enumerate(tasas_eval):
                                ax.annotate(f"{int(tasa*10000)}bps", (tasas_eval[i], convexidad_despues[i]),
                                    textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8, color='orange')
                                ax.annotate(f"{int(tasa*10000)}bps", (tasas_eval[i], convexidad_antes[i]),
                                    textcoords="offset points", xytext=(0, -10), ha='center', fontsize=8, color='blue')

                            st.subheader("📉 Curva de Convexidad del Portafolio de Activos")
                            st.pyplot(fig_convex)

                            st.markdown("""
                    #### 🔍 Interpretación de la Curva de Convexidad
                    La curva de convexidad muestra cómo cambia el valor total del portafolio de activos ante distintos cambios en la tasa de interés.
                    
                    - El eje X representa el cambio en la tasa de interés (en puntos base).
                    - El eje Y muestra el valor estimado del portafolio bajo ese cambio de tasa.
                    - Una curva más **pronunciada** indica **mayor sensibilidad** a los cambios de tasa.
                    - Si la curva **sube más rápido** hacia los extremos, hay mayor **convexidad positiva**, lo cual es deseable desde la perspectiva de gestión de riesgo.
                    
                    Comparar las curvas **antes y después de la optimización** permite visualizar si el nuevo portafolio es **más estable ante shocks de tasas**.

Una curva optimizada por encima de la original indica que:

Se logró una estructura de portafolio más robusta ante cambios de tasa.
Se maximizó el valor económico del capital (EVE, Economic Value of Equity).
Se mejoró la protección del banco frente a riesgos de tasa de interés, lo cual es un objetivo central del ALM.
                    """)

                        else:
                            st.info("Ejecuta primero la optimización para ver la curva de convexidad.")



                    # ===========================
                    # 💰 EVE Antes vs Después
                    # ===========================
                    with eve_tab:
                        if resultado is not None:
                            tasa_base = tasa_promedio_actual / 100  # convertir a decimal
                            shocks = [-0.02, -0.01, 0.0, 0.01, 0.02]  # ±100 y ±200 puntos base

                            df_eve_antes = calcular_eve(df, tasa_base, shocks, columna_monto='Monto (USD B)')
                            df_eve_despues = calcular_eve(resultado, tasa_base, shocks, columna_monto='Valor Asignado (USD B)')

                            st.subheader("📉 Valor Económico del Capital (EVE) - Antes vs Después")

                            fig_eve, ax_eve = plt.subplots()
                            ax_eve.plot(df_eve_antes['Shock (%)'], df_eve_antes['EVE (USD B)'], label='Antes', marker='o', linestyle='-')
                            ax_eve.plot(df_eve_despues['Shock (%)'], df_eve_despues['EVE (USD B)'], label='Después', marker='o', linestyle='--')

                            ax_eve.set_title("Análisis EVE ante Shocks de Tasa")
                            ax_eve.set_xlabel("Shock de Tasa (%)")
                            ax_eve.set_ylabel("EVE (USD B)")
                            ax_eve.axhline(0, color='gray', linestyle='--')
                            ax_eve.legend()
                            ax_eve.grid(True)
                            st.pyplot(fig_eve)

                            st.markdown("""
                    **Interpretación**:
                    - EVE representa el impacto en el valor económico del capital ante cambios paralelos en la curva de tasas.
                    - Una curva más alta y estable indica mayor resiliencia del portafolio ante variaciones de tasas de interés.
                    - Comparar las líneas *Antes* y *Después* permite evaluar si la optimización mejoró la sensibilidad estructural del balance.
                    """)


                        else:
                            st.info("Ejecuta primero la optimización para ver el análisis EVE.")



                    # ===========================
                    # 📊 Nueva Distribución: Categoría (Antes vs Después)
                    # ===========================
                    st.subheader("📊 Distribución por Categoría - Antes vs Después")

                    df_categoria_antes = df.groupby('Categoría', as_index=False)['Monto (USD B)'].sum()
                    df_categoria_despues = resultado_agrupado.groupby('Categoría', as_index=False)['Valor Asignado (USD B)'].sum()

                    df_dist_categoria = pd.merge(df_categoria_antes, df_categoria_despues, on='Categoría', how='outer').fillna(0)
                    df_dist_categoria.rename(columns={'Monto (USD B)': 'Monto Antes (USD B)', 'Valor Asignado (USD B)': 'Monto Después (USD B)'}, inplace=True)

                    fig_cat_dist, ax_cat_dist = plt.subplots(figsize=(12, 6))
                    x = np.arange(len(df_dist_categoria['Categoría']))
                    width = 0.35

                    ax_cat_dist.bar(x - width/2, df_dist_categoria['Monto Antes (USD B)'], width, label="Antes", color='gray')
                    ax_cat_dist.bar(x + width/2, df_dist_categoria['Monto Después (USD B)'], width, label="Después", color='royalblue')

                    ax_cat_dist.set_xticks(x)
                    ax_cat_dist.set_xticklabels(df_dist_categoria['Categoría'], rotation=45, ha='right')
                    ax_cat_dist.set_ylabel("USD Billones")
                    ax_cat_dist.set_title("Distribución por Categoría (Antes vs Después)")
                    ax_cat_dist.legend()
                    st.pyplot(fig_cat_dist)

                    # ===========================
                    # 📊 Nueva Distribución: Subtipo (Antes vs Después)
                    # ===========================
                    st.subheader("📊 Distribución por Subtipo - Antes vs Después")

                    df_subtipo_antes = df.groupby('Subtipo', as_index=False)['Monto (USD B)'].sum()
                    df_subtipo_despues = resultado_agrupado.groupby('Subtipo', as_index=False)['Valor Asignado (USD B)'].sum()

                    df_dist_subtipo = pd.merge(df_subtipo_antes, df_subtipo_despues, on='Subtipo', how='outer').fillna(0)
                    df_dist_subtipo.rename(columns={'Monto (USD B)': 'Monto Antes (USD B)', 'Valor Asignado (USD B)': 'Monto Después (USD B)'}, inplace=True)

                    fig_subtipo_dist, ax_subtipo_dist = plt.subplots(figsize=(14, 6))
                    x = np.arange(len(df_dist_subtipo['Subtipo']))
                    width = 0.35

                    ax_subtipo_dist.bar(x - width/2, df_dist_subtipo['Monto Antes (USD B)'], width, label="Antes", color='gray')
                    ax_subtipo_dist.bar(x + width/2, df_dist_subtipo['Monto Después (USD B)'], width, label="Después", color='royalblue')

                    ax_subtipo_dist.set_xticks(x)
                    ax_subtipo_dist.set_xticklabels(df_dist_subtipo['Subtipo'], rotation=45, ha='right')
                    ax_subtipo_dist.set_ylabel("USD Billones")
                    ax_subtipo_dist.set_title("Distribución por Subtipo (Antes vs Después)")
                    ax_subtipo_dist.legend()
                    st.pyplot(fig_subtipo_dist)






                    st.subheader("📊 Duración Promedio - Antes vs Después")
                    fig_dur, ax_dur = plt.subplots()
                    x = np.arange(2)
                    antes = [dur_act_orig, dur_pas_orig]
                    despues = [resumen['Duración Promedio Activos (años)'], resumen['Duración Promedio Pasivos (años)']]
                    width = 0.35
                    ax_dur.bar(x - width / 2, antes, width, label="Antes", color='gray')
                    ax_dur.bar(x + width / 2, despues, width, label="Después", color='blue')
                    ax_dur.set_xticks(x)
                    ax_dur.set_xticklabels(["Activos", "Pasivos"])
                    ax_dur.legend()
                    st.pyplot(fig_dur)

                    st.subheader("📉 Valor en Riesgo (VaR 95%)")
                    fig_var, ax_var = plt.subplots()
                    valores = [var_orig, resumen['Valor en Riesgo (VaR 95%) USD B']]
                    ax_var.bar(["Antes", "Después"], valores, color=["gray", "blue"])
                    for i, val in enumerate(valores):
                        ax_var.text(i, val + 0.01 * max(valores), format_b(val), ha='center')
                    ax_var.set_ylabel("USD Billones")
                    st.pyplot(fig_var)

                    if 'Peso de Riesgo' in resultado_agrupado.columns:
                        riesgo_promedio_despues = np.sum(resultado_agrupado['Valor Asignado (USD B)'] * resultado_agrupado['Peso de Riesgo']) / np.sum(resultado_agrupado['Valor Asignado (USD B)'])
                        st.subheader("⚡ Riesgo Promedio Ponderado")
                        st.write(f"Antes: {riesgo_promedio_actual:.2f} | Después: {riesgo_promedio_despues:.2f}")






                    st.subheader("📄 Comparativa de Resultados - Indicadores")

                    # 🔵 Calcular el Índice de Balance Óptimo (IBO) antes de optimizar
                    ibo_antes = 1 - abs(dur_act_orig - dur_pas_orig) / dur_act_orig

                    tasa_optimizada = resumen['Tasa Promedio del Portafolio (%)']
                    activos_total_despues = resultado_agrupado[resultado_agrupado['Tipo'] == 'Activo']['Valor Asignado (USD B)'].sum()
                    ganancia_antes = total_activos * tasa_promedio_actual / 100
                    ganancia_despues = activos_total_despues * tasa_optimizada / 100

                    comparativa = {
                        "Indicador": [
                            "Duración Promedio Activos (años)",
                            "Duración Promedio Pasivos (años)",
                            "Índice de Balance Óptimo (IBO)",
                            "Tasa Promedio (%)",
                            "Liquidez (%)",
                            "Valor en Riesgo (VaR 95%)",
                        ],
                        "Antes": [
                            dur_act_orig,
                            dur_pas_orig,
                            ibo_antes,
                            tasa_promedio_actual,
                            porcentaje_liquidez_actual,
                            var_orig,
                        ],
                        "Después": [
                            resumen['Duración Promedio Activos (años)'],
                            resumen['Duración Promedio Pasivos (años)'],
                            resumen['Índice de Balance Óptimo (IBO)'],
                            resumen['Tasa Promedio del Portafolio (%)'],
                            resumen['Liquidez % Después'],
                            resumen['Valor en Riesgo (VaR 95%) USD B'],
                        ],
                        "Monto USD Antes": [
                            None,
                            None,
                            None,
                            tasa_promedio_actual * total_activos / 100,
                            porcentaje_liquidez_actual * total_activos / 100,
                            var_orig,
                        ],
                        "Monto USD Después": [
                            None,
                            None,
                            None,
                            tasa_optimizada * activos_total_despues / 100,
                            resumen['Liquidez % Después'] * activos_total_despues / 100,
                            resumen['Valor en Riesgo (VaR 95%) USD B'],
                        ]
                    }

                    df_comp = pd.DataFrame(comparativa)
                    df_comp["Diferencia Absoluta"] = df_comp["Después"] - df_comp["Antes"]
                    df_comp["Diferencia Monto USD"] = df_comp["Monto USD Después"] - df_comp["Monto USD Antes"]

                    df_comp = df_comp[[
                        "Indicador", "Antes", "Después",
                        "Diferencia Absoluta", "Monto USD Antes",
                        "Monto USD Después", "Diferencia Monto USD"
                    ]]

                    # Formateo manual para cada columna
                    format_dict = {
                        "Antes": "{:.2f}",
                        "Después": "{:.2f}",
                        "Diferencia Absoluta": "{:.2f}",
                        "Monto USD Antes": "{:.2f}",
                        "Monto USD Después": "{:.2f}",
                        "Diferencia Monto USD": "{:.2f}"
                    }

                    styled_df = df_comp.style.format(format_dict)

                    st.dataframe(styled_df)

                    st.markdown("""
                    ### 🧾 Interpretación de Indicadores

                    - **Duración Promedio Activos / Pasivos**: Representa el plazo promedio ponderado de vencimiento. Una mayor duración indica mayor exposición al riesgo de tasa.
                    - **Índice de Balance Óptimo (IBO)**: Mide el alineamiento entre la duración de activos y pasivos. Se acerca a 1 cuando hay un balance perfecto.
                    - **Tasa Promedio (%)**: Es el rendimiento promedio del portafolio de activos.
                    - **Liquidez (%)**: Proporción de activos líquidos como porcentaje del total de activos. Mayor liquidez mejora la capacidad de respuesta ante salidas de fondos.
                    - **Valor en Riesgo (VaR 95%)**: Estimación de pérdida máxima esperada con un 95% de confianza. Un menor VaR indica menor riesgo de pérdida extrema.
                    - **Diferencia Absoluta**: Variación en cada indicador tras la optimización.
                    - **Monto USD Antes / Después**: Impacto financiero en términos monetarios de cada indicador.
                    - **Diferencia Monto USD**: Cambio neto en valor monetario generado por la optimización.
                    """)




    # ===========================
    # PESTAÑA: SIMULACIÓN DE TASAS
    # ===========================
    with sim_tab:
        st.header("Simulación de Tasas de Interés")

        cambio_tasa = st.slider("Cambio en la tasa (%)", -3.0, 3.0, 0.5, step=0.1)

        if 'resultado' in locals():
            df_sim = simular_escenario(resultado, cambio_tasa)
        else:
            df_sim = simular_escenario(df, cambio_tasa)

        # Agrupar por Tipo, Categoría y Subtipo
        df_sim_grouped = df_sim.groupby(['Tipo', 'Categoría', 'Subtipo'], as_index=False).agg({
            'Tasa (%)': 'mean',
            'Tasa Simulada (%)': 'mean',
            'Interés Estimado (USD B)': 'sum'
        })

        st.subheader("Resultado de la Simulación")
        st.dataframe(df_sim_grouped[['Tipo', 'Categoría', 'Subtipo', 'Tasa (%)', 'Tasa Simulada (%)', 'Interés Estimado (USD B)']])

        # Calcular impacto total: activos - pasivos
        impacto_activos = df_sim_grouped[df_sim_grouped['Tipo'] == 'Activo']['Interés Estimado (USD B)'].sum()
        impacto_pasivos = df_sim_grouped[df_sim_grouped['Tipo'] == 'Pasivo']['Interés Estimado (USD B)'].sum()
        impacto_total = impacto_activos - abs(impacto_pasivos)

        # ========== Primer gráfico: Impacto por CATEGORÍA principal ==========
        st.subheader("Impacto Estimado por Categoría Principal")
        df_categoria = df_sim_grouped.groupby(['Categoría'], as_index=False).agg({'Interés Estimado (USD B)': 'sum'})

        # Agregar el Total (activos - pasivos) al dataframe de categoría
        total_row_cat = pd.DataFrame({
            'Categoría': ['Impacto Total'],
            'Interés Estimado (USD B)': [impacto_total]
        })
        df_categoria = pd.concat([df_categoria, total_row_cat], ignore_index=True)

        fig_cat, ax_cat = plt.subplots(figsize=(12, 6))
        bars_cat = ax_cat.bar(df_categoria['Categoría'], df_categoria['Interés Estimado (USD B)'])

        for i, row in df_categoria.iterrows():
            ax_cat.text(
                bars_cat[i].get_x() + bars_cat[i].get_width() / 2,
                row['Interés Estimado (USD B)'] + 0.01,
                f"${row['Interés Estimado (USD B)']:.2f}B",
                ha='center',
                va='bottom',
                fontsize=9
            )

        ax_cat.set_ylabel("Interés Estimado (USD B)")
        ax_cat.set_title("Impacto por Categoría Principal (Incluyendo Total)")
        ax_cat.tick_params(axis='x', rotation=45)
        st.pyplot(fig_cat)

        # ========== Segundo gráfico: Impacto por SUBTIPO ==========
        st.subheader("Impacto Estimado por Subtipo")

        # Agregar el Total (activos - pasivos) al dataframe de subtipo
        total_row_subtipo = pd.DataFrame({
            'Tipo': ['Total'],
            'Categoría': ['Total'],
            'Subtipo': ['Impacto Total'],
            'Tasa (%)': [np.nan],
            'Tasa Simulada (%)': [np.nan],
            'Interés Estimado (USD B)': [impacto_total]
        })
        df_sim_grouped_total = pd.concat([df_sim_grouped, total_row_subtipo], ignore_index=True)

        fig_subtipo, ax_subtipo = plt.subplots(figsize=(14, 6))
        bars_subtipo = ax_subtipo.bar(df_sim_grouped_total['Subtipo'], df_sim_grouped_total['Interés Estimado (USD B)'])

        for i, row in df_sim_grouped_total.iterrows():
            if not pd.isna(row['Interés Estimado (USD B)']):
                ax_subtipo.text(
                    bars_subtipo[i].get_x() + bars_subtipo[i].get_width() / 2,
                    row['Interés Estimado (USD B)'] + 0.01,
                    f"${row['Interés Estimado (USD B)']:.2f}B",
                    ha='center',
                    va='bottom',
                    fontsize=8
                )

        ax_subtipo.set_ylabel("Interés Estimado (USD B)")
        ax_subtipo.set_title("Impacto por Subtipo (Incluyendo Total)")
        ax_subtipo.tick_params(axis='x', rotation=45)
        st.pyplot(fig_subtipo)

        st.subheader("Impacto Total en Interés Estimado")
        st.metric(label="Impacto Neto Estimado (USD B)", value=f"{impacto_total:.2f}B")



    # ===========================
    # PESTAÑA: ANÁLISIS DE SENSIBILIDAD
    # ===========================
    with sens_tab:
        st.header("Análisis de Sensibilidad a Cambios en la Tasa")

        cambios = [-2.0, -1.0, 0.0, 1.0, 2.0]
        resumenes = []

        for delta in cambios:
            df_escenario = simular_escenario(resultado if 'resultado' in locals() else df, delta)

            df_escenario_grouped = df_escenario.groupby(['Tipo'], as_index=False).agg({'Interés Estimado (USD B)': 'sum'})

            impacto_activos = df_escenario_grouped[df_escenario_grouped['Tipo'] == 'Activo']['Interés Estimado (USD B)'].sum()
            impacto_pasivos = df_escenario_grouped[df_escenario_grouped['Tipo'] == 'Pasivo']['Interés Estimado (USD B)'].sum()
            impacto_total = impacto_activos - abs(impacto_pasivos)

            resumenes.append({
                "Cambio en Tasa (%)": delta,
                "Impacto Neto Estimado (USD B)": round(impacto_total, 3)
            })

        df_resumen_sens = pd.DataFrame(resumenes)

        st.subheader("Tabla de Resultados de Sensibilidad")
        st.dataframe(df_resumen_sens)

        fig3, ax3 = plt.subplots(figsize=(8, 4))
        ax3.plot(df_resumen_sens["Cambio en Tasa (%)"], df_resumen_sens["Impacto Neto Estimado (USD B)"], marker='o', linestyle='-')
        ax3.axhline(0, color='gray', linestyle='--')
        ax3.set_title("Sensibilidad del Impacto Neto ante Cambios de Tasa")
        ax3.set_xlabel("Cambio en la Tasa (%)")
        ax3.set_ylabel("Impacto Neto Estimado (USD B)")
        ax3.grid(True)
        st.pyplot(fig3)

        st.subheader("Interpretación del Análisis de Sensibilidad")
        st.markdown("""
        - Un impacto positivo significa que **el banco se beneficia** ante el cambio de tasas.
        - Un impacto negativo indica que **el banco pierde rentabilidad** ante el cambio de tasas.
        - El gráfico ayuda a visualizar si el portafolio es sensible positiva o negativamente ante cambios del mercado.
        """)


else:
    st.warning("⚠️ Por favor, carga un archivo CSV para comenzar.")
