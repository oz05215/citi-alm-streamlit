
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

st.set_page_config(page_title="Citi ALM Optimizer", layout="wide")
st.title("Optimización y Simulación ALM - Citigroup - Por Oscar Zeledón - ADEN Business School")

st.sidebar.header("📂 Cargar archivo de entrada")
uploaded_file = st.sidebar.file_uploader("Selecciona un archivo CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file, encoding='latin1')
    st.success("Archivo cargado correctamente ✅")

    opt_tab, nii_tab, sim_tab, convex_tab, eve_tab, gap_tab = st.tabs([
        "🔏 Optimización de Portafolio",
        "📊 NII Simulado",
#        "📊 Análisis de Sensibilidad",
        "📈 Simulación de Tasas",
        "🔁 Curva de Convexidad",
        "💼 EVE (Valor Económico del Capital)",
        "🔍 Gap de Liquidez",

#        "💰 Cashflow Matching",
#        "📉 Fronteras Eficientes"

    ])

    def format_b(val):
        return f"{val:.3f}B"

    with opt_tab:
        st.header("Parámetros de Optimización")

        activos = df[df['Tipo'] == 'Activo']
        pasivos = df[df['Tipo'] == 'Pasivo']
        dur_act_orig = np.sum(activos['Monto (USD B)'] * activos['Duración (años)']) / np.sum(activos['Monto (USD B)'])
        dur_pas_orig = np.sum(pasivos['Monto (USD B)'] * pasivos['Duración (años)']) / np.sum(pasivos['Monto (USD B)'])
        #if var_original is None:        
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

        st.markdown("### 📉 Restricción Dinámica del VaR")
        porcentaje_var_tolerado = st.slider(
            "Variación permitida en el VaR (%) respecto al VaR original",
            min_value=-50.0, max_value=50.0, value=0.0, step=1.0,
            help="Ej: -10 significa que el VaR después debe ser al menos 10% menor que el VaR antes"
        )

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
                tolerancia = st.number_input(f"Tolerancia para {cat} (%)", min_value=0.0, max_value=100.0, value=100.0, step=1.0)
                tolerancias_categorias[cat] = tolerancia

        st.markdown("### 🧩 Tolerancia de Variación por Subtipo (%)")
        subtipo_por_categoria = df.groupby('Categoría')['Subtipo'].unique().to_dict()
        tolerancias_subtipos = {}

        with st.expander("Configurar Tolerancias Individuales por Subtipo"):
            for cat, subtipos in subtipo_por_categoria.items():
                st.markdown(f"**{cat}**")
                for sub in subtipos:
                    tolerancia_sub = st.number_input(f"↳ Tolerancia para {sub} (%)", min_value=0.0, max_value=100.0, value=100.0, step=1.0)
                    tolerancias_subtipos[sub] = tolerancia_sub

        # El resto del código continúa igual...


        factible = check_feasibility(
            df,
            tasa_objetivo,
            porcentaje_liquidez_objetivo,
            tolerancia_duracion,
            tolerancia_monto,
            tolerancias_categorias,
            tolerancias_subtipos,
            optimizar_hacia_arriba,
            optimizar_hacia_abajo,
            var_original=var_orig,
            porcentaje_var_tolerado=porcentaje_var_tolerado,
            cambio_tasa_simulacion=0.0
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
                    tolerancias_subtipos,
                    optimizar_hacia_arriba,
                    optimizar_hacia_abajo,
                    penalizar_concentracion=penalizar_concentracion,
                    penalizar_diversificacion=penalizar_diversificacion,
                    considerar_riesgo=considerar_riesgo,
                    peso_riesgo=0.5,
                    var_original=var_orig,
                    porcentaje_var_tolerado=porcentaje_var_tolerado,
                    cambio_tasa_simulacion=0.0
                )

                st.session_state['resultado'] = resultado  # ⬅️ AQUÍ VA ESTA LÍNEA
                st.session_state['resumen'] = resumen


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



#SIMULADOR CAMBIO TASA



    with sim_tab:
        st.header("Simulación de Tasas de Interés")

        cambio_tasa = st.slider("Cambio en la tasa (%)", -3.0, 3.0, 0.5, step=0.1)

        if 'resultado' in st.session_state:
            df_optimizado = st.session_state['resultado']
            resumen_opt = st.session_state['resumen']
        else:
            df_optimizado = df.copy()
            resumen_opt = {}

        df_sim_antes, df_sim_despues, resumen_impacto = simular_escenario(df, df_optimizado, cambio_tasa)

        # Agrupar por Tipo, Categoría y Subtipo
        df_sim_grouped_antes = df_sim_antes.groupby(['Tipo', 'Categoría', 'Subtipo'], as_index=False).agg({
            'Tasa (%)': 'mean',
            'Tasa Simulada (%)': 'mean',
            'Interés Estimado (USD B)': 'sum'
        })

        df_sim_grouped_despues = df_sim_despues.groupby(['Tipo', 'Categoría', 'Subtipo'], as_index=False).agg({
            'Tasa (%)': 'mean',
            'Tasa Simulada (%)': 'mean',
            'Interés Estimado (USD B)': 'sum'
        })

        st.subheader("Resultado de la Simulación (Después de Optimizar)")
        st.dataframe(df_sim_grouped_despues[['Tipo', 'Categoría', 'Subtipo', 'Tasa (%)', 'Tasa Simulada (%)', 'Interés Estimado (USD B)']])

        # Calcular impactos netos reales
        impacto_total_antes = resumen_impacto["Impacto Antes"]
        impacto_total_despues = resumen_impacto["Impacto Después"]

        # ========== Gráfico: Impacto por CATEGORÍA principal ==========
        st.subheader("Impacto Estimado por Categoría Principal")

        df_cat_antes = df_sim_antes.groupby('Categoría', as_index=False)['Interés Estimado (USD B)'].sum()
        df_cat_despues = df_sim_despues.groupby('Categoría', as_index=False)['Interés Estimado (USD B)'].sum()

        df_cat = pd.merge(df_cat_antes, df_cat_despues, on='Categoría', how='outer', suffixes=(' Antes', ' Después')).fillna(0)
        df_cat = pd.concat([df_cat, pd.DataFrame([{
            'Categoría': 'Impacto Total',
            'Interés Estimado (USD B) Antes': impacto_total_antes,
            'Interés Estimado (USD B) Después': impacto_total_despues
        }])], ignore_index=True)

        fig_cat, ax_cat = plt.subplots(figsize=(12, 6))
        x = np.arange(len(df_cat['Categoría']))
        width = 0.35
        ax_cat.bar(x - width/2, df_cat['Interés Estimado (USD B) Antes'], width, label='Antes', color='gray')
        ax_cat.bar(x + width/2, df_cat['Interés Estimado (USD B) Después'], width, label='Después', color='royalblue')
        ax_cat.set_xticks(x)
        ax_cat.set_xticklabels(df_cat['Categoría'], rotation=45, ha='right')
        ax_cat.set_ylabel("Interés Estimado (USD B)")
        ax_cat.set_title("Impacto por Categoría Principal")
        ax_cat.legend()
        st.pyplot(fig_cat)

        # ========== Gráfico: Impacto por SUBTIPO ==========
        st.subheader("Impacto Estimado por Subtipo")

        df_sub_antes = df_sim_antes.groupby('Subtipo', as_index=False)['Interés Estimado (USD B)'].sum()
        df_sub_despues = df_sim_despues.groupby('Subtipo', as_index=False)['Interés Estimado (USD B)'].sum()

        df_sub = pd.merge(df_sub_antes, df_sub_despues, on='Subtipo', how='outer', suffixes=(' Antes', ' Después')).fillna(0)
        df_sub = pd.concat([df_sub, pd.DataFrame([{
            'Subtipo': 'Impacto Total',
            'Interés Estimado (USD B) Antes': impacto_total_antes,
            'Interés Estimado (USD B) Después': impacto_total_despues
        }])], ignore_index=True)

        fig_sub, ax_sub = plt.subplots(figsize=(14, 6))
        x = np.arange(len(df_sub['Subtipo']))
        width = 0.35
        ax_sub.bar(x - width/2, df_sub['Interés Estimado (USD B) Antes'], width, label='Antes', color='gray')
        ax_sub.bar(x + width/2, df_sub['Interés Estimado (USD B) Después'], width, label='Después', color='royalblue')
        ax_sub.set_xticks(x)
        ax_sub.set_xticklabels(df_sub['Subtipo'], rotation=45, ha='right')
        ax_sub.set_ylabel("Interés Estimado (USD B)")
        ax_sub.set_title("Impacto por Subtipo")
        ax_sub.legend()
        st.pyplot(fig_sub)

        # ========== Métricas finales ==========
        st.subheader("Impacto Total en Interés Estimado")

        col1, col2 = st.columns(2)
        col1.metric("Impacto Neto Antes (USD B)", f"{impacto_total_antes:.2f}B")
        col2.metric("Impacto Neto Después (USD B)", f"{impacto_total_despues:.2f}B")

        # ========== Validación visual si cambio es 0 ==========
        if cambio_tasa == 0.0 and resumen_opt:
            esperado = resumen_opt.get('Ganancia Estimada Después (USD B)', None)
            if esperado is not None:
                diferencia = impacto_total_despues - esperado
                st.markdown(f"✅ Validación: Cuando cambio = 0%, ganancia esperada ≈ **{esperado:.2f}B**")
                st.markdown(f"📊 Diferencia vs Simulación: **{diferencia:+.4f}B**")

                # Gráfico de comparación
                fig_comp, ax_comp = plt.subplots()
                valores = [esperado, impacto_total_despues]
                etiquetas = ["Ganancia del Resumen", "Impacto Simulado"]
                colores = ["orange", "blue"]
                ax_comp.bar(etiquetas, valores, color=colores)
                for i, val in enumerate(valores):
                    ax_comp.text(i, val + 0.01 * max(valores), f"{val:.2f}B", ha='center')
                ax_comp.set_title("Comparación de Ganancia Estimada vs Impacto Simulado (Cambio = 0%)")
                st.pyplot(fig_comp)


    # ===========================
    # PESTAÑA: ANÁLISIS DE SENSIBILIDAD
    # ===========================
#        with sens_tab:
#            st.header("Análisis de Sensibilidad a Cambios en la Tasa")
#
#            cambios = [-2.0, -1.0, 0.0, 1.0, 2.0]
#            resumenes = []
#
#            if 'resultado' in st.session_state:
#                df_optimizado = st.session_state['resultado']
#            else:
#                df_optimizado = df.copy()
#
#            for delta in cambios:
#                df_sim_antes, df_sim_despues, resumen_impacto = simular_escenario(df, df_optimizado, delta)
#
#                resumenes.append({
#                    "Cambio en Tasa (%)": delta,
#                    "Impacto Neto Antes (USD B)": round(resumen_impacto['Impacto Antes'], 3),
#                    "Impacto Neto Después (USD B)": round(resumen_impacto['Impacto Después'], 3),
#                })
#
#            df_resumen_sens = pd.DataFrame(resumenes)
#
#            st.subheader("Tabla de Resultados de Sensibilidad")
#            st.dataframe(df_resumen_sens)
#
#            # Graficar ambas curvas
#            fig3, ax3 = plt.subplots(figsize=(8, 4))
#            ax3.plot(df_resumen_sens["Cambio en Tasa (%)"], df_resumen_sens["Impacto Neto Antes (USD B)"], marker='o', linestyle='-', label="Antes", color="gray")
#            ax3.plot(df_resumen_sens["Cambio en Tasa (%)"], df_resumen_sens["Impacto Neto Después (USD B)"], marker='o', linestyle='--', label="Después", color="royalblue")
#            ax3.axhline(0, color='black', linestyle='--')
#            ax3.set_title("Sensibilidad del Impacto Neto ante Cambios de Tasa")
#            ax3.set_xlabel("Cambio en la Tasa (%)")
#            ax3.set_ylabel("Impacto Neto Estimado (USD B)")
#            ax3.legend()
#            ax3.grid(True)
#            st.pyplot(fig3)
#
#            st.subheader("Interpretación del Análisis de Sensibilidad")
#            st.markdown("""
#            - Un impacto positivo significa que **el banco se beneficia** ante el cambio de tasas.
#            - Un impacto negativo indica que **el banco pierde rentabilidad** ante el cambio de tasas.
#            - El gráfico compara el comportamiento del portafolio **antes y después de optimizar**.
#            """)


    # ===========================
    # 📈 Curva de Convexidad - Antes y Después
    # ===========================
    with convex_tab:
        if 'resultado' in st.session_state and st.session_state['resultado'] is not None:
            resultado = st.session_state['resultado']
            tasas_eval = np.linspace(-0.03, 0.03, 13)

            # Calcula convexidad antes y después
            _, convexidad_antes = calcular_convexidad(df, tasas_eval)
            _, convexidad_despues = calcular_convexidad(resultado, tasas_eval)

            fig_convex = graficar_convexidad(tasas_eval, convexidad_antes, convexidad_despues)

            # Etiquetas en puntos base
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
                - El eje X representa el cambio en la tasa de interés (en puntos base).
                - El eje Y muestra el valor estimado del portafolio bajo ese cambio de tasa.
                - Una curva más **pronunciada** indica **mayor sensibilidad** a los cambios de tasa.
                - Una curva optimizada por encima de la original indica que:
                  - Se logró una estructura de portafolio más robusta ante cambios de tasa.
                  - Se maximizó el valor económico del capital (EVE).
                  - Se mejoró la protección frente al riesgo de tasa de interés.
            """)

        else:
            st.info("Ejecuta primero la optimización para ver la curva de convexidad.")

    # ===========================
    # 💰 EVE Antes vs Después
    # ===========================
    with eve_tab:
        if 'resultado' in st.session_state and st.session_state['resultado'] is not None:
            resultado = st.session_state['resultado']
            tasa_base = tasa_promedio_actual / 100
            shocks = [-0.02, -0.01, 0.0, 0.01, 0.02]

            df_eve_antes = calcular_eve(df, tasa_base, shocks, columna_monto='Monto (USD B)')
            # Asegura que la columna exista (por si se salta la optimización)
            if 'Valor Asignado (USD B)' not in resultado.columns:
                resultado['Valor Asignado (USD B)'] = resultado['Monto (USD B)']

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
                - EVE mide el impacto en el valor económico del capital ante shocks paralelos en tasas.
                - Una línea más alta y estable sugiere mayor resiliencia del portafolio.
                - La comparación *Antes* vs *Después* indica si la optimización mejoró la sensibilidad estructural.
            """)


        else:
            st.info("Ejecuta primero la optimización para ver el análisis EVE.")

#    # FRONTERA TAB
#    with frontera_tab:
#        st.header("📉 Fronteras Eficientes: EVE vs Duración y Sensibilidad vs IBO")
#
#        if 'resultado' not in st.session_state:
#            st.info("Ejecuta la optimización primero para generar las fronteras eficientes.")
#        else:
#            resultado = st.session_state['resultado']
#            tasa_base = tasa_promedio_actual / 100
#            shocks = [-0.02, -0.01, 0.01, 0.02]
#
#            df_eve = calcular_eve(resultado, tasa_base, shocks, columna_monto='Valor Asignado (USD B)')
#            eve_promedio = df_eve['EVE (USD B)'].mean()
#
#            activos = resultado[resultado['Tipo'] == 'Activo']
#            duracion_activos = np.sum(activos['Valor Asignado (USD B)'] * activos['Duración (años)']) / np.sum(activos['Valor Asignado (USD B)'])
#
#            pasivos = resultado[resultado['Tipo'] == 'Pasivo']
#            dur_pasivos = np.sum(pasivos['Valor Asignado (USD B)'] * pasivos['Duración (años)']) / np.sum(pasivos['Valor Asignado (USD B)'])
#            ibo = 1 - abs(duracion_activos - dur_pasivos) / duracion_activos if duracion_activos != 0 else 0
#
#            impacto_base = np.sum(activos['Valor Asignado (USD B)'] * activos['Tasa (%)'] / 100)
#            sensibilidad_score = 0
#            ganancia_total = 0
#            perdida_total = 0
#            for delta in shocks:
#                tasa_sim = activos['Tasa (%)'] + delta * 100
#                interes = activos['Valor Asignado (USD B)'] * tasa_sim / 100
#                impacto = interes.sum()
#                if delta < 0:
#                    perdida_total += max(0, impacto_base - impacto)
#                else:
#                    ganancia_total += max(0, impacto - impacto_base)
#            sensibilidad_score = perdida_total - 1.25 * ganancia_total
#
#            # ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
#            # 🔵 EXPLICACIÓN DEL GRÁFICO: EVE vs DURACIÓN PROMEDIO
#            # ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
#            st.markdown("""
#            ### 🔷 Interpretación: EVE vs Duración Promedio
#            - **EVE (Economic Value of Equity)** representa la estabilidad del valor del portafolio ante shocks de tasas.
#            - **Duración promedio** mide la exposición del portafolio a tasas de interés: duraciones largas implican mayor riesgo de tasa.
#            - Idealmente se busca **maximizar el EVE** manteniendo una **duración razonable (ej. entre 2 y 4 años)**.
#            - Un buen punto se ubica en la parte **superior izquierda del gráfico** (alto EVE, duración moderada).
#            """)
#
#            fig1, ax1 = plt.subplots()
#            ax1.scatter(duracion_activos, eve_promedio, color='royalblue', s=100)
#            ax1.set_title("Frontera Eficiente: EVE vs Duración Promedio Activos")
#            ax1.set_xlabel("Duración Promedio Activos (años)")
#            ax1.set_ylabel("EVE Promedio (USD B)")
#            ax1.grid(True)
#            st.pyplot(fig1)
#
#            # ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
#            # 🟠 EXPLICACIÓN DEL GRÁFICO: SENSIBILIDAD vs IBO
#            # ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
#            st.markdown("""
#            ### 🟠 Interpretación: Sensibilidad a Tasas vs IBO
#            - **Sensibilidad** representa el efecto neto de cambios en tasas sobre los ingresos (penaliza más las pérdidas).
#            - **IBO (Índice de Balance Óptimo)** mide el alineamiento entre duración de activos y pasivos (ideal ≈ 1).
#            - Se busca una combinación donde el **IBO sea alto** (cercano a 1) y la **sensibilidad sea baja (negativa, pero controlada)**.
#            - El mejor punto está en la **esquina inferior derecha del gráfico**: bajo impacto negativo y alto balance.
#            """)
#
#            fig2, ax2 = plt.subplots()
#            ax2.scatter(ibo, sensibilidad_score, color='orange', s=100)
#            ax2.set_title("Frontera Eficiente: Sensibilidad a Tasas vs IBO")
#            ax2.set_xlabel("Índice de Balance Óptimo (IBO)")
#            ax2.set_ylabel("Sensibilidad (pérdidas - 1.25 * ganancias)")
#            ax2.grid(True)
#            st.pyplot(fig2)
#


# GAP LIQUIDEZ TAB

    with gap_tab:
        st.header("💧 Gap de Liquidez por Buckets Temporales")

        if 'resultado' not in st.session_state:
            st.info("Ejecuta primero la optimización para analizar los gaps.")
        else:
            resultado = st.session_state['resultado']

            def calcular_gap(df, etiqueta):
                buckets = ['0-30 días', '31-90 días', '91-180 días', '181-365 días', '1-3 años', '3-5 años', '5+ años']
                rangos = [30, 90, 180, 365, 3*365, 5*365, np.inf]
                df = df.copy()

                monto_col = 'Valor Asignado (USD B)' if 'Valor Asignado (USD B)' in df.columns else 'Monto (USD B)'

                df['Dias'] = df['Duración (años)'] * 365
                df['Bucket'] = pd.cut(df['Dias'], bins=[0] + rangos, labels=buckets, right=True)

                activos = df[df['Tipo'] == 'Activo'].groupby('Bucket')[monto_col].sum()
                pasivos = df[df['Tipo'] == 'Pasivo'].groupby('Bucket')[monto_col].sum()

                df_gap = pd.DataFrame({
                    'Activos (USD B)': activos,
                    'Pasivos (USD B)': pasivos
                }).fillna(0)

                df_gap['Gap (USD B)'] = df_gap['Activos (USD B)'] - df_gap['Pasivos (USD B)']
                df_gap['Gap Acumulado (USD B)'] = df_gap['Gap (USD B)'].cumsum()
                df_gap['Escenario'] = etiqueta

                return df_gap.reset_index()

            df_gap_antes = calcular_gap(df, "Antes")
            df_gap_despues = calcular_gap(resultado, "Después")

            st.subheader("📋 Gap de Liquidez - Antes de Optimizar")
            st.dataframe(df_gap_antes)

            st.subheader("📋 Gap de Liquidez - Después de Optimizar")
            st.dataframe(df_gap_despues)

            # Gráfico acumulado
            st.subheader("📈 Gap Acumulado - Comparación Visual")

            fig_gap, ax_gap = plt.subplots(figsize=(12, 5))
            buckets = df_gap_antes['Bucket']
            x = np.arange(len(buckets))
            width = 0.35

            ax_gap.bar(x - width, df_gap_antes['Gap Acumulado (USD B)'], width, label='Antes', color='gray')
            ax_gap.bar(x + width, df_gap_despues['Gap Acumulado (USD B)'], width, label='Después', color='royalblue')

            ax_gap.set_xticks(x)
            ax_gap.set_xticklabels(buckets, rotation=45, ha='right')
            ax_gap.set_ylabel("Gap Acumulado (USD B)")
            ax_gap.set_title("Gap de Liquidez Acumulado por Bucket")
            ax_gap.legend()
            ax_gap.grid(True)

            st.pyplot(fig_gap)




    with nii_tab:
        st.header("📊 NII Simulado - Net Interest Income (Antes vs Después)")

        if 'resultado' not in st.session_state:
            st.info("⚠️ Ejecuta primero la optimización para simular el NII.")
        else:
            df_antes = df.copy()
            df_despues = st.session_state['resultado'].copy()

            shocks = [-0.02, -0.01, 0.0, 0.01, 0.02]
            resumen_nii = []

            for shock in shocks:
                tasa_simulada = df_antes['Tasa (%)'] + shock * 100

                # Antes de la optimización
                df_antes['Tasa Simulada (%)'] = tasa_simulada
                df_antes['NII Simulado'] = df_antes['Monto (USD B)'] * df_antes['Tasa Simulada (%)'] / 100
                df_antes.loc[df_antes['Tipo'] == 'Pasivo', 'NII Simulado'] *= -1
                nii_antes = df_antes['NII Simulado'].sum()

                # Después de la optimización
                df_despues['Tasa Simulada (%)'] = df_despues['Tasa (%)'] + shock * 100
                df_despues['NII Simulado'] = df_despues['Valor Asignado (USD B)'] * df_despues['Tasa Simulada (%)'] / 100
                df_despues.loc[df_despues['Tipo'] == 'Pasivo', 'NII Simulado'] *= -1
                nii_despues = df_despues['NII Simulado'].sum()

                resumen_nii.append({
                    "Shock de Tasa (%)": shock * 100,
                    "NII Antes (USD B)": round(nii_antes, 3),
                    "NII Después (USD B)": round(nii_despues, 3)
                })

            df_resumen_nii = pd.DataFrame(resumen_nii)

            st.subheader("📋 Tabla de Resultados del NII Simulado")
            st.dataframe(df_resumen_nii)

            # Gráfico de comparación NII
            fig_nii, ax_nii = plt.subplots(figsize=(8, 4))
            ax_nii.plot(df_resumen_nii["Shock de Tasa (%)"], df_resumen_nii["NII Antes (USD B)"],
                        marker='o', label='Antes', color='gray')
            ax_nii.plot(df_resumen_nii["Shock de Tasa (%)"], df_resumen_nii["NII Después (USD B)"],
                        marker='o', linestyle='--', label='Después', color='royalblue')
            ax_nii.axhline(0, color='black', linestyle='--')
            ax_nii.set_title("Net Interest Income (NII) bajo Diferentes Shocks de Tasa")
            ax_nii.set_xlabel("Shock de Tasa (%)")
            ax_nii.set_ylabel("NII Estimado (USD B)")
            ax_nii.legend()
            ax_nii.grid(True)
            st.pyplot(fig_nii)

            st.markdown("""
            ### 🧾 Interpretación
            - **NII (Net Interest Income)** representa la ganancia neta por intereses: activos generan ingresos y pasivos implican costos.
            - **Eje X**: simula shocks paralelos de tasa en ±200 puntos base.
            - **Eje Y**: muestra el NII total estimado del portafolio.
            - Una línea más estable y más alta (azul) después de la optimización es preferible.
            - Permite evaluar si el portafolio gana robustez y rentabilidad frente a cambios en tasas.
            """)



#    with cashflow_tab:
#        st.header("💰 Cash Flow Matching - Comparativo Antes vs Después")
#
#       if 'resultado' not in st.session_state:
#            st.info("⚠️ Ejecuta primero la optimización para simular el Cash Flow Matching.")
#        else:
#            def calcular_cashflow(df, label):
#                df_temp = df.copy()
#
#                # Si no existe, crear la columna Bucket en base a duración
#                if 'Bucket' not in df_temp.columns:
#                    bins = [0, 1, 3, 5, 10, np.inf]
#                    labels = ["0-1 años", "1-3 años", "3-5 años", "5-10 años", "10+ años"]
#                    df_temp['Bucket'] = pd.cut(df_temp['Duración (años)'], bins=bins, labels=labels, right=False)
#
#                # Si no existe, asumir que los valores asignados son iguales al monto original
#                if 'Valor Asignado (USD B)' not in df_temp.columns:
#                    df_temp['Valor Asignado (USD B)'] = df_temp['Monto (USD B)']
#
#                flujo = df_temp.groupby(['Bucket', 'Tipo'])['Valor Asignado (USD B)'].sum().unstack().fillna(0)
#                flujo['Neto'] = flujo.get('Activo', 0) - flujo.get('Pasivo', 0)
#                flujo.reset_index(inplace=True)
#                flujo['Escenario'] = label
#                return flujo
#
#            # Calcular antes y después
#            df_cashflow_antes = calcular_cashflow(df, "Antes")
#            df_cashflow_despues = calcular_cashflow(st.session_state['resultado'], "Después")
#
#            # Mostrar tablas por separado
#            st.subheader("📋 Cash Flow Matching - Antes de Optimizar")
#            st.dataframe(df_cashflow_antes)
#
#            st.subheader("📋 Cash Flow Matching - Después de Optimizar")
#            st.dataframe(df_cashflow_despues)
#
#            # Graficar comparación
#            st.subheader("📊 Comparación Visual del Matching por Bucket")
#
#            fig, ax = plt.subplots(figsize=(12, 6))
#            x = np.arange(len(df_cashflow_antes['Bucket']))
#            width = 0.35
#
#            ax.bar(x - width/2, df_cashflow_antes['Neto'], width, label="Antes", color='gray')
#            ax.bar(x + width/2, df_cashflow_despues['Neto'], width, label="Después", color='royalblue')
#
#            ax.set_xticks(x)
#            ax.set_xticklabels(df_cashflow_antes['Bucket'], rotation=45)
#            ax.set_ylabel("Flujo Neto (USD B)")
#            ax.set_title("Cash Flow Matching por Bucket - Antes vs Después")
#            ax.legend()
#            ax.grid(True)
#
#            st.pyplot(fig)
#
#            st.markdown("""
#            **Interpretación:**
#            - Cada barra muestra el **flujo neto** (activos menos pasivos) por periodo de tiempo.
#            - El objetivo es que el flujo neto esté lo más cercano posible a cero en cada bucket, lo que indica un mejor emparejamiento.
#            - La comparación permite evaluar si la optimización logró mejorar la cobertura de pasivos con activos de vencimiento similar.
#            """)



else:
    st.warning("⚠️ Por favor, carga un archivo CSV para comenzar.")
