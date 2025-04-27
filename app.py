import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from modules.optimizer import run_optimization, simular_escenario, check_feasibility
from modules.optimizer import PARAM_DESCRIPTION

st.set_page_config(page_title="Citi ALM Optimizer", layout="wide")
st.title("Optimización y Simulación ALM - Citigroup - Por Oscar Zeledón - ADEN Business School")

st.sidebar.header("📂 Cargar archivo de entrada")
uploaded_file = st.sidebar.file_uploader("Selecciona un archivo CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file, encoding='latin1')
    st.success("Archivo cargado correctamente ✅")

    opt_tab, sim_tab, sens_tab = st.tabs([
        "🔧 Optimización de Portafolio",
        "📈 Simulación de Tasas",
        "📊 Análisis de Sensibilidad"
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
        tasa_promedio_actual = (np.sum(activos['Monto (USD B)'] * activos['Tasa (%)']) / np.sum(activos['Monto (USD B)']))

        st.markdown(f"**💵 Liquidez Actual: {porcentaje_liquidez_actual:.2f}% del total de activos**")

        st.markdown("### 📏 Tolerancias Globales (Escribe en %)")
        tolerancia_duracion = st.number_input("Tolerancia de Desbalance de Duración (%)", value=5.0)
        tolerancia_monto = st.number_input("Tolerancia de Variación del Monto Total (%)", value=5.0)

        st.markdown("### 🌟 Parámetros de Optimización Globales")
        tasa_min = float(np.min(activos['Tasa (%)']))
        tasa_max = float(np.max(activos['Tasa (%)']))
        st.markdown(f"**Tasa Promedio Actual del Portafolio: {tasa_promedio_actual:.2f}%**")
        tasa_objetivo = st.number_input("Tasa Objetivo Promedio (%)", min_value=tasa_min, max_value=tasa_max, value=round((tasa_min + tasa_max)/2, 2))
        st.caption(PARAM_DESCRIPTION.get("tasa_objetivo", ""))

        porcentaje_liquidez_objetivo = st.number_input("Liquidez Objetivo (% del total activos)", 0.0, 100.0, 5.0)


        st.markdown("### ⚖️ Opciones de Penalización")
        penalizar_concentracion = st.checkbox("Penalizar concentración de categorías", value=True)
        penalizar_diversificacion = st.checkbox("Penalizar falta de diversificación", value=True)

#        st.markdown("### ⚖️ Pesos de Penalización")
#        peso_concentracion = st.slider("Peso Penalización Concentración", 0.0, 10.0, 0.5, #step=0.1)
#        peso_diversificacion = st.slider("Peso Penalización Diversificación", 0.0, 10.0, ##0.25, step=0.05)


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
                penalizar_diversificacion=penalizar_diversificacion
                )



                if "error" in resumen:
                    st.error("❌ Optimización no exitosa.")
                else:
                    # PRIMERO crear la columna Valor Asignado (USD B)
                    resultado['Valor Asignado (USD B)'] = (resultado['Asignación Óptima (%)'] / 100) * resultado['Monto (USD B)']


                    # AHORA filtrar activos y calcular VaR
                    # AHORA filtrar activos y calcular VaR con ponderación
#                    activos_opt = resultado[resultado['Tipo'] == 'Activo']
#                    ponderaciones = activos_opt['Valor Asignado (USD B)'] / #activos_opt['Valor Asignado (USD B)'].sum()
#                    var_despues = np.sqrt(np.sum((ponderaciones * activos_opt['Tasa (%)'] #/ 100)**2)) * 1.65 * activos_opt['Valor Asignado (USD B)'].sum()
#                    resumen['Valor en Riesgo (VaR 95%) USD B'] = var_despues

#Nueva formula var sin ponderaciones
                    activos_opt = resultado[resultado['Tipo'] == 'Activo']
                    rendimientos = activos_opt['Tasa (%)'] / 100
                    var_despues = np.std(rendimientos) * 1.65 * activos_opt['Valor Asignado (USD B)'].sum()
                    
                    
                    resumen['Valor en Riesgo (VaR 95%) USD B'] = var_despues

                    st.success("✅ Optimización exitosa")




                    st.subheader("📋 Datos Comparativos - Original + Optimizado")
                    df_completo = df.copy()
                    resultado['Valor Asignado (USD B)'] = (resultado['Asignación Óptima (%)'] / 100) * resultado['Monto (USD B)']


                    df_completo = df_completo.merge(
                        resultado[['Categoría', 'Asignación Óptima (%)', 'Valor Asignado (USD B)']],
                        on='Categoría',
                        how='left'
                    )
                    st.dataframe(df_completo)

                    st.subheader("📊 Duración Promedio - Antes vs Después")
                    fig_dur, ax_dur = plt.subplots()
                    x = np.arange(2)
                    antes = [dur_act_orig, dur_pas_orig]
                    despues = [resumen['Duración Promedio Activos (años)'], resumen['Duración Promedio Pasivos (años)']]
                    width = 0.35
                    ax_dur.bar(x - width/2, antes, width, label="Antes", color='gray')
                    ax_dur.bar(x + width/2, despues, width, label="Después", color='blue')
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

                    st.subheader("🏦 Top 5 Asignaciones - Antes vs Después")
                    top_resultado = resultado.sort_values('Valor Asignado (USD B)', ascending=False).head(5)
                    categorias_top = top_resultado['Categoría'].tolist()
                    top_orig = df[df['Categoría'].isin(categorias_top)].copy()
                    top_orig = top_orig.set_index('Categoría').reindex(categorias_top).fillna(0).reset_index()

                    fig_top, ax_top = plt.subplots(figsize=(9, 5))
                    x = np.arange(len(categorias_top))
                    ax_top.bar(x - width/2, top_orig['Monto (USD B)'], width, label="Antes", color='gray')
                    ax_top.bar(x + width/2, top_resultado['Valor Asignado (USD B)'], width, label="Después", color='royalblue')
                    ax_top.set_xticks(x)
                    ax_top.set_xticklabels(categorias_top, rotation=45)
                    ax_top.legend()
                    st.pyplot(fig_top)

                    st.subheader("📄 Comparativa de Resultados - Indicadores")
                    tasa_objetivo_actual = tasa_promedio_actual
                    tasa_optimizada = resumen['Tasa Promedio del Portafolio (%)']
                    activos_total_despues = resultado[resultado['Tipo'] == 'Activo']['Valor Asignado (USD B)'].sum()
                    ganancia_antes = total_activos * tasa_objetivo_actual / 100
                    ganancia_despues = activos_total_despues * tasa_optimizada / 100

                    comparativa = {
                        "Indicador": [
                            "Duración Promedio Activos (años)",
                            "Duración Promedio Pasivos (años)",
                            "Tasa Promedio (%)",
                            "Liquidez (%)",
                            "Valor en Riesgo (VaR 95%)",
                            
                        ],
                        "Antes": [
                            dur_act_orig,
                            dur_pas_orig,
                            tasa_objetivo_actual,
                            porcentaje_liquidez_actual,
                            var_orig,
                            
                        ],
                        "Después": [
                            resumen['Duración Promedio Activos (años)'],
                            resumen['Duración Promedio Pasivos (años)'],
                            tasa_optimizada,
                            resumen['Liquidez % Después'],
                            resumen['Valor en Riesgo (VaR 95%) USD B'],
                            
                        ],
                        "Monto USD Antes": [
                            None,
                            None,
                            tasa_objetivo_actual * total_activos / 100,
                            porcentaje_liquidez_actual * total_activos / 100,
                            var_orig * 1_000,  # porque originalmente estaba en B
                            
                        ],
                        "Monto USD Después": [
                            None,
                            None,
                            tasa_optimizada * activos_total_despues / 100,
                            resumen['Liquidez % Después'] * activos_total_despues / 100,
                            resumen['Valor en Riesgo (VaR 95%) USD B'] * 1_000,  # convertir de B a M
                            
                        ]
                    }



                    df_comp = pd.DataFrame(comparativa)
                    df_comp["Diferencia Absoluta"] = df_comp["Después"] - df_comp["Antes"]
                    df_comp["% Antes sobre Total"] = (df_comp["Antes"] / total_activos) * 100
                    df_comp["% Después sobre Total"] = (df_comp["Después"] / total_activos) * 100

                    st.dataframe(df_comp.style.format({
                        "Antes": "{:.2f}",
                        "Después": "{:.2f}",
                        "Diferencia Absoluta": "{:.2f}",
                        "% Antes sobre Total": "{:.2f}%",
                        "% Después sobre Total": "{:.2f}%"
                    }))

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

        st.subheader("Resultado de la Simulación")
        st.dataframe(df_sim[['Categoría', 'Tasa (%)', 'Tasa Simulada (%)', 'Interés Estimado (USD B)']])

        st.subheader("Interés Estimado por Categoría")
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        df_sim_sorted = df_sim.sort_values('Interés Estimado (USD B)', ascending=False).head(5)
        impacto_total = df_sim_sorted['Interés Estimado (USD B)'].sum()

        df_sim_sorted = pd.concat([
            df_sim_sorted,
            pd.DataFrame({'Categoría': ['Impacto Total'], 'Interés Estimado (USD B)': [impacto_total]})
        ], ignore_index=True)

        bars2 = ax2.bar(df_sim_sorted['Categoría'], df_sim_sorted['Interés Estimado (USD B)'])

        for i, row in df_sim_sorted.iterrows():
            ax2.text(
                bars2[i].get_x() + bars2[i].get_width() / 2,
                row['Interés Estimado (USD B)'] + 0.01,
                f"${row['Interés Estimado (USD B)']:.1f}B",
                ha='center',
                va='bottom',
                fontsize=9
            )

        ax2.set_ylabel("Interés Estimado (USD B)")
        ax2.set_title("Top 5 Categorías + Impacto Total de la Simulación de Tasas")
        st.pyplot(fig2)

        st.subheader("Impacto Total en Interés Estimado")
        st.metric(label="Impacto Total Estimado (USD B)", value=f"{impacto_total:.2f}B")

    # ===========================
    # PESTAÑA: ANÁLISIS DE SENSIBILIDAD
    # ===========================
    with sens_tab:
        st.header("Análisis de Sensibilidad a Cambios en la Tasa")

        cambios = [-2.0, -1.0, 0.0, 1.0, 2.0]
        resumenes = []

        for delta in cambios:
            df_escenario = simular_escenario(resultado if 'resultado' in locals() else df, delta)
            activos_sim = df_escenario[df_escenario['Tipo'] == 'Activo']
            pasivos_sim = df_escenario[df_escenario['Tipo'] == 'Pasivo']
            impacto_total = activos_sim['Interés Estimado (USD B)'].sum() - abs(pasivos_sim['Interés Estimado (USD B)'].sum())

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
