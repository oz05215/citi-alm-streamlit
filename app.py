# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from modules.optimizer import run_optimization, simular_escenario
from modules.optimizer import PARAM_DESCRIPTION

# Configuración inicial
st.set_page_config(page_title="Citi ALM Optimizer", layout="wide")
st.title("Optimización y Simulación ALM - Citigroup - Por Oscar Zeledon - ADEN Business School")

# ========================
# Carga de archivo
# ========================
st.sidebar.header("📂 Cargar archivo de entrada")
uploaded_file = st.sidebar.file_uploader("Selecciona un archivo CSV", type="csv")

if uploaded_file:
    #df = pd.read_csv(uploaded_file)
    df = pd.read_csv(uploaded_file, encoding='utf-8')

    st.success("Archivo cargado correctamente ✅")

    # Crear pestañas
    opt_tab, sim_tab, sens_tab = st.tabs([
        "🔧 Optimización de Portafolio",
        "📈 Simulación de Tasas",
        "📊 Análisis de Sensibilidad"
    ])

    # ===========================
    # PESTAÑA: OPTIMIZACIÓN
    # ===========================
    with opt_tab:
        st.header("Parámetros de Optimización")
        col1, col2 = st.columns(2)
		with col1:
		    tasa_objetivo = st.slider("Tasa Objetivo Promedio (%)", 0.0, 10.0, 4.0)
		    st.caption(PARAM_DESCRIPTION["tasa_objetivo"])
		with col2:
		    liquidez_minima = st.number_input("Liquidez mínima requerida (USD millones)", value=100000.0)
		    st.caption(PARAM_DESCRIPTION["liquidez_minima"])


        # Rango de tasas posibles
        activos = df[df['Tipo'] == 'Activo']
        tasa_min = activos['Tasa (%)'].min()
        tasa_max = activos['Tasa (%)'].max()
        st.info(f"🔍 Rango de tasas posibles: entre **{tasa_min:.2f}%** y **{tasa_max:.2f}%**")

        if st.button("🚀 Ejecutar Optimización"):
            if tasa_objetivo < tasa_min or tasa_objetivo > tasa_max:
                st.error(f"La tasa objetivo de {tasa_objetivo}% está fuera del rango alcanzable.")
            else:
                resultado, resumen = run_optimization(df, tasa_objetivo, liquidez_minima)

                if "error" in resumen:
                    st.error(resumen["error"])
                else:
                    st.success("✅ Optimización exitosa")
                    st.subheader("Resumen de Optimización")
                    st.json(resumen)

                    st.subheader("Asignación Óptima")
                    st.dataframe(resultado)

                    # Gráfico Top 5 con % y montos
                    st.subheader("Visualización de la Asignación Óptima")
                    top_resultado = resultado.sort_values('Asignación Óptima (%)', ascending=False).head(5).reset_index(drop=True)

                    fig, ax = plt.subplots(figsize=(8, 4))
                    bars = ax.bar(top_resultado['Categoría'], top_resultado['Asignación Óptima (%)'])

                    for i, row in top_resultado.iterrows():
                        porcentaje = row['Asignación Óptima (%)']
                        monto = row['Valor Asignado (USD M)']
                        label = f"{porcentaje:.1f}% - ${monto:,.0f}M"
                        ax.text(
                            bars[i].get_x() + bars[i].get_width() / 2,
                            bars[i].get_height() + 1,
                            label,
                            ha='center',
                            va='bottom',
                            fontsize=9
                        )

                    ax.set_ylabel("Asignación (%)")
                    ax.set_title("Top 5 Asignaciones Óptimas por Categoría")
                    st.pyplot(fig)

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
        st.dataframe(df_sim[['Categoría', 'Tasa (%)', 'Tasa Simulada (%)', 'Interés Estimado (USD M)']])

        # Gráfico de interés estimado simulado
        st.subheader("Interés Estimado por Categoría")
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        df_sim_sorted = df_sim.sort_values('Interés Estimado (USD M)', ascending=False).head(5)
        bars2 = ax2.bar(df_sim_sorted['Categoría'], df_sim_sorted['Interés Estimado (USD M)'])

        for i, row in df_sim_sorted.iterrows():
            ax2.text(
                bars2[i - df_sim_sorted.index[0]].get_x() + bars2[i - df_sim_sorted.index[0]].get_width() / 2,
                row['Interés Estimado (USD M)'] + 1,
                f"${row['Interés Estimado (USD M)']:.1f}M",
                ha='center',
                va='bottom',
                fontsize=9
            )

        ax2.set_ylabel("Interés Estimado (USD M)")
        ax2.set_title("Top 5 Categorías por Interés Estimado Simulado")
        st.pyplot(fig2)

    # ===========================
    # PESTAÑA: ANÁLISIS DE SENSIBILIDAD
    # ===========================
    with sens_tab:
        st.header("Análisis de Sensibilidad a Cambios en la Tasa")
        cambios = [-2, -1, 0, 1, 2]
        resumenes = []

        for delta in cambios:
            df_escenario = simular_escenario(resultado if 'resultado' in locals() else df, delta)
            interes_total = df_escenario['Interés Estimado (USD M)'].sum()
            resumenes.append({
                "Cambio Tasa (%)": delta,
                "Interés Total Estimado (USD M)": round(interes_total, 2)
            })

        df_resumen_sens = pd.DataFrame(resumenes)
        st.dataframe(df_resumen_sens)

        # Gráfico de sensibilidad
        fig3, ax3 = plt.subplots(figsize=(8, 4))
        ax3.plot(df_resumen_sens["Cambio Tasa (%)"], df_resumen_sens["Interés Total Estimado (USD M)"], marker='o')
        ax3.set_title("Sensibilidad del Interés Total Estimado")
        ax3.set_xlabel("Cambio en la Tasa (%)")
        ax3.set_ylabel("Interés Estimado (USD M)")
        st.pyplot(fig3)

else:
    st.warning("⚠️ Por favor, carga un archivo CSV desde la barra lateral para comenzar.")
