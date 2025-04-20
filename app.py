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
    df = pd.read_csv(uploaded_file, encoding='utf-8')
    st.success("Archivo cargado correctamente ✅")

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
            liquidez_minima = st.number_input("Liquidez mínima requerida (USD millones)", value=100.0)
            st.caption(PARAM_DESCRIPTION["liquidez_minima"])

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
                    df_resumen = pd.DataFrame(list(resumen.items()), columns=["Indicador", "Valor"])
                    st.dataframe(df_resumen)

                    st.subheader("Asignación Óptima")
                    st.dataframe(resultado)

                    st.markdown("""
📘 **Descripción de las Categorías del Portafolio:**

- **Préstamos (Activo):** Son colocaciones de crédito que realiza el banco a clientes individuales o corporativos. Representan una de las principales fuentes de ingreso financiero, ya que generan intereses. Desde la perspectiva del banco, es un **activo generador de rentabilidad**, aunque con riesgo crediticio.

- **Inversiones (Activo):** Comprende instrumentos financieros como bonos, títulos públicos o privados que el banco mantiene para obtener rendimiento. Son activos utilizados para diversificar ingresos y gestionar excedentes de liquidez.

- **Efectivo (Activo):** Representa el dinero disponible en caja o en cuentas a la vista en el banco central. Es el activo más líquido, fundamental para cubrir necesidades inmediatas y cumplir con requerimientos regulatorios de reservas mínimas.

- **Depósitos (Pasivo):** Fondos que los clientes colocan en el banco, ya sea en cuentas corrientes, de ahorro o a plazo. Para el banco son un **pasivo**, ya que debe devolverlos en el futuro. Sin embargo, constituyen su principal fuente de fondeo.

- **Deuda (Pasivo):** Obligaciones que el banco asume mediante la emisión de bonos u otros instrumentos de financiamiento. Es un pasivo que permite captar capital en los mercados para financiar actividades del banco, aunque a un costo mayor que los depósitos.

✳️ *Estas categorías son clave para la gestión de activos y pasivos (ALM), reflejando cómo el banco obtiene y utiliza recursos para generar valor financiero de forma equilibrada y sostenible.*
""")

                    # Visualización: Duración
                    st.subheader("📊 Comparación de Duración Activos vs Pasivos")
                    fig_dur, ax_dur = plt.subplots(figsize=(6, 3))
                    ax_dur.bar(["Duración Activos", "Duración Pasivos"], [
                        resumen['Duración Promedio Activos (años)'],
                        resumen['Duración Promedio Pasivos (años)']
                    ])
                    ax_dur.set_ylabel("Años")
                    ax_dur.set_title("Duración Promedio")
                    st.pyplot(fig_dur)
                    st.markdown("📌 **Interpretación:** Esta gráfica compara la duración promedio ponderada entre activos y pasivos. "
                                "Un desajuste muy alto puede indicar sensibilidad al riesgo de tasa de interés. Idealmente, las barras deberían estar lo más alineadas posible.")

                    # Visualización: VaR
                    st.subheader("📉 Valor en Riesgo (VaR 95%)")
                    fig_var, ax_var = plt.subplots(figsize=(6, 3))
                    ax_var.bar(["VaR 95%"], [resumen['Valor en Riesgo (VaR 95%) USD M']])
                    ax_var.set_ylabel("USD Millones")
                    ax_var.set_title("VaR del Portafolio")
                    st.pyplot(fig_var)
                    st.markdown("📌 **Interpretación:** El VaR representa la pérdida máxima esperada en condiciones normales en el 95% de los casos. "
                                "Un VaR más bajo indica menor riesgo de pérdida bajo escenarios de estrés moderado.")

                    # Visualización: Top 5 asignaciones
                    st.subheader("🏦 Top 5 Asignaciones Óptimas por Categoría")
                    top_resultado = resultado.sort_values('Asignación Óptima (%)', ascending=False).head(5).reset_index(drop=True)

                    fig3, ax3 = plt.subplots(figsize=(6, 3))
                    bars = ax3.bar(top_resultado['Categoría'], top_resultado['Asignación Óptima (%)'])

                    for i, row in top_resultado.iterrows():
                        porcentaje = row['Asignación Óptima (%)']
                        monto = row['Valor Asignado (USD M)']
                        label = f"{porcentaje:.1f}% - ${monto:,.0f}M"
                        ax3.text(
                            bars[i].get_x() + bars[i].get_width() / 2,
                            bars[i].get_height() + 1,
                            label,
                            ha='center',
                            va='bottom',
                            fontsize=8
                        )

                    ax3.set_ylabel("Asignación (%)")
                    ax3.set_title("Top 5 por Asignación")
                    st.pyplot(fig3)
                    st.markdown("📌 **Interpretación:** Muestra las cinco categorías con mayor proporción asignada en el portafolio optimizado. "
                                "Útil para entender dónde se concentra el capital tras la optimización.")

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

        st.subheader("Interés Estimado por Categoría")
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        df_sim_sorted = df_sim.sort_values('Interés Estimado (USD M)', ascending=False).head(5)
        bars2 = ax2.bar(df_sim_sorted['Categoría'], df_sim_sorted['Interés Estimado (USD M)'])

        for i, row in df_sim_sorted.iterrows():
            ax2.text(
                bars2[i - df_sim_sorted.index[0]].get_x() + bars2[i - df_sim_sorted.index[0]].get_width() / 2,
                row['Interés Estimado (USD M)'] + 1,
                f"${row['Interés Estimado (USD M)']:.1f}M",
                ha='center',
                va='bottom',
                fontsize=8
            )

        ax2.set_ylabel("Interés Estimado (USD M)")
        ax2.set_title("Top 5 Categorías por Interés Simulado")
        st.pyplot(fig2)
        st.markdown("📌 **Interpretación:** Evalúa cómo cambiaría el ingreso financiero ante un aumento o disminución en las tasas. "
                    "Útil para análisis de sensibilidad de ingresos al riesgo de mercado.")

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

        fig3, ax3 = plt.subplots(figsize=(6, 3))
        ax3.plot(df_resumen_sens["Cambio Tasa (%)"], df_resumen_sens["Interés Total Estimado (USD M)"], marker='o')
        ax3.set_title("Sensibilidad del Interés Total Estimado")
        ax3.set_xlabel("Cambio en la Tasa (%)")
        ax3.set_ylabel("Interés Estimado (USD M)")
        st.pyplot(fig3)
        st.markdown("📌 **Interpretación:** Muestra cómo varía el ingreso total estimado ante cambios en tasas de interés. "
                    "Ayuda a visualizar la sensibilidad general del portafolio.")

else:
    st.warning("⚠️ Por favor, carga un archivo CSV desde la barra lateral para comenzar.")
