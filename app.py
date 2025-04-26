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
    df = pd.read_csv(uploaded_file, encoding='utf-8')
    st.success("Archivo cargado correctamente ✅")

    opt_tab, sim_tab, sens_tab = st.tabs([
        "🔧 Optimización de Portafolio",
        "📈 Simulación de Tasas",
        "📊 Análisis de Sensibilidad"
    ])

    def format_b(val):
        return f"{val / 1e3:.3f}B"

    with opt_tab:
        st.header("Parámetros de Optimización")

        activos = df[df['Tipo'] == 'Activo']
        pasivos = df[df['Tipo'] == 'Pasivo']
        total_activos = activos['Monto (USD M)'].sum()

        tasa_promedio_actual = activos['Tasa (%)'].mean()
        ganancia_esperada_actual = total_activos * tasa_promedio_actual / 100

        dur_act_orig = np.sum(activos['Monto (USD M)'] * activos['Duración (años)']) / total_activos
        dur_pas_orig = np.sum(pasivos['Monto (USD M)'] * pasivos['Duración (años)']) / pasivos['Monto (USD M)'].sum()
        var_orig = np.std(activos['Tasa (%)'] / 100) * 1.65 * total_activos

        liquidez_monto_actual = activos[activos['Categoría'] == 'Efectivo']['Monto (USD M)'].sum()
        porcentaje_liquidez_actual = (liquidez_monto_actual / total_activos) * 100 if total_activos != 0 else 0

        st.markdown(f"**💵 Liquidez Actual: {porcentaje_liquidez_actual:.2f}% ({liquidez_monto_actual:.2f} M USD)**")
        st.markdown(f"**📈 Tasa Promedio Actual: {tasa_promedio_actual:.2f}%**")
        st.markdown(f"**💰 Ganancia Estimada Actual: {ganancia_esperada_actual:.2f} M USD**")

        st.markdown("### 📏 Tolerancias Globales (Escribe en %)")
        tolerancia_duracion = st.number_input("Tolerancia de Desbalance de Duración (%)", value=5.0)
        tolerancia_monto = st.number_input("Tolerancia de Variación del Monto Total (%)", value=5.0)

        st.markdown("### 🎯 Parámetros de Optimización Globales")
        tasa_min = float(np.min(activos['Tasa (%)']))
        tasa_max = float(np.max(activos['Tasa (%)']))
        tasa_objetivo = st.number_input("Tasa Objetivo Promedio (%)", min_value=tasa_min, max_value=tasa_max, value=round((tasa_min + tasa_max)/2, 2))
        porcentaje_liquidez_objetivo = st.number_input("Liquidez Objetivo (% del total activos)", 0.0, 100.0, 5.0)

        opcion_optimizar = st.selectbox(
            "¿Hacia dónde deseas optimizar el monto total?",
            options=["Cualquiera", "Subir Total", "Bajar Total"]
        )
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
                    optimizar_hacia_abajo
                )

                if "error" in resumen:
                    st.error("❌ Optimización no exitosa.")
                else:
                    st.success("✅ Optimización exitosa")

                    st.subheader("📈 Liquidez Después de Optimización")
                    st.write(f"💧 {resumen.get('Liquidez % Después', 0):.2f}% ({resumen.get('Liquidez Disponible (USD M)', 0):.2f} M USD)")

                    st.subheader("📋 Datos Comparativos - Original + Optimizado")
                    df_completo = df.copy()
                    df_completo = df_completo.merge(
                        resultado[['Categoría', 'Asignación Óptima (%)', 'Valor Asignado (USD M)']],
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
                    valores = [var_orig, resumen['Valor en Riesgo (VaR 95%) USD M']]
                    ax_var.bar(["Antes", "Después"], valores, color=["gray", "blue"])
                    for i, val in enumerate(valores):
                        ax_var.text(i, val + 0.01 * max(valores), format_b(val), ha='center')
                    ax_var.set_ylabel("USD Millones")
                    st.pyplot(fig_var)

                    st.subheader("🏦 Top 5 Asignaciones - Antes vs Después")
                    top_resultado = resultado.sort_values('Valor Asignado (USD M)', ascending=False).head(5)
                    categorias_top = top_resultado['Categoría'].tolist()
                    top_orig = df[df['Categoría'].isin(categorias_top)].copy()
                    top_orig = top_orig.set_index('Categoría').reindex(categorias_top).fillna(0).reset_index()
                    fig_top, ax_top = plt.subplots(figsize=(9, 5))
                    width = 0.35
                    x = np.arange(len(categorias_top))
                    ax_top.bar(x - width/2, top_orig['Monto (USD M)'], width, label="Antes", color='gray')
                    ax_top.bar(x + width/2, top_resultado['Valor Asignado (USD M)'], width, label="Después", color='royalblue')
                    ax_top.set_xticks(x)
                    ax_top.set_xticklabels(categorias_top, rotation=45)
                    ax_top.legend()
                    st.pyplot(fig_top)

                    st.subheader("📄 Comparativa de Resultados - Indicadores")

                    ganancia_estimada_despues = resumen['Ganancia Estimada Después (USD M)'] if 'Ganancia Estimada Después (USD M)' in resumen else 0.0

                    comparativa = {
                        "Indicador": [
                            "Duración Promedio Activos (años)",
                            "Duración Promedio Pasivos (años)",
                            "Tasa Promedio (%)",
                            "Liquidez (%)",
                            "Liquidez (USD M)",
                            "Ganancia Estimada (USD M)",
                            "Valor en Riesgo (VaR 95%)"
                        ],
                        "Antes": [
                            dur_act_orig,
                            dur_pas_orig,
                            tasa_promedio_actual,
                            porcentaje_liquidez_actual,
                            liquidez_monto_actual,
                            ganancia_esperada_actual,
                            var_orig
                        ],
                        "Después": [
                            resumen['Duración Promedio Activos (años)'],
                            resumen['Duración Promedio Pasivos (años)'],
                            resumen['Tasa Promedio del Portafolio (%)'],
                            resumen['Liquidez % Después'],
                            resumen['Liquidez Disponible (USD M)'],
                            ganancia_estimada_despues,
                            resumen['Valor en Riesgo (VaR 95%) USD M']
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

        fig3, ax3 = plt.subplots(figsize=(8, 4))
        ax3.plot(df_resumen_sens["Cambio Tasa (%)"], df_resumen_sens["Interés Total Estimado (USD M)"], marker='o')
        ax3.set_title("Sensibilidad del Interés Total Estimado")
        ax3.set_xlabel("Cambio en la Tasa (%)")
        ax3.set_ylabel("Interés Estimado (USD M)")
        st.pyplot(fig3)


else:
    st.warning("⚠️ Por favor, carga un archivo CSV para comenzar.")
