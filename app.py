"""
Sistema de Apoyo a la Decisión - Método ANP (Analytic Network Process)
Aplicación Streamlit para análisis de decisión multicriterio
"""

# ===============================
# IMPORTAR LIBRERÍAS
# ===============================
import streamlit as st
import pandas as pd
pd.set_option("styler.render.max_elements", 2000000)   # ← ESTA LÍNEA NUEVA
import numpy as np
import itertools
from fpdf import FPDF

# ===============================
# IMPORTAR FUNCIONES DEL MÉTODO ANP
# ===============================
from anp import (
    crear_matriz,
    llenar_matriz,
    resumen_resultados,
    construir_supermatriz,
    supermatriz_ponderada,
    supermatriz_limite,
    ranking_final
)

# ===============================
# CONFIGURACIÓN DE LA PÁGINA
# ===============================
st.set_page_config(
    page_title="Sistema ANP - Apoyo a la Decisión",
    page_icon="🎯",
    layout="wide"
)

# ===============================
# INICIALIZAR SESSION STATE
# ===============================
if "df" not in st.session_state:
    st.session_state["df"] = None
if "problema" not in st.session_state:
    st.session_state["problema"] = ""
if "alternativa" not in st.session_state:
    st.session_state["alternativa"] = ""
if "criterios" not in st.session_state:
    st.session_state["criterios"] = []
if "tipos" not in st.session_state:
    st.session_state["tipos"] = {}

# ===============================
# ESTILOS CSS
# ===============================
st.markdown("""
<style>
    .stApp {
        background: #F4F8FB;
    }
    h1 {
        color: #0B5394;
        text-align: center;
    }
    h2 {
        color: #1F77B4;
    }
    h3 {
        color: #1F77B4;
    }
    div[data-testid="stButton"] button {
        background: #1976D2;
        color: white;
        border-radius: 10px;
        height: 50px;
        font-size: 16px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ===============================
# SIDEBAR
# ===============================
st.sidebar.title("🎯 Menú")
st.sidebar.success("Método ANP")
st.sidebar.markdown("---")
st.sidebar.write("📋 **Pasos:**")
st.sidebar.write("1️⃣ Cargar Base")
st.sidebar.write("2️⃣ Alternativas")
st.sidebar.write("3️⃣ Criterios")
st.sidebar.write("4️⃣ Costo/Beneficio")
st.sidebar.write("5️⃣ Comparaciones Saaty")
st.sidebar.write("6️⃣ Dependencias ANP")
st.sidebar.write("7️⃣ Supermatriz")
st.sidebar.write("8️⃣ Resultado Final")

# ===============================
# TÍTULO PRINCIPAL
# ===============================
st.title("🎯 Sistema de Apoyo a la Decisión")
st.subheader("Método ANP (Analytic Network Process)")
st.markdown("---")
st.info("Esta aplicación resuelve problemas de decisión multicriterio mediante el método ANP.")

# ============================================================
# PASO 1: DEFINICIÓN DEL PROBLEMA
# ============================================================
st.header("1️⃣ Definición del Problema")
problema = st.text_input("Ingrese el nombre del problema:", value=st.session_state["problema"])
if problema:
    st.session_state["problema"] = problema
st.markdown("---")

# ============================================================
# PASO 2: CARGAR BASE DE DATOS
# ============================================================
st.header("2️⃣ Cargar Base de Datos")
archivo = st.file_uploader("Seleccione un archivo CSV o Excel:", type=["csv", "xlsx"])

if archivo is not None:
    try:
        if archivo.name.endswith(".csv"):
            try:
                df = pd.read_csv(archivo, sep=None, engine="python", encoding="utf-8", on_bad_lines="skip")
            except:
                archivo.seek(0)
                try:
                    df = pd.read_csv(archivo, sep=";", encoding="latin1", on_bad_lines="skip")
                except:
                    archivo.seek(0)
                    df = pd.read_csv(archivo, sep=",", encoding="latin1", on_bad_lines="skip")
        else:
            df = pd.read_excel(archivo)
        
        st.session_state["df"] = df
        st.success("✅ Base cargada correctamente")
        st.subheader("📊 Vista Previa")
        st.dataframe(df.head(10), use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Filas", df.shape[0])
        col2.metric("Columnas", df.shape[1])
        col3.metric("Variables", len(df.columns))
        
    except Exception as e:
        st.error("❌ No fue posible leer el archivo.")
        st.exception(e)
else:
    st.warning("⚠️ Seleccione un archivo para continuar.")

# ============================================================
# PASO 3 y 4: ALTERNATIVAS Y CRITERIOS
# ============================================================
if st.session_state["df"] is not None:
    df = st.session_state["df"]
    
    st.markdown("---")
    st.header("3️⃣ Selección de Alternativas")
    alternativa = st.selectbox("Columna de alternativas:", df.columns)
    st.session_state["alternativa"] = alternativa
    st.success(f"✅ Alternativa: **{alternativa}**")
    
    st.markdown("---")
    st.header("4️⃣ Selección de Criterios")
    
    columnas_disponibles = [c for c in df.columns if c != alternativa]
    columnas_numericas = [c for c in columnas_disponibles if pd.api.types.is_numeric_dtype(df[c])]
    
    if len(columnas_numericas) == 0:
        st.error("❌ No hay columnas numéricas disponibles.")
    else:
        criterios = st.multiselect("Seleccione criterios (mínimo 2):", columnas_numericas)
        
        if len(criterios) >= 2:
            st.session_state["criterios"] = criterios
            st.success(f"✅ {len(criterios)} criterios seleccionados.")
            
            st.markdown("---")
            st.header("5️⃣ Tipo de Criterio")
            
            tipos = {}
            for criterio in criterios:
                tipos[criterio] = st.radio(
                    f"**{criterio}**:", ["Beneficio", "Costo"],
                    horizontal=True, key=f"tipo_{criterio}"
                )
            st.session_state["tipos"] = tipos
            
            # Resumen
            st.markdown("---")
            st.subheader("📋 Resumen")
            resumen = pd.DataFrame({"Criterio": criterios, "Tipo": [tipos[c] for c in criterios]})
            st.dataframe(resumen, use_container_width=True)
            st.success("✅ Listo para continuar.")
        else:
            st.warning("⚠️ Seleccione al menos 2 criterios.")

# ============================================================
# PASO 6: COMPARACIONES POR PARES (SAATY)
# ============================================================
if "criterios" in st.session_state and len(st.session_state["criterios"]) >= 2:
    criterios = st.session_state["criterios"]
    
    st.markdown("---")
    st.header("6️⃣ Comparaciones por Pares (Saaty)")
    
    with st.expander("ℹ️ Escala de Saaty"):
        st.write("1=Igual | 3=Moderada | 5=Fuerte | 7=Muy fuerte | 9=Extrema")
    
    pares = list(itertools.combinations(criterios, 2))
    comparaciones = {}
    
    for c1, c2 in pares:
        col1, col2, col3 = st.columns([2, 0.5, 2])
        with col1:
            st.markdown(f"**{c1}**")
        with col2:
            st.markdown("<center><b>vs</b></center>", unsafe_allow_html=True)
        with col3:
            st.markdown(f"**{c2}**")
        
        comparaciones[(c1, c2)] = st.select_slider(
            "¿Cuál es más importante?",
            options=[1, 2, 3, 4, 5, 6, 7, 8, 9],
            value=1, key=f"saaty_{c1}_{c2}"
        )
        st.markdown("---")
    
    st.session_state["comparaciones"] = comparaciones
    st.session_state["pares"] = pares
    
    if st.button("🔨 Construir Matriz de Comparación", type="primary"):
        try:
            valores = [comparaciones[par] for par in pares]
            matriz = crear_matriz(len(criterios))
            matriz = llenar_matriz(matriz, valores)
            resultados = resumen_resultados(matriz)
            
            st.session_state["matriz_saaty"] = matriz
            st.session_state["resultados_saaty"] = resultados
            
            st.markdown("---")
            st.subheader("📊 Matriz de Comparación")
            st.dataframe(
                pd.DataFrame(resultados["matriz"], index=criterios, columns=criterios)
                .style.format("{:.3f}"), use_container_width=True
            )
            
            st.subheader("📊 Matriz Normalizada")
            st.dataframe(
                pd.DataFrame(resultados["normalizada"], index=criterios, columns=criterios)
                .style.format("{:.4f}"), use_container_width=True
            )
            
            st.subheader("📋 Prioridades")
            prioridades_df = pd.DataFrame({
                "Criterio": criterios,
                "Peso": resultados["prioridades"]
            }).sort_values("Peso", ascending=False)
            
            st.dataframe(
                prioridades_df.style.format({"Peso": "{:.4f}"})
                .background_gradient(subset=["Peso"], cmap="Blues"),
                use_container_width=True
            )
            st.bar_chart(prioridades_df.set_index("Criterio")["Peso"])
            
            # Consistencia
            st.markdown("---")
            st.subheader("🔍 Consistencia")
            col1, col2, col3 = st.columns(3)
            col1.metric("λ Máximo", f"{resultados['lambda']:.4f}")
            col2.metric("CI", f"{resultados['ci']:.4f}")
            col3.metric("CR", f"{resultados['cr']:.4f}")
            
            if resultados["consistente"]:
                st.success("✅ Matriz CONSISTENTE (CR ≤ 0.10)")
                st.balloons()
                
                ranking = prioridades_df.copy()
                ranking["Posición"] = range(1, len(ranking) + 1)
                ranking = ranking[["Posición", "Criterio", "Peso"]]
                st.session_state["ranking"] = ranking
            else:
                st.error("❌ Matriz NO consistente (CR > 0.10). Revise las comparaciones.")
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.exception(e)
    
    elif "resultados_saaty" in st.session_state:
        st.info("📌 Matriz ya calculada.")
        resultados = st.session_state["resultados_saaty"]
        if resultados["consistente"]:
            st.success(f"✅ Consistente (CR = {resultados['cr']:.4f})")

# ============================================================
# PASO 7: DEPENDENCIAS ANP
# ============================================================
if "ranking" in st.session_state and "criterios" in st.session_state:
    criterios = st.session_state["criterios"]
    
    st.markdown("---")
    st.header("7️⃣ Dependencias entre Criterios (ANP)")
    st.write("Evalúe cómo influyen los demás criterios sobre cada criterio.")
    
    dependencias = {}
    
    for objetivo in criterios:
        st.markdown("---")
        st.subheader(f"🎯 Influencia sobre: **{objetivo}**")
        
        influyentes = [c for c in criterios if c != objetivo]
        
        if len(influyentes) < 2:
            st.info(f"Solo hay un criterio que influye. Peso = 1.0")
            dependencias[objetivo] = pd.DataFrame({"Criterio": influyentes, "Peso": [1.0]})
            continue
        
        pares_obj = list(itertools.combinations(influyentes, 2))
        comparaciones_obj = []
        
        for c1, c2 in pares_obj:
            col1, col2, col3 = st.columns([2, 0.5, 2])
            with col1:
                st.markdown(f"**{c1}**")
            with col2:
                st.markdown("<center><b>vs</b></center>", unsafe_allow_html=True)
            with col3:
                st.markdown(f"**{c2}**")
            
            valor = st.select_slider(
                f"¿Quién influye más sobre {objetivo}?",
                options=[1, 2, 3, 4, 5, 6, 7, 8, 9],
                value=1, key=f"dep_{objetivo}_{c1}_{c2}"
            )
            comparaciones_obj.append(valor)
            st.markdown("---")
        
        try:
            matriz_obj = crear_matriz(len(influyentes))
            matriz_obj = llenar_matriz(matriz_obj, comparaciones_obj)
            resultados_obj = resumen_resultados(matriz_obj)
            
            df_prioridad = pd.DataFrame({
                "Criterio": influyentes,
                "Peso": resultados_obj["prioridades"]
            })
            dependencias[objetivo] = df_prioridad
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Matriz:**")
                st.dataframe(
                    pd.DataFrame(resultados_obj["matriz"], index=influyentes, columns=influyentes)
                    .style.format("{:.3f}"), use_container_width=True
                )
            with col2:
                st.write("**Influencias:**")
                st.dataframe(
                    df_prioridad.style.format({"Peso": "{:.4f}"})
                    .background_gradient(subset=["Peso"], cmap="Oranges"),
                    use_container_width=True
                )
            
            if resultados_obj["consistente"]:
                st.success(f"✅ Consistente (CR = {resultados_obj['cr']:.4f})")
            else:
                st.warning(f"⚠️ Inconsistente (CR = {resultados_obj['cr']:.4f})")
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
    
    st.session_state["dependencias"] = dependencias
    st.success("✅ Dependencias registradas.")

# ============================================================
# PASO 8: SUPERMATRIZ
# ============================================================
if "dependencias" in st.session_state and "ranking" in st.session_state:
    st.markdown("---")
    st.header("8️⃣ Supermatriz ANP")
    
    if st.button("🔨 Construir Supermatriz", type="primary"):
        try:
            criterios = st.session_state["criterios"]
            prioridades_principales = st.session_state["ranking"]["Peso"].values
            dependencias = st.session_state["dependencias"]
            
            super_matriz = construir_supermatriz(criterios, prioridades_principales, dependencias)
            st.session_state["supermatriz"] = super_matriz
            
            st.subheader("📊 Supermatriz Original")
            st.dataframe(
                pd.DataFrame(super_matriz, index=criterios, columns=criterios)
                .style.format("{:.4f}").background_gradient(cmap="YlOrRd"),
                use_container_width=True
            )
            
            sumas = super_matriz.sum(axis=0)
            st.write("**Suma de columnas:**")
            for i, c in enumerate(criterios):
                icono = "✅" if abs(sumas[i] - 1.0) < 0.01 else "❌"
                st.write(f"{icono} {c}: {sumas[i]:.4f}")
            
            st.success("✅ Supermatriz construida.")
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.exception(e)
    
    elif "supermatriz" in st.session_state:
        st.info("📌 Supermatriz ya construida.")

# ============================================================
# PASO 9: SUPERMATRIZ PONDERADA
# ============================================================
if "supermatriz" in st.session_state:
    st.markdown("---")
    st.header("9️⃣ Supermatriz Ponderada")
    
    if st.button("⚖️ Calcular Ponderada", type="primary"):
        try:
            super_p = supermatriz_ponderada(st.session_state["supermatriz"])
            st.session_state["supermatriz_ponderada"] = super_p
            
            st.subheader("📊 Supermatriz Ponderada")
            st.dataframe(
                pd.DataFrame(super_p, index=st.session_state["criterios"], 
                           columns=st.session_state["criterios"])
                .style.format("{:.4f}").background_gradient(cmap="Blues"),
                use_container_width=True
            )
            st.success("✅ Listo.")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
    
    elif "supermatriz_ponderada" in st.session_state:
        st.info("📌 Ya calculada.")

# ============================================================
# PASO 10: SUPERMATRIZ LÍMITE
# ============================================================
if "supermatriz_ponderada" in st.session_state:
    st.markdown("---")
    st.header("🔟 Supermatriz Límite")
    
    if st.button("🔄 Calcular Límite", type="primary"):
        try:
            with st.spinner("Calculando..."):
                super_lim = supermatriz_limite(st.session_state["supermatriz_ponderada"])
                st.session_state["supermatriz_limite"] = super_lim
            
            criterios = st.session_state["criterios"]
            
            st.subheader("📊 Supermatriz Límite")
            st.dataframe(
                pd.DataFrame(super_lim, index=criterios, columns=criterios)
                .style.format("{:.6f}").background_gradient(cmap="Greens"),
                use_container_width=True
            )
            
            primera_col = super_lim[:, 0]
            convergio = all(np.allclose(primera_col, super_lim[:, j], atol=1e-6) 
                          for j in range(super_lim.shape[1]))
            
            if convergio:
                st.success("✅ Convergió correctamente.")
                
                prioridades_finales = pd.DataFrame({
                    "Criterio": criterios,
                    "Prioridad ANP": primera_col
                }).sort_values("Prioridad ANP", ascending=False)
                
                st.subheader("📋 Prioridades Finales ANP")
                st.dataframe(
                    prioridades_finales.style.format({"Prioridad ANP": "{:.6f}"})
                    .background_gradient(subset=["Prioridad ANP"], cmap="Greens"),
                    use_container_width=True
                )
                st.bar_chart(prioridades_finales.set_index("Criterio")["Prioridad ANP"])
                st.session_state["prioridades_anp"] = prioridades_finales
            else:
                st.warning("⚠️ Puede no haber convergido completamente.")
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
    
    elif "supermatriz_limite" in st.session_state:
        st.info("📌 Ya calculada.")

# ============================================================
# PASO 11: RANKING FINAL
# ============================================================
if "supermatriz_limite" in st.session_state and st.session_state["df"] is not None:
    st.markdown("---")
    st.header("🏆 Ranking Final de Alternativas")
    
    if st.button("🏆 Calcular Ranking Final", type="primary"):
        try:
            df = st.session_state["df"]
            criterios = st.session_state["criterios"]
            alternativa_col = st.session_state["alternativa"]
            tipos = st.session_state["tipos"]
            
            super_lim = st.session_state["supermatriz_limite"]
            pesos_criterios = super_lim[:, 0]
            
            st.subheader("📊 Pesos Finales")
            pesos_df = pd.DataFrame({
                "Criterio": criterios,
                "Peso ANP": pesos_criterios
            }).sort_values("Peso ANP", ascending=False)
            
            st.dataframe(
                pesos_df.style.format({"Peso ANP": "{:.4f}"})
                .background_gradient(subset=["Peso ANP"], cmap="Blues"),
                use_container_width=True
            )
            
            df_alt = df[[alternativa_col] + criterios].copy()
            df_alt[criterios] = df_alt[criterios].fillna(0)
            
            datos_norm = {}
            for criterio in criterios:
                valores = df_alt[criterio].astype(float).values
                
                if tipos[criterio] == "Beneficio":
                    max_val = valores.max()
                    datos_norm[criterio] = valores / max_val if max_val != 0 else np.zeros_like(valores)
                else:
                    valores_pos = np.where(valores > 0, valores, np.inf)
                    min_val = valores_pos.min()
                    if min_val == np.inf:
                        min_val = 1
                    datos_norm[criterio] = min_val / np.where(valores == 0, min_val, valores)
            
            matriz_datos = np.array([datos_norm[c] for c in criterios]).T
            scores = matriz_datos.dot(pesos_criterios)
            
            df_final = pd.DataFrame({
                "Alternativa": df_alt[alternativa_col],
                "Score": scores
            }).sort_values("Score", ascending=False).reset_index(drop=True)
            df_final["Posición"] = range(1, len(df_final) + 1)
            df_final = df_final[["Posición", "Alternativa", "Score"]]
            
            st.session_state["resultado_final"] = df_final
            
            st.subheader("🏆 Ranking Final")
            st.dataframe(
                df_final.style.format({"Score": "{:.4f}"})
                .background_gradient(subset=["Score"], cmap="RdYlGn"),
                use_container_width=True
            )
            st.bar_chart(df_final.set_index("Alternativa")["Score"])
            
            ganadora = df_final.iloc[0]
            st.markdown("---")
            st.subheader("🥇 Ganadora")
            st.metric(label="Mejor Alternativa", value=f"{ganadora['Alternativa']}",
                     delta=f"Score: {ganadora['Score']:.4f}")
            st.balloons()
            st.success("✅ ¡Análisis ANP completado!")
            
            # PDF
            st.markdown("---")
            if st.button("📥 Generar PDF"):
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", 'B', 16)
                pdf.cell(0, 10, "Reporte ANP", ln=True, align='C')
                pdf.ln(5)
                pdf.set_font("Arial", size=12)
                pdf.cell(0, 10, f"Problema: {st.session_state.get('problema', 'N/A')}", ln=True)
                pdf.cell(0, 10, f"Ganadora: {ganadora['Alternativa']}", ln=True)
                pdf.ln(5)
                pdf.set_font("Arial", size=10)
                for _, row in df_final.iterrows():
                    pdf.cell(0, 8, f"{row['Posicion']}. {row['Alternativa']} - {row['Score']:.4f}", ln=True)
                
                st.download_button(
                    label="📥 Descargar PDF",
                    data=pdf.output(dest='S').encode('latin-1', errors='replace'),
                    file_name="Reporte_ANP.pdf",
                    mime="application/pdf"
                )
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.exception(e)
    
    elif "resultado_final" in st.session_state:
        st.info("📌 Ranking ya calculado.")
        st.dataframe(
            st.session_state["resultado_final"].style.format({"Score": "{:.4f}"})
            .background_gradient(subset=["Score"], cmap="RdYlGn"),
            use_container_width=True
        )