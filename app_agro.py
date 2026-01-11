import streamlit as st
import pandas as pd
from ultralytics import YOLO
from exif import Image as ExifImage
import folium
from streamlit_folium import st_folium
import os
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# 1. CONFIGURAÇÃO DE INTERFACE "PREMIUM"
st.set_page_config(page_title="AgroVision Pro | Intelligence", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f4f7f6; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 15px; border-top: 5px solid #2e7d32; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    div[data-testid="stExpander"] { background-color: #ffffff; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# 2. CABEÇALHO DINÂMICO
col_logo, col_tit = st.columns([1, 6])
with col_tit:
    st.title("AgroVision Pro AI 🛰️")
    st.caption(f"Plataforma de Diagnóstico Digital | Sessão: {datetime.now().strftime('%d/%m/%Y %H:%M')}")

st.markdown("---")

# 3. FICHA TÉCNICA E CONTROLE (SIDEBAR)
st.sidebar.header("📋 Cadastro de Campo")
with st.sidebar.expander("Identificação", expanded=True):
    nome_fazenda = st.text_input("Propriedade", "Fazenda Santa Fé")
    nome_tecnico = st.text_input("Responsável Técnico", "Seu Nome")
    tipo_plantio = st.selectbox("Cultura Atual", ["Soja", "Milho", "Algodão", "Cana", "Outros"])
    safra = st.text_input("Ciclo / Safra", "2025/2026")
    talhao_id = st.text_input("Identificação do Talhão", "Talhão 01")

with st.sidebar.expander("Configurações de IA"):
    conf_threshold = st.slider("Sensibilidade (Confidence)", 0.01, 1.0, 0.15)
    st.info("Ajuste a sensibilidade se a IA estiver ignorando pragas pequenas.")

# 4. FUNÇÃO DE GEOLOCALIZAÇÃO
def extrair_gps_st(img_file):
    try:
        img = ExifImage(img_file)
        if img.has_exif:
            lat = (img.gps_latitude[0] + img.gps_latitude[1]/60 + img.gps_latitude[2]/3600) * (-1 if img.gps_latitude_ref == 'S' else 1)
            lon = (img.gps_longitude[0] + img.gps_longitude[1]/60 + img.gps_longitude[2]/3600) * (-1 if img.gps_longitude_ref == 'W' else 1)
            return lat, lon
    except: return None
    return None

# 5. ÁREA DE UPLOAD BLINDADA (Bloqueia PDF e arquivos estranhos)
uploaded_files = st.file_uploader(
    "📂 ARRASTE AS FOTOS DA VARREDURA (Apenas JPG, PNG)", 
    accept_multiple_files=True, 
    type=['jpg', 'jpeg', 'png']
)

if uploaded_files:
    # Carregamento do Modelo
    model = YOLO('best.pt' if os.path.exists('best.pt') else 'yolov8n.pt')
    dados_lavoura = []
    
    st.write("### ⚙️ Processando Inteligência Artificial...")
    progresso = st.progress(0)
    
    for i, file in enumerate(uploaded_files):
        try:
            img = Image.open(file)
            results = model.predict(source=img, conf=conf_threshold)
            
            file.seek(0)
            coords = extrair_gps_st(file)
            num_pragas = len(results[0].boxes)
            
            dados_lavoura.append({
                "Amostra": file.name, 
                "Pragas": num_pragas,
                "Lat": coords[0] if coords else None, 
                "Lon": coords[1] if coords else None
            })
            progresso.progress((i + 1) / len(uploaded_files))
        except Exception: continue

    if dados_lavoura:
        df = pd.DataFrame(dados_lavoura)
        total_encontrado = df['Pragas'].sum()
        media_ponto = df['Pragas'].mean()

        # 6. MÉTRICAS DE IMPACTO (KPIs)
        st.markdown(f"### 📊 Sumário Executivo: {nome_fazenda}")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Técnico", nome_tecnico)
        k2.metric("Cultura", tipo_plantio)
        k3.metric("Total Detectado", f"{total_encontrado} un")
        status_sanitario = "CRÍTICO" if total_encontrado > 20 else "NORMAL"
        k4.metric("Status Sanitário", status_sanitario, delta="Alerta" if status_sanitario == "CRÍTICO" else "Ok")

        st.markdown("---")

        # 7. MAPA E CENTRO DE INTELIGÊNCIA
        col_mapa, col_intel = st.columns([1.6, 1])
        
        with col_mapa:
            st.subheader("📍 Georreferenciamento de Pragas")
            df_geo = df.dropna(subset=['Lat', 'Lon'])
            if not df_geo.empty:
                m = folium.Map(location=[df_geo['Lat'].mean(), df_geo['Lon'].mean()], zoom_start=18, tiles='OpenStreetMap')
                for _, row in df_geo.iterrows():
                    cor_ponto = 'red' if row['Pragas'] > 15 else 'orange' if row['Pragas'] > 5 else 'green'
                    folium.CircleMarker(
                        location=[row['Lat'], row['Lon']],
                        radius=10 + row['Pragas'],
                        color=cor_ponto, fill=True, fill_opacity=0.7,
                        popup=f"Amostra: {row['Amostra']}<br>Detectado: {row['Pragas']} pragas"
                    ).add_to(m)
                st_folium(m, width="100%", height=500)
            else:
                st.warning("⚠️ Fotos sem metadados de GPS detectadas.")

        with col_intel:
            st.subheader("📈 Análise de Pressão")
            
            # Gráfico de Velocímetro (Gauge)
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = media_ponto,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Média de Pragas / Ponto", 'font': {'size': 18}},
                gauge = {
                    'axis': {'range': [None, 50]},
                    'bar': {'color': "#1b5e20"},
                    'steps': [
                        {'range': [0, 15], 'color': "#c8e6c9"},
                        {'range': [15, 30], 'color': "#fff9c4"},
                        {'range': [30, 50], 'color': "#ffcdd2"}]
                }
            ))
            fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Ranking de Infestação
            st.write("**Top 5 Pontos de Maior Infestação**")
            fig_ranking = px.bar(df.nlargest(5, 'Pragas'), x='Pragas', y='Amostra', orientation='h', 
                                 color='Pragas', color_continuous_scale='Reds')
            fig_ranking.update_layout(height=250, showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_ranking, use_container_width=True)

        # 8. RECOMENDAÇÃO TÉCNICA AUTOMATIZADA
        st.markdown("---")
        with st.container():
            st.subheader("💡 Recomendação de Manejo (IA)")
            rec_col1, rec_col2 = st.columns([1, 3])
            
            with rec_col1:
                if status_sanitario == "CRÍTICO":
                    st.error("ALTA INFESTAÇÃO")
                else:
                    st.success("BAIXA INFESTAÇÃO")
            
            with rec_col2:
                if total_encontrado > 20:
                    st.write(f"**Atenção {nome_tecnico}:** O talhão **{talhao_id}** apresenta focos severos. Recomenda-se a aplicação localizada (Taxa Variável) nos pontos vermelhos indicados no mapa para otimizar o uso de defensivos na cultura de {tipo_plantio}.")
                else:
                    st.write(f"Os níveis de infestação em **{nome_fazenda}** estão dentro do limite tolerável. Continue o monitoramento semanal.")

        # 9. TABELA DE DADOS PARA EXPORTAÇÃO
        with st.expander("Ver Dados Brutos"):
            st.dataframe(df, use_container_width=True)
            st.download_button("📥 Exportar Relatório CSV", df.to_csv(index=False).encode('utf-8'), f"Relatorio_{nome_fazenda}.csv", "text/csv")
else:
from fpdf import FPDF

def gerar_pdf(df_dados, fazenda, tecnico):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, "Relatório de Monitoramento AgroVision Pro", ln=True, align='C')
    
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(200, 10, f"Propriedade: {fazenda}", ln=True)
    pdf.cell(200, 10, f"Responsável Técnico: {tecnico}", ln=True)
    pdf.cell(200, 10, f"Data do Diagnóstico: {datetime.now().strftime('%d/%m/%Y')}", ln=True)
    
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, "Resumo de Amostragem:", ln=True)
    pdf.set_font("Arial", size=10)
    
    for index, row in df_dados.iterrows():
        pdf.cell(200, 8, f"- Amostra {row['Amostra']}: {row['Pragas']} pragas detectadas", ln=True)
    
    return pdf.output(dest='S').encode('latin-1')

# Botão de Download na Interface
pdf_file = gerar_pdf(df, nome_fazenda, nome_tecnico)
st.download_button(
    label="📄 Baixar Relatório Técnico para o Produtor",
    data=pdf_file,
    file_name=f"Relatorio_{nome_fazenda}.pdf",
    mime="application/pdf"
)
    st.info("💡 Dica: Arraste as fotos de inspeção do seu drone ou celular para gerar o diagnóstico em tempo real.")
