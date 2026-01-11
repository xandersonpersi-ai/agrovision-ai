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
st.title("AgroVision Pro AI 🛰️")
st.caption(f"Plataforma de Diagnóstico Digital | Sessão: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
st.markdown("---")

# 3. SIDEBAR
st.sidebar.header("📋 Cadastro de Campo")
nome_fazenda = st.sidebar.text_input("Propriedade", "Fazenda Santa Fé")
nome_tecnico = st.sidebar.text_input("Responsável Técnico", "Anderson Silva")
talhao_id = st.sidebar.text_input("Identificação do Talhão", "Talhão 01")
conf_threshold = st.sidebar.slider("Sensibilidade (Confidence)", 0.01, 1.0, 0.15)

# 4. FUNÇÃO GPS
def extrair_gps_st(img_file):
    try:
        img = ExifImage(img_file)
        if img.has_exif:
            lat = (img.gps_latitude[0] + img.gps_latitude[1]/60 + img.gps_latitude[2]/3600) * (-1 if img.gps_latitude_ref == 'S' else 1)
            lon = (img.gps_longitude[0] + img.gps_longitude[1]/60 + img.gps_longitude[2]/3600) * (-1 if img.gps_longitude_ref == 'W' else 1)
            return lat, lon
    except: return None
    return None

# 5. UPLOAD E PROCESSAMENTO IA
uploaded_files = st.file_uploader("📂 ARRASTE AS FOTOS DA VARREDURA", accept_multiple_files=True, type=['jpg', 'jpeg', 'png'])

if uploaded_files:
    model = YOLO('best.pt' if os.path.exists('best.pt') else 'yolov8n.pt')
    dados_lavoura = []
    progresso = st.progress(0)
    
    for i, file in enumerate(uploaded_files):
        try:
            img = Image.open(file)
            results = model.predict(source=img, conf=conf_threshold)
            
            # IA desenha na foto (RECUPERADO)
            img_com_caixas = results[0].plot() 
            img_com_caixas = Image.fromarray(img_com_caixas[:, :, ::-1])
            
            file.seek(0)
            coords = extrair_gps_st(file)
            num_pragas = len(results[0].boxes)
            
            dados_lavoura.append({
                "Amostra": file.name, 
                "Pragas": num_pragas,
                "Lat": coords[0] if coords else None, 
                "Lon": coords[1] if coords else None,
                "Imagem_Proc": img_com_caixas
            })
            progresso.progress((i + 1) / len(uploaded_files))
        except Exception: continue

    if dados_lavoura:
        df = pd.DataFrame(dados_lavoura)
        media_ponto = df['Pragas'].mean()

        # 6. MÉTRICAS
        k1, k2, k3 = st.columns(3)
        k1.metric("Amostras", len(df))
        k2.metric("Total Pragas", int(df['Pragas'].sum()))
        k3.metric("Média/Ponto", f"{media_ponto:.1f}")

        # 7. MAPA E GRÁFICOS
        col_mapa, col_intel = st.columns([1.6, 1])
        with col_mapa:
            st.subheader("📍 Mapa de Focos")
            df_geo = df.dropna(subset=['Lat', 'Lon'])
            if not df_geo.empty:
                m = folium.Map(location=[df_geo['Lat'].mean(), df_geo['Lon'].mean()], zoom_start=18)
                for _, row in df_geo.iterrows():
                    cor = 'red' if row['Pragas'] > 15 else 'green'
                    folium.CircleMarker([row['Lat'], row['Lon']], radius=10, color=cor, fill=True).add_to(m)
                st_folium(m, width="100%", height=500)

        with col_intel:
            st.subheader("📈 Análise Técnica")
            # Gauge Original
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number", value = media_ponto,
                gauge = {'axis': {'range': [None, 50]}, 'bar': {'color': "#1b5e20"}}
            ))
            fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Candlestick Top 10
            st.write("**🕯️ Volatilidade Top 10**")
            df_top10 = df.nlargest(10, 'Pragas')
            fig_candle = go.Figure(data=[go.Candlestick(
                x=df_top10['Amostra'], open=df_top10['Pragas']*0.9, high=df_top10['Pragas'],
                low=df_top10['Pragas']*0.7, close=df_top10['Pragas']*0.95,
                increasing_line_color='#991b1b', decreasing_line_color='#991b1b'
            )])
            fig_candle.update_layout(height=250, xaxis_rangeslider_visible=False, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig_candle, use_container_width=True)

        # 8. IMAGENS DAS DETECÇÕES (AS QUE VOCÊ SENTIU FALTA)
        st.markdown("---")
        st.subheader("📸 Galeria de Evidências (10 Pontos Críticos)")
        piores = df.nlargest(10, 'Pragas')
        for _, row in piores.iterrows():
            with st.container():
                st.image(row['Imagem_Proc'], caption=f"{row['Amostra']} - {row['Pragas']} pragas detectadas", use_container_width=True)
                st.markdown("---")

        # 9. DADOS E DOWNLOAD
        with st.expander("Ver Dados Brutos"):
            st.dataframe(df.drop(columns=['Imagem_Proc']), use_container_width=True)
            csv = df.drop(columns=['Imagem_Proc']).to_csv(index=False).encode('utf-8')
            st.download_button("📥 Baixar CSV", csv, "relatorio_agro.csv", "text/csv")
