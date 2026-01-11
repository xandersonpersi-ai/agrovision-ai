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

# 1. CONFIGURAÇÃO DE INTERFACE
st.set_page_config(page_title="AgroVision Pro | Intelligence", layout="wide", page_icon="🌱")

st.markdown("""
    <style>
    .main { background-color: #f4f7f6; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 12px; border-top: 5px solid #2e7d32; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .plotly-graph-div { min-height: 400px !important; }
    </style>
    """, unsafe_allow_html=True)

# 2. CABEÇALHO
st.title("AgroVision Pro AI 🛰️")
st.caption(f"Análise de Volatilidade e Diagnóstico Digital | {datetime.now().strftime('%d/%m/%Y %H:%M')}")
st.markdown("---")

# 3. SIDEBAR
st.sidebar.header("📋 Cadastro")
nome_fazenda = st.sidebar.text_input("Propriedade", "Fazenda Santa Fé")
conf_threshold = st.sidebar.slider("Sensibilidade IA", 0.01, 1.0, 0.15)

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

# 5. UPLOAD
uploaded_files = st.file_uploader("📂 ARRASTE AS FOTOS PARA ANÁLISE DE VELAS", accept_multiple_files=True, type=['jpg', 'jpeg', 'png'])

if uploaded_files:
    @st.cache_resource
    def load_yolo():
        return YOLO('best.pt' if os.path.exists('best.pt') else 'yolov8n.pt')
    
    model = load_yolo()
    dados_lavoura = []
    progresso = st.progress(0)
    
    for i, file in enumerate(uploaded_files):
        try:
            img = Image.open(file)
            results = model.predict(source=img, conf=conf_threshold)
            img_com_caixas = results[0].plot() 
            img_com_caixas = Image.fromarray(img_com_caixas[:, :, ::-1])
            
            file.seek(0)
            coords = extrair_gps_st(file)
            num_pragas = len(results[0].boxes)
            
            # Agrupamento para as Velas (Setores de 5 em 5 amostras)
            setor_idx = (i // 5) + 1
            
            dados_lavoura.append({
                "Amostra": f"Ponto {i+1:02d}", 
                "Setor": f"Setor {setor_idx:02d}",
                "Pragas": num_pragas,
                "Lat": coords[0] if coords else None, 
                "Lon": coords[1] if coords else None,
                "Imagem_Proc": img_com_caixas
            })
            progresso.progress((i + 1) / len(uploaded_files))
        except Exception: continue

    if dados_lavoura:
        df = pd.DataFrame(dados_lavoura)
        
        # --- LÓGICA DO GRÁFICO DE VELAS (CANDLESTICK) ---
        # Calculamos Min, Max, Primeiro e Último valor de cada setor
        df_candle = df.groupby('Setor')['Pragas'].agg(['min', 'max', 'first', 'last']).reset_index()

        st.subheader("🕯️ Volatilidade de Infestação por Setor")
        fig_candle = go.Figure(data=[go.Candlestick(
            x=df_candle['Setor'],
            open=df_candle['first'],
            high=df_candle['max'],
            low=df_candle['min'],
            close=df_candle['last'],
            increasing_line_color='#ef4444', # Vermelho: Infestação subindo no setor
            decreasing_line_color='#059669'  # Verde: Infestação baixando/estável
        )])
        
        fig_candle.update_layout(
            xaxis_rangeslider_visible=False,
            height=450,
            margin=dict(l=10, r=10, t=10, b=10),
            template="plotly_white"
        )
        st.plotly_chart(fig_candle, use_container_width=True)

        # 6. MÉTRICAS
        st.markdown("---")
        k1, k2, k3 = st.columns(3)
        k1.metric("Total Amostras", len(df))
        k2.metric("Pico no Talhão", f"{df['Pragas'].max()} un")
        k3.metric("Média Geral", f"{df['Pragas'].mean():.1f}")

        # 7. MAPA
        st.subheader("📍 Mapa de Calor")
        df_geo = df.dropna(subset=['Lat', 'Lon'])
        if not df_geo.empty:
            m = folium.Map(location=[df_geo['Lat'].mean(), df_geo['Lon'].mean()], zoom_start=18)
            for _, row in df_geo.iterrows():
                cor = 'red' if row['Pragas'] > 15 else 'green'
                folium.CircleMarker([row['Lat'], row['Lon']], radius=10, color=cor, fill=True).add_to(m)
            st_folium(m, width="100%", height=400)

        # 8. EVIDÊNCIAS DO TOP 10
        st.markdown("---")
        st.subheader("📸 Pontos Mais Críticos (Evidências)")
        piores = df.nlargest(10, 'Pragas')
        for _, row in piores.iterrows():
            st.image(row['Imagem_Proc'], caption=f"{row['Amostra']} ({row['Setor']}) - {row['Pragas']} pragas", use_container_width=True)

else:
    st.info("💡 Pronto para análise. Arraste as fotos para gerar o gráfico de velas.")
