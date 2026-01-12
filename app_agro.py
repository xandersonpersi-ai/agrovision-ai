import streamlit as st
import pandas as pd
from ultralytics import YOLO
from exif import Image as ExifImage
import folium
from streamlit_folium import st_folium
import os
from PIL import Image
import plotly.graph_objects as go
from datetime import datetime
import cv2  
import numpy as np

# 1. CONFIGURAÇÃO DE INTERFACE
st.set_page_config(page_title="AgroVision Pro | Intelligence", layout="wide")

# 2. INICIALIZAÇÃO DA MEMÓRIA (SESSION STATE)
if 'dados_analise' not in st.session_state:
    st.session_state.dados_analise = None

# CSS
st.markdown("""
    <style>
    iframe { width: 100% !important; border-radius: 15px; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 15px; border-top: 5px solid #2e7d32; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .loc-btn { display: inline-block; padding: 15px 25px; color: #fff; background-color: #68CAED; border: 3px solid #FF0000; border-radius: 12px; font-weight: bold; width: 100%; text-transform: uppercase; text-decoration: none; text-align: center; }
    .source-tag { font-size: 12px; padding: 4px 8px; border-radius: 5px; font-weight: bold; background-color: #e3f2fd; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return YOLO('best.pt' if os.path.exists('best.pt') else 'yolov8n.pt')

model = load_model()

# 3. SIDEBAR
st.sidebar.header("🕹️ Central de Comando")
modo_operacao = st.sidebar.radio("Selecione o Modo:", ["📂 Analisar Fotos", "🛸 Drone Real-Time"])

with st.sidebar.expander("📋 Cadastro de Campo", expanded=True):
    nome_fazenda = st.text_input("Propriedade", "Fazenda Santa Fé")
    nome_tecnico = st.text_input("Responsável Técnico", "Anderson Silva")
    tipo_plantio = st.selectbox("Cultura Atual", ["Soja", "Milho", "Algodão", "Cana", "Outros"])
    safra = st.text_input("Ciclo / Safra", "2025/2026")

# FUNÇÕES AUXILIARES
def extrair_gps_hibrido(img_file):
    try:
        img = ExifImage(img_file)
        if img.has_exif and hasattr(img, 'gps_latitude'):
            lat = (img.gps_latitude[0] + img.gps_latitude[1]/60 + img.gps_latitude[2]/3600) * (-1 if img.gps_latitude_ref == 'S' else 1)
            lon = (img.gps_longitude[0] + img.gps_longitude[1]/60 + img.gps_longitude[2]/3600) * (-1 if img.gps_longitude_ref == 'W' else 1)
            return lat, lon, "🛰️ Drone"
    except: pass
    return None, None, "📱 Celular"

# 4. MODO DRONE REAL-TIME
if modo_operacao == "🛸 Drone Real-Time":
    st.subheader("🎮 Live Stream: Monitoramento Aéreo")
    st.info("As fotos analisadas anteriormente continuam salvas na memória do sistema.")
    run_drone = st.toggle("🚀 ATIVAR CÂMERA DO DRONE")
    FRAME_WINDOW = st.image([]) 
    if run_drone:
        camera = cv2.VideoCapture(0)
        while run_drone:
            ret, frame = camera.read()
            if not ret: break
            results = model.predict(frame, conf=0.25, verbose=False)
            ann_frame = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(ann_frame)
        camera.release()

# 5. MODO ANALISAR FOTOS (COM MEMÓRIA)
else:
    uploaded_files = st.file_uploader("📂 ARRASTE AS FOTOS", accept_multiple_files=True, type=['jpg', 'jpeg', 'png'])

    if uploaded_files:
        temp_dados = []
        progresso = st.progress(0)
        for i, file in enumerate(uploaded_files):
            img = Image.open(file)
            results = model.predict(source=img, conf=0.25, verbose=False)
            img_plot = Image.fromarray(results[0].plot()[:, :, ::-1])
            file.seek(0)
            lat, lon, fonte = extrair_gps_hibrido(file)
            temp_dados.append({
                "Amostra": file.name, "Pragas": len(results[0].boxes),
                "Lat": lat, "Lon": lon, "Fonte": fonte,
                "Cultura": tipo_plantio, "Safra": safra, "_img": img_plot
            })
            progresso.progress((i + 1) / len(uploaded_files))
        
        # Salva na memória global da sessão
        st.session_state.dados_analise = pd.DataFrame(temp_dados)

    # Exibe os dados se eles existirem na memória (mesmo se mudar de aba e voltar)
    if st.session_state.dados_analise is not None:
        df = st.session_state.dados_analise
        
        st.markdown(f"### 📊 Relatório: {nome_fazenda}")
        col1, col2, col3 = st.columns(3)
        col1.metric("Cultura", tipo_plantio)
        col2.metric("Total Pragas", int(df['Pragas'].sum()))
        col3.metric("Média/Ponto", f"{df['Pragas'].mean():.1f}")

        # MAPA
        df_geo = df.dropna(subset=['Lat', 'Lon'])
        if not df_geo.empty:
            m = folium.Map(location=[df_geo['Lat'].mean(), df_geo['Lon'].mean()], zoom_start=18)
            for _, r in df_geo.iterrows():
                folium.CircleMarker([r['Lat'], r['Lon']], radius=10, color='red', fill=True).add_to(m)
            st_folium(m, use_container_width=True, height=400)

        # GALERIA
        st.markdown("---")
        for _, row in df.iterrows():
            c1, c2 = st.columns([1, 2])
            c1.image(row['_img'])
            c2.markdown(f"""
                <span class="source-tag">{row['Fonte']}</span>
                <h4>{row['Pragas']} Pragas</h4>
                <p>Cultura: {row['Cultura']} | Safra: {row['Safra']}</p>
            """, unsafe_allow_html=True)
            if row['Lat']:
                maps_url = f"https://www.google.com/maps?q={row['Lat']},{row['Lon']}"
                c2.markdown(f'<a href="{maps_url}" target="_blank" class="loc-btn">📍 LOCALIZAR</a>', unsafe_allow_html=True)
            st.markdown("---")
