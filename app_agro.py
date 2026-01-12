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

# CSS Corrigido (Sem erros de sintaxe)
st.markdown("""
    <style>
    iframe { width: 100% !important; border-radius: 15px; }
    .stMetric { 
        background-color: #ffffff; padding: 20px; border-radius: 15px; 
        border-top: 5px solid #2e7d32; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .loc-btn {
        display: inline-block; padding: 15px 25px; font-size: 16px; cursor: pointer;
        text-align: center; text-decoration: none; color: #fff; background-color: #68CAED;
        border: 3px solid #FF0000; border-radius: 12px; font-weight: bold; width: 100%;
        text-transform: uppercase; letter-spacing: 1px;
    }
    .source-tag {
        font-size: 12px; padding: 4px 8px; border-radius: 5px; font-weight: bold; background-color: #e3f2fd;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return YOLO('best.pt' if os.path.exists('best.pt') else 'yolov8n.pt')

model = load_model()

# 2. CABEÇALHO
st.title("AgroVision Pro AI 🛰️")
st.caption(f"Plataforma de Diagnóstico Digital | {datetime.now().strftime('%d/%m/%Y %H:%M')}")
st.markdown("---")

# 3. SIDEBAR
st.sidebar.header("🕹️ Central de Comando")
modo_operacao = st.sidebar.radio("Selecione o Modo:", ["📂 Analisar Fotos", "🛸 Drone Real-Time"])

with st.sidebar.expander("📋 Cadastro de Campo", expanded=True):
    nome_fazenda = st.text_input("Propriedade", "Fazenda Santa Fé")
    nome_tecnico = st.text_input("Responsável Técnico", "Anderson Silva")
    tipo_plantio = st.selectbox("Cultura Atual", ["Soja", "Milho", "Algodão", "Cana", "Outros"])
    safra = st.text_input("Ciclo / Safra", "2025/2026")
    talhao_id = st.text_input("Identificação do Talhão", "Talhão 01")

with st.sidebar.expander("⚙️ Configurações de IA"):
    conf_threshold = st.slider("Sensibilidade", 0.01, 1.0, 0.25)
    rtsp_url = st.text_input("URL do Stream (RTSP/IP)", "0")

# 4. FUNÇÃO GPS HÍBRIDA
def extrair_gps_hibrido(img_file):
    try:
        img = ExifImage(img_file)
        if img.has_exif and hasattr(img, 'gps_latitude'):
            lat = (img.gps_latitude[0] + img.gps_latitude[1]/60 + img.gps_latitude[2]/3600) * (-1 if img.gps_latitude_ref == 'S' else 1)
            lon = (img.gps_longitude[0] + img.gps_longitude[1]/60 + img.gps_longitude[2]/3600) * (-1 if img.gps_longitude_ref == 'W' else 1)
            return lat, lon, "🛰️ GPS/Drone"
    except:
        pass
    return None, None, "📱 Celular/Manual"

def link_google_maps(lat, lon):
    return f"https://www.google.com/maps?q={lat},{lon}" if lat else "#"

# 5. MODO DRONE REAL-TIME
if modo_operacao == "🛸 Drone Real-Time":
    st.subheader("🎮 Live Stream: Monitoramento Aéreo")
    run_drone = st.toggle("🚀 ATIVAR CÂMERA DO DRONE")
    FRAME_WINDOW = st.image([]) 
    
    if run_drone:
        cam_source = int(rtsp_url) if rtsp_url.isdigit() else rtsp_url
        camera = cv2.VideoCapture(cam_source)
        while run_drone:
            ret, frame = camera.read()
            if not ret:
                st.error("Falha ao receber imagem.")
                break
            results = model.predict(frame, conf=conf_threshold, verbose=False)
            ann_frame = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(ann_frame)
        camera.release()
    else:
        st.info("Sistema em Stand-by.")

# 6. MODO ANALISAR FOTOS
else:
    uploaded_files = st.file_uploader("📂 ARRASTE AS FOTOS (Drone ou Celular)", accept_multiple_files=True, type=['jpg', 'jpeg', 'png'])

    if uploaded_files:
        dados_lavoura = []
        progresso = st.progress(0)
        
        for i, file in enumerate(uploaded_files):
            try:
                img = Image.open(file)
                results = model.predict(source=img, conf=conf_threshold, verbose=False)
                img_com_caixas = Image.fromarray(results[0].plot()[:, :, ::-1])
                file.seek(0)
                
                lat, lon, fonte = extrair_gps_hibrido(file)
                
                dados_lavoura.append({
                    "Amostra": file.name, "Pragas": len(results[0].boxes),
                    "Latitude": lat, "Longitude": lon, "Fonte": fonte,
                    "Maps_Link": link_google_maps(lat, lon),
                    "Fazenda": nome_fazenda, "Safra": safra, "Cultura": tipo_plantio, 
                    "Tecnico": nome_tecnico, "_img_obj": img_com_caixas
                })
                progresso.progress((i + 1) / len(uploaded_files))
            except Exception as e:
                st.error(f"Erro ao processar {file.name}: {e}")

        if dados_lavoura:
            df = pd.DataFrame(dados_lavoura)
            media_ponto = df['Pragas'].mean()
            
            # KPIs PRINCIPAIS
            st.markdown(f"### 📊 Relatório: {nome_fazenda}")
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Responsável Técnico", nome_tecnico)
            k2.metric("Cultura", tipo_plantio)
            k3.metric("Safra", safra)
            k4.metric("Total Detectado", f"{int(df['Pragas'].sum())} un")
            k5.metric("Média por Ponto", f"{media_ponto:.1f}")

            st.markdown("---")
            
            # MAPA E ANÁLISE
            col_mapa, col_intel = st.columns([1.6, 1])
            with col_mapa:
                st.subheader("📍 Georreferenciamento (Drone)")
                df_geo = df.dropna(subset=['Latitude', 'Longitude'])
                if not df_geo.empty:
                    m = folium.Map(location=[df_geo['Latitude'].mean(), df_geo['Longitude'].mean()], zoom_start=18)
                    for _, row in df_geo.iterrows():
                        cor = 'red' if row['Pragas'] > 15 else 'orange' if row['Pragas'] > 5 else 'green'
                        folium.CircleMarker([row['Latitude'], row['Longitude']], radius=10, color=cor, fill=True, popup=f"{row['Pragas']} pragas").add_to(m)
                    st_folium(m, use_container_width=True, height=500, returned_objects=[])
                else:
                    st.warning("⚠️ Fotos sem coordenadas GPS detectadas.")

            with col_intel:
                st.subheader("📈 Pressão de Pragas")
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number", value = media_ponto,
                    gauge = {'axis': {'range': [0, 50]}, 'bar': {'color': "#1b5e20"}}))
                st.plotly_chart(fig_gauge, use_container_width=True)

            # PARECER TÉCNICO
            st.info(f"O técnico **{nome_tecnico}** analisou a cultura de **{tipo_plantio}** ({safra}). Média de **{media_ponto:.1f}** pragas/ponto.")

            # GALERIA LIMPA (Cultura e Safra apenas)
            st.markdown("---")
            st.subheader("📸 Detalhes dos Focos")
            for _, row in df.iterrows():
                g1, g2 = st.columns([1.5, 1])
                with g1: st.image(row['_img_obj'], use_container_width=True)
                with g2:
                    st.markdown(f"""
                    <div style="background: white; padding: 20px; border-radius: 15px; border: 1px solid #eee;">
                        <span class="source-tag">{row['Fonte']}</span>
                        <h3 style="margin-top:10px;">🪲 {row['Pragas']} Detectadas</h3>
                        <p><b>Amostra:</b> {row['Amostra']}</p>
                        <p><b>Cultura:</b> {row['Cultura']} | <b>Safra:</b> {row['Safra']}</p>
                        <hr>
                        {"<a href='"+row['Maps_Link']+"' target='_blank'><button class='loc-btn'>📍 LOCALIZAR</button></a>" if row['Latitude'] else "<i>GPS Indisponível</i>"}
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown("---")
