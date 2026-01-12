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

# 1. CONFIGURAÇÃO DE INTERFACE PREMIUM
st.set_page_config(page_title="AgroVision Pro | Intelligence", layout="wide")

st.markdown(f"""
    <style>
    @keyframes fadeInUp {{
        from {{ opacity: 0; transform: translateY(20px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    @keyframes neonPulseRed {{
        0% {{ box-shadow: 0 0 5px #FF0000, 0 0 10px #FF0000; }}
        50% {{ box-shadow: 0 0 20px #FF0000, 0 0 30px #FF0000; }}
        100% {{ box-shadow: 0 0 5px #FF0000, 0 0 10px #FF0000; }}
    }}
    .main {{ background-color: #f4f7f6; }}
    .stMetric {{ 
        background-color: #ffffff; padding: 20px; border-radius: 15px; 
        border-top: 5px solid #2e7d32; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }}
    .loc-btn {{
        display: inline-block; padding: 15px 25px; font-size: 16px; cursor: pointer;
        text-align: center; text-decoration: none; color: #fff; background-color: #68CAED;
        border: 3px solid #FF0000; border-radius: 12px; font-weight: bold; width: 100%;
        animation: neonPulseRed 1.5s infinite ease-in-out; text-transform: uppercase;
    }}
    .report-section {{ animation: fadeInUp 0.6s ease-out; }}
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return YOLO('best.pt' if os.path.exists('best.pt') else 'yolov8n.pt')

model = load_model()

# 2. CABEÇALHO
st.title("AgroVision Pro AI 🛰️")
st.caption(f"Diagnóstico Digital | {datetime.now().strftime('%d/%m/%Y %H:%M')}")
st.markdown("---")

# 3. SIDEBAR
st.sidebar.header("🕹️ Central de Comando")
modo_operacao = st.sidebar.radio("Modo de Operação:", ["📂 Analisar Fotos", "🛸 Drone Real-Time"])

with st.sidebar.expander("📋 Cadastro de Campo", expanded=True):
    nome_fazenda = st.text_input("Propriedade", "Fazenda Santa Fé")
    nome_tecnico = st.text_input("Responsável Técnico", "Anderson Silva")
    tipo_plantio = st.selectbox("Cultura Atual", ["Soja", "Milho", "Algodão", "Cana", "Outros"])
    safra = st.text_input("Ciclo / Safra", "2025/2026")
    talhao_id = st.text_input("Identificação do Talhão", "Talhão 01")

with st.sidebar.expander("⚙️ Configurações de IA"):
    conf_threshold = st.slider("Sensibilidade", 0.01, 1.0, 0.25)
    rtsp_url = st.text_input("URL do Stream (RTSP/IP)", "0")

# 4. FUNÇÕES AUXILIARES
def extrair_gps_st(img_file):
    try:
        img = ExifImage(img_file)
        if img.has_exif:
            lat = (img.gps_latitude[0] + img.gps_latitude[1]/60 + img.gps_latitude[2]/3600) * (-1 if img.gps_latitude_ref == 'S' else 1)
            lon = (img.gps_longitude[0] + img.gps_longitude[1]/60 + img.gps_longitude[2]/3600) * (-1 if img.gps_longitude_ref == 'W' else 1)
            return lat, lon
    except: return None
    return None

def link_google_maps(lat, lon):
    return f"https://www.google.com/maps?q={lat},{lon}" if lat != "N/A" else "#"

# 5. MODO DRONE
if modo_operacao == "🛸 Drone Real-Time":
    st.subheader("🎮 Live Stream: Monitoramento Aéreo")
    run_drone = st.toggle("🚀 INICIAR VOO (LIVE)")
    FRAME_WINDOW = st.image([]) 
    
    if run_drone:
        cam_source = int(rtsp_url) if rtsp_url.isdigit() else rtsp_url
        vid = cv2.VideoCapture(cam_source)
        while run_drone:
            ret, frame = vid.read()
            if not ret:
                st.error("Conexão perdida com o drone.")
                break
            results = model.predict(frame, conf=conf_threshold, verbose=False)
            ann_frame = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(ann_frame)
            if not run_drone: break
        vid.release()
    else:
        st.info("Aguardando ativação do drone...")

# 6. MODO FOTOS
else:
    uploaded_files = st.file_uploader("📂 CARREGAR IMAGENS", accept_multiple_files=True, type=['jpg', 'jpeg', 'png'])
    if uploaded_files:
        dados_lavoura = []
        progresso = st.progress(0)
        for i, file in enumerate(uploaded_files):
            img = Image.open(file)
            results = model.predict(source=img, conf=conf_threshold, verbose=False)
            img_com_caixas = Image.fromarray(results[0].plot()[:, :, ::-1])
            file.seek(0)
            coords = extrair_gps_st(file)
            lat, lon = (coords[0], coords[1]) if coords else ("N/A", "N/A")
            dados_lavoura.append({
                "Amostra": file.name, "Pragas": len(results[0].boxes),
                "Latitude": lat, "Longitude": lon, "Maps_Link": link_google_maps(lat, lon),
                "Fazenda": nome_fazenda, "Safra": safra, "Talhao": talhao_id,
                "Cultura": tipo_plantio, "Data": datetime.now().strftime('%d/%m/%Y'),
                "_img_obj": img_com_caixas
            })
            progresso.progress((i + 1) / len(uploaded_files))

        if dados_lavoura:
            df = pd.DataFrame(dados_lavoura)
            media_ponto = df['Pragas'].mean()
            status_sanitario = "CRÍTICO" if media_ponto > 15 else "NORMAL"

            # KPIs
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Propriedade", nome_fazenda)
            k2.metric("Talhão", talhao_id)
            k3.metric("Total Pragas", f"{int(df['Pragas'].sum())} un")
            k4.metric("Status", status_sanitario, delta="Alerta" if status_sanitario == "CRÍTICO" else "Ok")

            # MAPA E GRÁFICOS
            st.markdown("---")
            c1, c2 = st.columns([1.6, 1])
            with c1:
                st.subheader("📍 Mapa de Infestação")
                df_geo = df[df['Latitude'] != "N/A"]
                if not df_geo.empty:
                    m = folium.Map(location=[df_geo['Latitude'].mean(), df_geo['Longitude'].mean()], zoom_start=17)
                    for _, r in df_geo.iterrows():
                        cor = 'red' if r['Pragas'] > 15 else 'green'
                        folium.CircleMarker([r['Latitude'], r['Longitude']], radius=10, color=cor, fill=True).add_to(m)
                    st_folium(m, width="100%", height=450)
            
            with c2:
                st.subheader("📈 Inteligência")
                fig_gauge = go.Figure(go.Indicator(mode="gauge+number", value=media_ponto, 
                    gauge={'axis':{'range':[0,50]}, 'bar':{'color':"#1b5e20"}, 'steps':[{'range':[0,15],'color':"#c8e6c9"},{'range':[15,50],'color':"#ffcdd2"}]}))
                fig_gauge.update_layout(height=250, margin=dict(l=10,r=10,t=40,b=10))
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                # REINTEGRADO: Volatilidade Candlestick
                df_v = df.nlargest(5, 'Pragas')
                fig_c = go.Figure(data=[go.Candlestick(x=df_v['Amostra'], open=df_v['Pragas']*0.8, high=df_v['Pragas'], low=df_v['Pragas']*0.6, close=df_v['Pragas']*0.9)])
                fig_c.update_layout(height=200, margin=dict(l=0,r=0,t=0,b=0), xaxis_rangeslider_visible=False)
                st.plotly_chart(fig_c, use_container_width=True)

            # PARECER
            st.markdown("---")
            st.subheader("💡 Recomendação Técnica")
            if status_sanitario == "CRÍTICO":
                st.error(f"🚨 **{nome_tecnico}**, nível crítico no talhão **{talhao_id}**. Média de {media_ponto:.1f} pragas/ponto. Recomenda-se intervenção imediata.")
            else:
                st.success(f"✅ Nível dentro da normalidade para **{tipo_plantio}**. Média de {media_ponto:.1f} pragas/ponto.")

            # EXPORTAR E GALERIA
            csv = df.drop(columns=['_img_obj']).to_csv(index=False, sep=';', encoding='utf-8-sig').encode('utf-8-sig')
            st.download_button("📥 Baixar Relatório CSV", csv, "relatorio.csv", "text/csv", use_container_width=True)

            st.subheader("📸 Focos Detectados (Top 10)")
            for _, row in df.nlargest(10, 'Pragas').iterrows():
                col_img, col_info = st.columns([1.5, 1])
                with col_img: st.image(row['_img_obj'], use_container_width=True)
                with col_info:
                    st.markdown(f"""<div style="background:white; padding:20px; border-radius:15px; border:1px solid #eee;">
                        <h3>🪲 {row['Pragas']} Pragas</h3><p><b>Amostra:</b> {row['Amostra']}</p>
                        <a href="{row['Maps_Link']}" target="_blank"><button class="loc-btn">📍 LOCALIZAR</button></a>
                    </div>""", unsafe_allow_html=True)
                st.markdown("---")
