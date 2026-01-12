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

# CSS para correção de bugs visual e animações
st.markdown(f"""
    <style>
    iframe {{ width: 100% !important; border-radius: 15px; }}
    @keyframes fadeInUp {{
        from {{ opacity: 0; transform: translateY(20px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    @keyframes neonPulseRed {{
        0% {{ box-shadow: 0 0 5px #FF0000, 0 0 10px #FF0000; }}
        50% {{ box-shadow: 0 0 20px #FF0000, 0 0 30px #FF0000; }}
        100% {{ box-shadow: 0 0 5px #FF0000, 0 0 10px #FF0000; }}
    }}
    .stMetric {{ 
        background-color: #ffffff; padding: 20px; border-radius: 15px; 
        border-top: 5px solid #2e7d32; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }}
    .loc-btn {{
        display: inline-block; padding: 15px 25px; font-size: 16px; cursor: pointer;
        text-align: center; text-decoration: none; color: #fff; background-color: #68CAED;
        border: 3px solid #FF0000; border-radius: 12px; font-weight: bold; width: 100%;
        animation: neonPulseRed 1.5s infinite ease-in-out; text-transform: uppercase;
        letter-spacing: 1px;
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
st.caption(f"Plataforma de Diagnóstico Digital | Sessão: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
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
    run_drone = st.toggle("🚀 ATIVAR CÂMERA DO DRONE")
    FRAME_WINDOW = st.image([]) 
    
    if run_drone:
        cam_source = int(rtsp_url) if rtsp_url.isdigit() else rtsp_url
        camera = cv2.VideoCapture(cam_source)
        while run_drone:
            ret, frame = camera.read()
            if not ret:
                st.error("Falha ao receber imagem. Verifique a conexão.")
                break
            results = model.predict(frame, conf=conf_threshold, verbose=False)
            ann_frame = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(ann_frame)
        camera.release()
    else:
        st.info("Sistema em Stand-by.")

# 6. MODO ANALISAR FOTOS
else:
    uploaded_files = st.file_uploader("📂 ARRASTE AS FOTOS PARA VARREDURA", accept_multiple_files=True, type=['jpg', 'jpeg', 'png'])

    if uploaded_files:
        dados_lavoura = []
        progresso = st.progress(0)
        
        for i, file in enumerate(uploaded_files):
            try:
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
                    "Tecnico": nome_tecnico,
                    "_img_obj": img_com_caixas
                })
                progresso.progress((i + 1) / len(uploaded_files))
            except: continue

        if dados_lavoura:
            df = pd.DataFrame(dados_lavoura)
            media_ponto = df['Pragas'].mean()
            status_sanitario = "CRÍTICO" if media_ponto > 15 else "NORMAL"

            st.markdown('<div class="report-section">', unsafe_allow_html=True)

            # KPIs COMPLETOS
            st.markdown(f"### 📊 Sumário Executivo: {nome_fazenda}")
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Responsável Técnico", nome_tecnico)
            k2.metric("Cultura Atual", tipo_plantio)
            k3.metric("Ciclo / Safra", safra)
            k4.metric("Total Detectado", f"{int(df['Pragas'].sum())} un")
            k5.metric("Status", status_sanitario, delta="Alerta" if status_sanitario == "CRÍTICO" else "Ok")

            st.markdown("---")
            
            # MAPA (COM CORREÇÃO DE BUG) E ANÁLISE TÉCNICA
            col_mapa, col_intel = st.columns([1.6, 1])
            with col_mapa:
                st.subheader("📍 Georreferenciamento")
                df_geo = df[df['Latitude'] != "N/A"]
                if not df_geo.empty:
                    # Centraliza o mapa e adiciona pontos
                    m = folium.Map(location=[df_geo['Latitude'].mean(), df_geo['Longitude'].mean()], zoom_start=18, control_scale=True)
                    for _, row in df_geo.iterrows():
                        cor = 'red' if row['Pragas'] > 15 else 'orange' if row['Pragas'] > 5 else 'green'
                        folium.CircleMarker(
                            location=[row['Latitude'], row['Longitude']],
                            radius=10,
                            color=cor,
                            fill=True,
                            fill_opacity=0.7,
                            popup=f"{row['Pragas']} pragas detectadas"
                        ).add_to(m)
                    # st_folium com use_container_width resolve o bug do mapa cortado
                    st_folium(m, use_container_width=True, height=500, returned_objects=[])
                else:
                    st.warning("⚠️ Fotos sem metadados de GPS detectadas. O mapa não pode ser exibido.")

            with col_intel:
                st.subheader("📈 Análise de Pressão")
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number", value = media_ponto,
                    gauge = {'axis': {'range': [0, 50]}, 'bar': {'color': "#1b5e20"},
                             'steps': [{'range': [0, 15], 'color': "#c8e6c9"}, {'range': [15, 30], 'color': "#fff9c4"}, {'range': [30, 50], 'color': "#ffcdd2"}]}))
                fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig_gauge, use_container_width=True)

                st.write("**🕯️ Volatilidade de Infestação**")
                df_top5 = df.nlargest(5, 'Pragas')
                fig_candle = go.Figure(data=[go.Candlestick(
                    x=df_top5['Amostra'], open=df_top5['Pragas']*0.9, high=df_top5['Pragas'],
                    low=df_top5['Pragas']*0.7, close=df_top5['Pragas']*0.95)])
                fig_candle.update_layout(height=220, xaxis_rangeslider_visible=False, margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(fig_candle, use_container_width=True)

            # RECOMENDAÇÃO TÉCNICA
            st.markdown("---")
            st.subheader("💡 Recomendação Técnica")
            rec_col1, rec_col2 = st.columns([1, 3])
            with rec_col1:
                if status_sanitario == "CRÍTICO": st.error("🚨 ALTA INFESTAÇÃO")
                else: st.success("✅ BAIXA INFESTAÇÃO")
            with rec_col2:
                texto_laudo = (
                    f"O técnico **{nome_tecnico}** realizou o diagnóstico para a cultura de **{tipo_plantio}** na safra **{safra}**. "
                    f"Análise no talhão **{talhao_id}** indica média de **{media_ponto:.1f}** pragas por ponto. "
                )
                if status_sanitario == "CRÍTICO":
                    texto_laudo += "⚠️ **Ação:** Níveis acima do limite econômico. Intervenção recomendada."
                else:
                    texto_laudo += "👍 **Ação:** Níveis sob controle. Manter monitoramento."
                st.info(texto_laudo)

            # EXPORTAÇÃO
            st.markdown("---")
            df_export = df.drop(columns=['_img_obj'])
            csv = df_export.to_csv(index=False, sep=';', encoding='utf-8-sig').encode('utf-8-sig')
            st.download_button(label="📥 Baixar CSV para Excel", data=csv, file_name=f"Relatorio_{nome_fazenda}.csv", use_container_width=True)

            # GALERIA (AJUSTADO: APENAS CULTURA E SAFRA)
            st.subheader("📸 Detalhes dos Focos (GPS)")
            for _, row in df.nlargest(10, 'Pragas').iterrows():
                g1, g2 = st.columns([1.5, 1])
                with g1: st.image(row['_img_obj'], use_container_width=True)
                with g2:
                    st.markdown(f"""
                    <div style="background: white; padding: 20px; border-radius: 15px; border: 1px solid #eee; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
                        <h3 style="margin-top:0;">🪲 {row['Pragas']} Detectadas</h3>
                        <p><b>Amostra:</b> {row['Amostra']}</p>
                        <p><b>Cultura:</b> {row['Cultura']} | <b>Safra:</b> {row['Safra']}</p>
                        <hr>
                        <a href="{row['Maps_Link']}" target="_blank"><button class="loc-btn">📍 LOCALIZAR AGORA</button></a>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown("---")
            st.markdown('</div>', unsafe_allow_html=True)
