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

# MEMÓRIA DO SISTEMA
if 'dados_analise' not in st.session_state:
    st.session_state.dados_analise = None

# CSS ORIGINAL (Protegido e com estilo para botão excluir)
st.markdown("""
    <style>
    iframe { width: 100% !important; border-radius: 15px; }
    @keyframes fadeInUp { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
    @keyframes neonPulseRed {
        0% { box-shadow: 0 0 5px #FF0000, 0 0 10px #FF0000; }
        50% { box-shadow: 0 0 20px #FF0000, 0 0 30px #FF0000; }
        100% { box-shadow: 0 0 5px #FF0000, 0 0 10px #FF0000; }
    }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 15px; border-top: 5px solid #2e7d32; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .loc-btn {
        display: inline-block; padding: 12px; font-size: 14px; cursor: pointer; text-align: center; text-decoration: none; 
        color: #fff; background-color: #68CAED; border: 3px solid #FF0000; border-radius: 10px; font-weight: bold; width: 100%;
        animation: neonPulseRed 1.5s infinite ease-in-out; text-transform: uppercase;
    }
    .source-tag { font-size: 10px; background: #e3f2fd; padding: 2px 8px; border-radius: 4px; font-weight: bold; color: #1565c0; margin-bottom: 5px; display: inline-block; }
    .report-section { animation: fadeInUp 0.6s ease-out; }
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

# FUNÇÕES AUXILIARES
def extrair_gps_st(img_file):
    try:
        img = ExifImage(img_file)
        if img.has_exif and hasattr(img, 'gps_latitude'):
            lat = (img.gps_latitude[0] + img.gps_latitude[1]/60 + img.gps_latitude[2]/3600) * (-1 if img.gps_latitude_ref == 'S' else 1)
            lon = (img.gps_longitude[0] + img.gps_longitude[1]/60 + img.gps_longitude[2]/3600) * (-1 if img.gps_longitude_ref == 'W' else 1)
            return lat, lon, "🛰️ DRONE"
    except: pass
    return None, None, "📱 CELULAR"

def link_google_maps(lat, lon):
    return f"https://www.google.com/maps?q={lat},{lon}" if lat != "N/A" and lat is not None else "#"

# 4. MODO DRONE
if modo_operacao == "🛸 Drone Real-Time":
    st.subheader("🎮 Live Stream: Monitoramento Aéreo")
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

# 5. MODO ANALISAR FOTOS
else:
    uploaded_files = st.file_uploader("📂 ADICIONAR NOVAS FOTOS", accept_multiple_files=True, type=['jpg', 'jpeg', 'png'])
    
    if uploaded_files:
        novos_dados = []
        progresso = st.progress(0)
        for i, file in enumerate(uploaded_files):
            try:
                img = Image.open(file)
                results = model.predict(source=img, conf=0.25, verbose=False)
                img_plot = Image.fromarray(results[0].plot()[:, :, ::-1])
                file.seek(0)
                lat, lon, fonte = extrair_gps_st(file)
                novos_dados.append({
                    "id": f"{file.name}_{i}_{datetime.now().timestamp()}",
                    "Amostra": file.name, "Pragas": len(results[0].boxes),
                    "Latitude": lat if lat else "N/A", "Longitude": lon if lon else "N/A",
                    "Fonte": fonte, "Maps_Link": link_google_maps(lat, lon),
                    "Fazenda": nome_fazenda, "Safra": safra, "Talhao": talhao_id,
                    "Cultura": tipo_plantio, "Tecnico": nome_tecnico, "_img_obj": img_plot
                })
                progresso.progress((i + 1) / len(uploaded_files))
            except: continue
        
        # Concatena com o que já existe no cache
        df_novos = pd.DataFrame(novos_dados)
        if st.session_state.dados_analise is None:
            st.session_state.dados_analise = df_novos
        else:
            st.session_state.dados_analise = pd.concat([st.session_state.dados_analise, df_novos], ignore_index=True)

# 6. EXIBIÇÃO DO RELATÓRIO DINÂMICO
if st.session_state.dados_analise is not None and not st.session_state.dados_analise.empty:
    df = st.session_state.dados_analise
    media_ponto = df['Pragas'].mean()
    status_sanitario = "CRÍTICO" if media_ponto > 15 else "NORMAL"

    st.markdown('<div class="report-section">', unsafe_allow_html=True)
    st.markdown(f"### 📊 Relatório Consolidado: {nome_fazenda}")
    
    # KPIs
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Técnico", nome_tecnico)
    k2.metric("Cultura", tipo_plantio)
    k3.metric("Safra", safra)
    k4.metric("Total Pragas", f"{int(df['Pragas'].sum())} un")
    k5.metric("Status", status_sanitario, delta="Alerta" if status_sanitario == "CRÍTICO" else "Ok")

    st.markdown("---")
    
    # MAPA E GRÁFICOS
    col_mapa, col_intel = st.columns([1.6, 1])
    with col_mapa:
        st.subheader("📍 Mapa de Infestação")
        df_geo = df[df['Latitude'] != "N/A"]
        if not df_geo.empty:
            m = folium.Map(location=[df_geo['Latitude'].mean(), df_geo['Longitude'].mean()], zoom_start=18)
            for _, row in df_geo.iterrows():
                cor = 'red' if row['Pragas'] > 15 else 'orange' if row['Pragas'] > 5 else 'green'
                folium.CircleMarker([row['Latitude'], row['Longitude']], radius=10, color=cor, fill=True, popup=f"{row['Pragas']} un").add_to(m)
            st_folium(m, use_container_width=True, height=500)

    with col_intel:
        st.subheader("📈 Pressão e Volatilidade")
        fig_gauge = go.Figure(go.Indicator(mode="gauge+number", value=media_ponto, gauge={'axis': {'range': [0, 50]}, 'bar': {'color': "#1b5e20"}}))
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        df_top = df.nlargest(5, 'Pragas')
        fig_candle = go.Figure(data=[go.Candlestick(x=df_top['Amostra'], open=df_top['Pragas']*0.9, high=df_top['Pragas'], low=df_top['Pragas']*0.7, close=df_top['Pragas']*0.95)])
        fig_candle.update_layout(height=220, xaxis_rangeslider_visible=False, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_candle, use_container_width=True)

    # RECOMENDAÇÃO E DOWNLOAD
    st.markdown("---")
    st.info(f"Diagnóstico: {status_sanitario}. Média de {media_ponto:.1f} pragas por ponto no talhão {talhao_id}.")
    csv = df.drop(columns=['_img_obj', 'id']).to_csv(index=False, sep=';', encoding='utf-8-sig').encode('utf-8-sig')
    st.download_button("📥 Exportar Relatório para Excel", data=csv, file_name=f"Relatorio_{nome_fazenda}.csv", use_container_width=True)

    # GALERIA COM OPÇÃO DE EXCLUIR
    st.markdown("---")
    st.subheader("📸 Gerenciamento de Amostras (Clique em 🗑️ para remover do cache)")
    
    # Criamos uma cópia do dataframe para iterar com segurança
    for index, row in df.iterrows():
        g1, g2 = st.columns([1.5, 1])
        with g1: 
            st.image(row['_img_obj'], use_container_width=True)
        with g2:
            st.markdown(f"""
            <div style="background: white; padding: 15px; border-radius: 12px; border: 1px solid #eee; margin-bottom:10px;">
                <span class="source-tag">{row['Fonte']}</span>
                <h4 style="margin:0;">🪲 {row['Pragas']} Pragas</h4>
                <p style="font-size:12px;"><b>Arquivo:</b> {row['Amostra']}</p>
                <a href="{row['Maps_Link']}" target="_blank"><button class="loc-btn">📍 LOCALIZAR</button></a>
            </div>
            """, unsafe_allow_html=True)
            
            # Botão de Excluir da Memória
            if st.button(f"🗑️ Remover Amostra {index}", key=f"del_{row['id']}"):
                st.session_state.dados_analise = st.session_state.dados_analise.drop(index).reset_index(drop=True)
                st.rerun()
        st.markdown("---")
else:
    st.warning("Aguardando carregamento de fotos para gerar o relatório...")
