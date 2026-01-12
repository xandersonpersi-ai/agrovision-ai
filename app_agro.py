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

if 'dados_analise' not in st.session_state:
    st.session_state.dados_analise = None

# CSS PREMIUM
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
        color: #fff !important; background-color: #68CAED; border: 3px solid #FF0000; border-radius: 10px; font-weight: bold; width: 100%;
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
with st.sidebar.expander("📋 Cadastro de Campo", expanded=True):
    nome_fazenda = st.text_input("Propriedade", "Fazenda Santa Fé")
    nome_tecnico = st.text_input("Responsável Técnico", "Anderson Silva")
    tipo_plantio = st.selectbox("Cultura Atual", ["Soja", "Milho", "Algodão", "Cana", "Outros"])
    safra = st.text_input("Ciclo / Safra", "2025/2026")
    talhao_id = st.text_input("Identificação do Talhão", "Talhão 01")

with st.sidebar.expander("⚙️ Configurações de IA", expanded=True):
    conf_threshold = st.slider("Sensibilidade (Confiança)", 0.01, 1.0, 0.25)

if st.sidebar.button("🗑️ Limpar Todos os Dados", use_container_width=True):
    st.session_state.dados_analise = None
    st.rerun()

# 4. FUNÇÕES AUXILIARES
def extrair_gps_st(img_file):
    try:
        img = ExifImage(img_file)
        if img.has_exif and hasattr(img, 'gps_latitude'):
            lat = (img.gps_latitude[0] + img.gps_latitude[1]/60 + img.gps_latitude[2]/3600) * (-1 if img.gps_latitude_ref == 'S' else 1)
            lon = (img.gps_longitude[0] + img.gps_longitude[1]/60 + img.gps_longitude[2]/3600) * (-1 if img.gps_longitude_ref == 'W' else 1)
            return lat, lon, "🛰️ GPS ATIVO"
    except: pass
    return None, None, "📱 MANUAL/CELULAR"

def link_google_maps(lat, lon):
    return f"https://www.google.com/maps?q={lat},{lon}" if lat is not None and lat != "N/A" else "#"

# 5. PROCESSAMENTO DE IMAGENS
st.subheader("📂 Varredura de Imagens")
uploaded_files = st.file_uploader("Arraste as fotos para análise técnica", accept_multiple_files=True, type=['jpg', 'jpeg', 'png'])

if uploaded_files:
    novos_dados = []
    nomes_no_cache = []
    if st.session_state.dados_analise is not None:
        nomes_no_cache = st.session_state.dados_analise['Amostra'].tolist()

    for i, file in enumerate(uploaded_files):
        if file.name not in nomes_no_cache:
            try:
                with st.spinner(f"Analisando {file.name}..."):
                    img = Image.open(file)
                    results = model.predict(source=img, conf=conf_threshold, verbose=False)
                    img_plot = Image.fromarray(results[0].plot()[:, :, ::-1])
                    file.seek(0)
                    lat, lon, fonte = extrair_gps_st(file)
                    novos_dados.append({
                        "id": f"{file.name}_{datetime.now().timestamp()}_{i}",
                        "Amostra": file.name, "Pragas": len(results[0].boxes),
                        "Latitude": lat if lat else "N/A", "Longitude": lon if lon else "N/A",
                        "Fonte": fonte, "Maps_Link": link_google_maps(lat, lon),
                        "Fazenda": nome_fazenda, "Safra": safra, "Talhao": talhao_id,
                        "Cultura": tipo_plantio, "Tecnico": nome_tecnico, "_img_obj": img_plot
                    })
            except: continue
    
    if novos_dados:
        df_novos = pd.DataFrame(novos_dados)
        if st.session_state.dados_analise is None:
            st.session_state.dados_analise = df_novos
        else:
            st.session_state.dados_analise = pd.concat([st.session_state.dados_analise, df_novos], ignore_index=True)

# 6. RELATÓRIO DINÂMICO
if st.session_state.dados_analise is not None and not st.session_state.dados_analise.empty:
    df = st.session_state.dados_analise
    media_ponto = df['Pragas'].mean()
    status_sanitario = "CRÍTICO" if media_ponto > 15 else "NORMAL"

    st.markdown('<div class="report-section">', unsafe_allow_html=True)
    st.markdown(f"### 📊 BI - Dashboard Agrícola: {nome_fazenda}")
    
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Técnico", nome_tecnico)
    k2.metric("Cultura", tipo_plantio)
    k3.metric("Safra", safra)
    k4.metric("Total Pragas", f"{int(df['Pragas'].sum())} un")
    k5.metric("Status", status_sanitario, delta="Alerta" if status_sanitario == "CRÍTICO" else "Ok")

    st.markdown("---")
    
    col_mapa, col_intel = st.columns([1.6, 1])
    with col_mapa:
        st.subheader("📍 Georreferenciamento de Pragas")
        df_geo = df[df['Latitude'] != "N/A"]
        if not df_geo.empty:
            m = folium.Map(location=[df_geo['Latitude'].astype(float).mean(), df_geo['Longitude'].astype(float).mean()], zoom_start=17)
            for _, row in df_geo.iterrows():
                cor = 'red' if row['Pragas'] > 15 else 'orange' if row['Pragas'] > 5 else 'green'
                folium.CircleMarker([row['Latitude'], row['Longitude']], radius=12, color=cor, fill=True, popup=f"{row['Amostra']}").add_to(m)
            st_folium(m, use_container_width=True, height=550, key="mapa_final")

    with col_intel:
        st.subheader("📈 Inteligência de Dados")
        
        # 1. Gauge (Gráfico de Pressão)
        fig_gauge = go.Figure(go.Indicator(mode="gauge+number", value=media_ponto, 
            title={'text': "Média Pragas/Ponto"},
            gauge={'axis': {'range': [0, 50]}, 'bar': {'color': "#1b5e20"},
                   'steps': [{'range': [0, 15], 'color': "#c8e6c9"}, {'range': [15, 50], 'color': "#ffcdd2"}]}))
        fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=0))
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # 2. Candlestick (Gráfico de Velas de Volatilidade)
        st.write("**🕯️ Volatilidade da Infestação**")
        df_top = df.nlargest(5, 'Pragas')
        fig_candle = go.Figure(data=[go.Candlestick(x=df_top['Amostra'], 
                                open=df_top['Pragas']*0.9, high=df_top['Pragas'], 
                                low=df_top['Pragas']*0.7, close=df_top['Pragas']*0.95)])
        fig_candle.update_layout(height=220, xaxis_rangeslider_visible=False, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_candle, use_container_width=True)

        # 3. Bar Chart (Gráfico de Barras de Top Focos)
        st.write("**📊 Top 5 Focos Críticos**")
        fig_bar = go.Figure(go.Bar(x=df_top['Pragas'], y=df_top['Amostra'], orientation='h', marker_color='#2e7d32'))
        fig_bar.update_layout(height=180, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_bar, use_container_width=True)

    # LAUDO TÉCNICO E EXPORTAÇÃO
    st.markdown("---")
    st.subheader("💡 Laudo e Recomendação")
    rec_col1, rec_col2 = st.columns([1, 3])
    with rec_col1:
        if status_sanitario == "CRÍTICO": st.error("🚨 INFESTAÇÃO ALTA")
        else: st.success("✅ SITUAÇÃO SOB CONTROLE")
    with rec_col2:
        st.info(f"O talhão **{talhao_id}** apresenta média de **{media_ponto:.1f}** pragas por ponto. {'Intervenção química recomendada.' if status_sanitario == 'CRÍTICO' else 'Seguir com monitoramento padrão.'}")

    csv = df.drop(columns=['_img_obj', 'id']).to_csv(index=False, sep=';', encoding='utf-8-sig').encode('utf-8-sig')
    st.download_button("📥 Baixar Relatório Full (Excel)", data=csv, file_name=f"Relatorio_{nome_fazenda}.csv", use_container_width=True)

    # GALERIA
    st.subheader("📸 Detalhes das Amostras")
    for index, row in df.iterrows():
        g1, g2 = st.columns([1.5, 1])
        with g1: st.image(row['_img_obj'], use_container_width=True)
        with g2:
            st.markdown(f"""
            <div style="background: white; padding: 15px; border-radius: 12px; border: 1px solid #eee; margin-bottom:10px;">
                <span class="source-tag">{row['Fonte']}</span>
                <h4 style="margin:0;">🪲 {row['Pragas']} Detectadas</h4>
                <p style="font-size:12px;"><b>Arquivo:</b> {row['Amostra']}</p>
                <hr>
                {"<a href='"+row['Maps_Link']+"' target='_blank'><button class='loc-btn'>📍 GOOGLE MAPS</button></a>" if row['Latitude'] != "N/A" else "<i>Sem GPS</i>"}
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"🗑️ Remover {index}", key=f"del_{row['id']}"):
                st.session_state.dados_analise = st.session_state.dados_analise.drop(index).reset_index(drop=True)
                st.rerun()
        st.markdown("---")
