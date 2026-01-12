import streamlit as st
import pandas as pd
from ultralytics import YOLO
from exif import Image as ExifImage
import folium
from streamlit_folium import st_folium
import os
import sqlite3
from PIL import Image
import plotly.graph_objects as go
from datetime import datetime
import cv2  
import numpy as np

# --- 1. BANCO DE DADOS (SaaS Foundation) ---
def init_db():
    conn = sqlite3.connect('agrovision_saas.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS analises
                 (id TEXT PRIMARY KEY, data TEXT, fazenda TEXT, tecnico TEXT, 
                  cultura TEXT, safra TEXT, talhao TEXT, pragas INTEGER, 
                  latitude REAL, longitude REAL, arquivo TEXT, fonte TEXT)''')
    conn.commit()
    conn.close()

def salvar_no_banco(dados_lista):
    conn = sqlite3.connect('agrovision_saas.db')
    c = conn.cursor()
    for d in dados_lista:
        c.execute('''INSERT OR REPLACE INTO analises 
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                  (d['id'], d['data'], d['fazenda'], d['tecnico'], 
                   d['cultura'], d['safra'], d['talhao'], d['Pragas'], 
                   d['Latitude'], d['Longitude'], d['Amostra'], d['Fonte']))
    conn.commit()
    conn.close()

init_db()

# --- 2. INTERFACE PREMIUM ---
st.set_page_config(page_title="AgroVision Pro | SaaS Intelligence", layout="wide")

if 'dados_analise' not in st.session_state:
    st.session_state.dados_analise = None

st.markdown("""
    <style>
    iframe { width: 100% !important; border-radius: 15px; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 15px; border-top: 5px solid #2e7d32; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .loc-btn { display: inline-block; padding: 12px; font-size: 14px; color: #fff !important; background-color: #68CAED; border: 3px solid #FF0000; border-radius: 10px; font-weight: bold; width: 100%; text-align: center; text-decoration: none; }
    .source-tag { font-size: 10px; background: #e3f2fd; padding: 2px 8px; border-radius: 4px; font-weight: bold; color: #1565c0; margin-bottom: 5px; display: inline-block; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return YOLO('best.pt' if os.path.exists('best.pt') else 'yolov8n.pt')

model = load_model()

# --- 3. SIDEBAR ---
st.sidebar.header("🕹️ Central de Comando")
with st.sidebar.expander("📋 Cadastro de Campo", expanded=True):
    propriedade = st.sidebar.text_input("Propriedade", "Fazenda Santa Fé")
    tecnico = st.sidebar.text_input("Responsável Técnico", "Anderson Silva")
    cultura = st.sidebar.selectbox("Cultura Atual", ["Soja", "Milho", "Algodão", "Cana", "Outros"])
    safra = st.sidebar.text_input("Ciclo / Safra", "2025/2026")
    talhao = st.sidebar.text_input("Identificação do Talhão", "Talhão 01")

conf_threshold = st.sidebar.slider("Sensibilidade IA", 0.01, 1.0, 0.25)

if st.sidebar.button("🗑️ Limpar Sessão"):
    st.session_state.dados_analise = None
    st.rerun()

# --- 4. FUNÇÕES ---
def extrair_gps(img_file):
    try:
        img = ExifImage(img_file)
        if img.has_exif and hasattr(img, 'gps_latitude'):
            lat = (img.gps_latitude[0] + img.gps_latitude[1]/60 + img.gps_latitude[2]/3600) * (-1 if img.gps_latitude_ref == 'S' else 1)
            lon = (img.gps_longitude[0] + img.gps_longitude[1]/60 + img.gps_longitude[2]/3600) * (-1 if img.gps_longitude_ref == 'W' else 1)
            return lat, lon, "🛰️ GPS ATIVO"
    except: pass
    return 0.0, 0.0, "📱 MANUAL"

# --- 5. PROCESSAMENTO ---
st.title("AgroVision Pro AI 🛰️")
uploaded_files = st.file_uploader("📂 Entrada de Dados", accept_multiple_files=True, type=['jpg', 'jpeg', 'png'])

if uploaded_files:
    novos = []
    nomes_existentes = []
    if st.session_state.dados_analise is not None:
        nomes_existentes = st.session_state.dados_analise['Amostra'].tolist()

    for i, file in enumerate(uploaded_files):
        if file.name not in nomes_existentes:
            try:
                with st.spinner(f"Analisando {file.name}..."):
                    img = Image.open(file)
                    results = model.predict(img, conf=conf_threshold, verbose=False)
                    img_plot = Image.fromarray(results[0].plot()[:, :, ::-1])
                    file.seek(0)
                    lat, lon, fonte = extrair_gps(file)
                    
                    item = {
                        "id": f"{file.name}_{i}_{datetime.now().timestamp()}",
                        "data": datetime.now().strftime("%d/%m/%Y %H:%M"),
                        "fazenda": propriedade,
                        "tecnico": tecnico,
                        "cultura": cultura,
                        "safra": safra,
                        "talhao": talhao,
                        "Pragas": len(results[0].boxes),
                        "Latitude": lat,
                        "Longitude": lon,
                        "Amostra": file.name,
                        "Fonte": fonte,
                        "Maps_Link": f"https://www.google.com/maps?q={lat},{lon}" if lat != 0.0 else "#",
                        "_img_obj": img_plot
                    }
                    novos.append(item)
            except: continue
    
    if novos:
        salvar_no_banco(novos)
        df_novos = pd.DataFrame(novos)
        if st.session_state.dados_analise is None:
            st.session_state.dados_analise = df_novos
        else:
            st.session_state.dados_analise = pd.concat([st.session_state.dados_analise, df_novos], ignore_index=True)

# --- 6. RELATÓRIO DINÂMICO ---
if st.session_state.dados_analise is not None and not st.session_state.dados_analise.empty:
    df = st.session_state.dados_analise
    media = df['Pragas'].mean()

    st.markdown(f"### 📊 BI - Dashboard: {propriedade}")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Técnico", tecnico)
    k2.metric("Safra", safra)
    k3.metric("Total Pragas", f"{int(df['Pragas'].sum())} un")
    k4.metric("Status", "CRÍTICO" if media > 15 else "NORMAL")

    st.divider()
    
    c_mapa, c_gauge = st.columns([2, 1])
    with c_mapa:
        st.subheader("📍 Georreferenciamento")
        df_geo = df[df['Latitude'] != 0.0]
        if not df_geo.empty:
            m = folium.Map(location=[df_geo['Latitude'].mean(), df_geo['Longitude'].mean()], zoom_start=17)
            for _, row in df_geo.iterrows():
                cor = 'red' if row['Pragas'] > 15 else 'green'
                folium.CircleMarker([row['Latitude'], row['Longitude']], radius=12, color=cor, fill=True).add_to(m)
            st_folium(m, use_container_width=True, height=400)
        else:
            st.warning("Nenhuma coordenada GPS encontrada nas imagens.")

    with c_gauge:
        st.subheader("📈 Infestação")
        fig = go.Figure(go.Indicator(mode="gauge+number", value=media, gauge={'axis': {'range': [0, 50]}, 'bar': {'color': "#2e7d32"}}))
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("📸 Amostras")
    for idx, row in df.iterrows():
        g1, g2 = st.columns([1.5, 1])
        with g1: st.image(row['_img_obj'], use_container_width=True)
        with g2:
            st.markdown(f"""
            <div style="background: white; padding: 15px; border-radius: 12px; border: 1px solid #eee;">
                <span class="source-tag">{row['Fonte']}</span>
                <h4>🪲 {row['Pragas']} Detectadas</h4>
                <p><b>Talhão:</b> {row['talhao']}</p>
                {"<a href='"+row['Maps_Link']+"' target='_blank'><button class='loc-btn'>📍 VER NO MAPA</button></a>" if row['Latitude'] != 0.0 else "<i>Sem GPS</i>"}
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"Remover {idx}", key=f"del_{row['id']}"):
                st.session_state.dados_analise = st.session_state.dados_analise.drop(idx).reset_index(drop=True)
                st.rerun()
else:
    st.info("Aguardando upload para iniciar o diagnóstico.")
