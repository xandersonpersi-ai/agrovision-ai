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

# --- 1. CONFIGURAÇÃO DE BANCO DE DADOS (MODELO SAAS) ---
def init_db():
    conn = sqlite3.connect('agrovision_saas.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS analises
                 (id TEXT PRIMARY KEY, data TEXT, fazenda TEXT, tecnico TEXT, 
                  cultura TEXT, safra TEXT, talhao TEXT, pragas INTEGER, 
                  lat REAL, lon REAL, arquivo TEXT, fonte TEXT)''')
    conn.commit()
    conn.close()

def salvar_no_banco(dados_lista):
    conn = sqlite3.connect('agrovision_saas.db')
    for d in dados_lista:
        c = conn.cursor()
        c.execute('''INSERT OR REPLACE INTO analises 
                     (id, data, fazenda, tecnico, cultura, safra, talhao, pragas, lat, lon, arquivo, fonte)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                  (d['id'], datetime.now().strftime("%d/%m/%Y %H:%M"), d['fazenda'], d['tecnico'], 
                   d['cultura'], d['safra'], d['talhao'], d['pragas'], 
                   d['lat'], d['lon'], d['Amostra'], d['Fonte']))
    conn.commit()
    conn.close()

init_db()

# --- 2. CONFIGURAÇÃO DE INTERFACE PREMIUM ---
st.set_page_config(page_title="AgroVision Pro | SaaS Intelligence", layout="wide")

if 'dados_analise' not in st.session_state:
    st.session_state.dados_analise = None

st.markdown("""
    <style>
    iframe { width: 100% !important; border-radius: 15px; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 15px; border-top: 5px solid #2e7d32; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .loc-btn {
        display: inline-block; padding: 12px; font-size: 14px; color: #fff !important; 
        background-color: #68CAED; border: 3px solid #FF0000; border-radius: 10px; 
        font-weight: bold; width: 100%; text-align: center; text-decoration: none;
    }
    .source-tag { font-size: 10px; background: #e3f2fd; padding: 2px 8px; border-radius: 4px; font-weight: bold; color: #1565c0; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return YOLO('best.pt' if os.path.exists('best.pt') else 'yolov8n.pt')

model = load_model()

# --- 3. CENTRAL DE COMANDO (SIDEBAR) ---
st.sidebar.header("🕹️ Console SaaS")
with st.sidebar.expander("📋 Cadastro de Campo", expanded=True):
    nome_fazenda = st.text_input("Propriedade", "Fazenda Santa Fé")
    nome_tecnico = st.text_input("Responsável Técnico", "Anderson Silva")
    tipo_plantio = st.selectbox("Cultura Atual", ["Soja", "Milho", "Algodão", "Cana", "Outros"])
    safra_val = st.text_input("Ciclo / Safra", "2025/2026")
    talhao_id = st.text_input("Identificação do Talhão", "Talhão 01")

with st.sidebar.expander("⚙️ Configurações de IA"):
    conf_threshold = st.slider("Sensibilidade (Confiança)", 0.01, 1.0, 0.25)

if st.sidebar.button("🗑️ Limpar Sessão Atual", use_container_width=True):
    st.session_state.dados_analise = None
    st.rerun()

# --- 4. FUNÇÕES AUXILIARES ---
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
    return f"https://www.google.com/maps?q={lat},{lon}" if lat is not None else "#"

# --- 5. PROCESSAMENTO DE IMAGENS ---
st.title("AgroVision Pro AI 🛰️")
st.caption(f"SaaS Mode Ativo | Analisando com {nome_tecnico}")

uploaded_files = st.file_uploader("📂 Entrada de Dados", accept_multiple_files=True, type=['jpg', 'jpeg', 'png'])

if uploaded_files:
    novos_dados = []
    nomes_no_cache = [d['Amostra'] for d in (st.session_state.dados_analise.to_dict('records') if st.session_state.dados_analise is not None else [])]

    for i, file in enumerate(uploaded_files):
        if file.name not in nomes_no_cache:
            try:
                with st.spinner(f"Processando {file.name}..."):
                    img = Image.open(file)
                    results = model.predict(source=img, conf=conf_threshold, verbose=False)
                    img_plot = Image.fromarray(results[0].plot()[:, :, ::-1])
                    file.seek(0)
                    lat, lon, fonte = extrair_gps_st(file)
                    
                    # CORREÇÃO: Inclusão de todos os campos que o banco de dados exige
                    dado_completo = {
                        "id": f"{file.name}_{datetime.now().timestamp()}_{i}",
                        "Amostra": file.name, 
                        "Pragas": len(results[0].boxes),
                        "lat": lat if lat else 0.0, 
                        "lon": lon if lon else 0.0,
                        "Latitude": lat if lat else "N/A", 
                        "Longitude": lon if lon else "N/A",
                        "Fonte": fonte, 
                        "fazenda": nome_fazenda,
                        "tecnico": nome_tecnico,
                        "cultura": tipo_plantio,
                        "safra": safra_val,
                        "talhao": talhao_id,
                        "Maps_Link": link_google_maps(lat, lon),
                        "_img_obj": img_plot
                    }
                    novos_dados.append(dado_completo)
            except Exception as e:
                st.error(f"Erro no arquivo {file.name}: {e}")
    
    if novos_dados:
        salvar_no_banco(novos_dados)
        df_novos = pd.DataFrame(novos_dados)
        if st.session_state.dados_analise is None:
            st.session_state.dados_analise = df_novos
        else:
            st.session_state.dados_analise = pd.concat([st.session_state.dados_analise, df_novos], ignore_index=True)

# --- 6. RELATÓRIO DINÂMICO ---
if st.session_state.dados_analise is not None and not st.session_state.dados_analise.empty:
    df = st.session_state.dados_analise
    media_ponto = df['Pragas'].mean()

    st.markdown(f"### 📊 Dashboard: {nome_fazenda}")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Técnico", nome_tecnico)
    k2.metric("Cultura", tipo_plantio)
    k3.metric("Total Pragas", f"{int(df['Pragas'].sum())}")
    k4.metric("Média/Ponto", f"{media_ponto:.1f}")

    st.markdown("---")
    
    col_mapa, col_gauge = st.columns([2, 1])
    with col_mapa:
        df_geo = df[df['Latitude'] != "N/A"]
        if not df_geo.empty:
            m = folium.Map(location=[df_geo['lat'].mean(), df_geo['lon'].mean()], zoom_start=16)
            for _, row in df_geo.iterrows():
                cor = 'red' if row['Pragas'] > 15 else 'green'
                folium.CircleMarker([row['lat'], row['lon']], radius=10, color=cor, fill=True).add_to(m)
            st_folium(m, use_container_width=True, height=400)

    with col_gauge:
        fig = go.Figure(go.Indicator(mode="gauge+number", value=media_ponto, title={'text': "Infestação"},
                                     gauge={'axis': {'range': [0, 50]}, 'bar': {'color': "darkgreen"}}))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("📸 Amostras")
    for index, row in df.iterrows():
        c1, c2 = st.columns([1, 1])
        with c1: st.image(row['_img_obj'], use_container_width=True)
        with c2:
            st.write(f"**Amostra:** {row['Amostra']}")
            st.write(f"**Pragas:** {row['Pragas']}")
            if row['Latitude'] != "N/A":
                st.markdown(f"[📍 Abrir no Google Maps]({row['Maps_Link']})")
            if st.button(f"Remover {index}", key=f"btn_{row['id']}"):
                st.session_state.dados_analise = st.session_state.dados_analise.drop(index).reset_index(drop=True)
                st.rerun()
        st.markdown("---")
