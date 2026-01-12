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

# --- 1. BANCO DE DADOS ---
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
    c = conn.cursor()
    for d in dados_lista:
        # Garantindo que as chaves existam antes de salvar
        c.execute('''INSERT OR REPLACE INTO analises 
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                  (d.get('id'), d.get('data'), d.get('fazenda'), d.get('tecnico'), 
                   d.get('cultura'), d.get('safra'), d.get('talhao'), d.get('Pragas'), 
                   d.get('lat'), d.get('lon'), d.get('Amostra'), d.get('Fonte')))
    conn.commit()
    conn.close()

init_db()

# --- 2. INTERFACE ---
st.set_page_config(page_title="AgroVision Pro AI", layout="wide")

if 'dados_analise' not in st.session_state:
    st.session_state.dados_analise = None

@st.cache_resource
def load_model():
    return YOLO('best.pt' if os.path.exists('best.pt') else 'yolov8n.pt')

model = load_model()

# --- 3. SIDEBAR ---
st.sidebar.header("🕹️ Cadastro de Campo")
nome_fazenda = st.sidebar.text_input("Propriedade", "Fazenda Santa Fé")
nome_tecnico = st.sidebar.text_input("Responsável Técnico", "Anderson Silva")
tipo_plantio = st.sidebar.selectbox("Cultura", ["Soja", "Milho", "Algodão", "Outros"])
safra_atual = st.sidebar.text_input("Safra", "2025/2026")
talhao_nome = st.sidebar.text_input("Talhão", "Talhão 01")
conf_threshold = st.sidebar.slider("Confiança IA", 0.01, 1.0, 0.25)

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
            return lat, lon, "🛰️ GPS"
    except: pass
    return 0.0, 0.0, "📱 MANUAL"

# --- 5. PROCESSAMENTO ---
st.title("AgroVision Pro AI 🛰️")
files = st.file_uploader("Upload de Fotos", accept_multiple_files=True)

if files:
    novos = []
    for i, f in enumerate(files):
        try:
            with st.spinner(f"Lendo {f.name}..."):
                img = Image.open(f)
                res = model.predict(img, conf=conf_threshold, verbose=False)
                img_plot = Image.fromarray(res[0].plot()[:,:,::-1])
                f.seek(0)
                lat, lon, fonte = extrair_gps(f)
                
                # DICIONÁRIO COMPLETO - Cada chave aqui deve bater com o salvar_no_banco
                item = {
                    "id": f"{f.name}_{i}_{datetime.now().timestamp()}",
                    "data": datetime.now().strftime("%d/%m/%Y"),
                    "fazenda": nome_fazenda,
                    "tecnico": nome_tecnico,
                    "cultura": tipo_plantio,
                    "safra": safra_atual,
                    "talhao": talhao_nome,
                    "Pragas": len(res[0].boxes),
                    "lat": lat,
                    "lon": lon,
                    "Amostra": f.name,
                    "Fonte": fonte,
                    "_img": img_plot
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

# --- 6. EXIBIÇÃO ---
if st.session_state.dados_analise is not None:
    df = st.session_state.dados_analise
    st.divider()
    
    # Dash Rápido
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Pragas", int(df['Pragas'].sum()))
    c2.metric("Média/Ponto", f"{df['Pragas'].mean():.1f}")
    c3.metric("Pontos Coletados", len(df))

    # Galeria
    for idx, row in df.iterrows():
        col1, col2 = st.columns([1.5, 1])
        with col1: st.image(row['_img'], use_container_width=True)
        with col2:
            st.write(f"**{row['Amostra']}**")
            st.write(f"🪲 {row['Pragas']} detectadas")
            st.caption(f"📍 {row['fazenda']} | {row['talhao']}")
    
    # Export
    csv = df.drop(columns=['_img']).to_csv(index=False).encode('utf-8')
    st.download_button("📥 Baixar Planilha", csv, "relatorio.csv", "text/csv", use_container_width=True)
