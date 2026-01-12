import streamlit as st
import pandas as pd
from ultralytics import YOLO
from exif import Image as ExifImage
import folium
from streamlit_folium import st_folium
import os
import mysql.connector  # Alterado para MySQL
from PIL import Image
import plotly.graph_objects as go
from datetime import datetime
import cv2  
import numpy as np

# --- 1. BANCO DE DADOS (MySQL SaaS) ---
def get_mysql_connection():
    # Substitua pelos dados do seu servidor MySQL
    return mysql.connector.connect(
        host="seu_host",
        user="seu_usuario",
        password="sua_senha",
        database="agrovision_db"
    )

def init_db():
    try:
        conn = get_mysql_connection()
        c = conn.cursor()
        # Sintaxe ajustada para MySQL (VARCHAR e DATETIME)
        c.execute('''CREATE TABLE IF NOT EXISTS analises
                     (id VARCHAR(255) PRIMARY KEY, 
                      data_hora DATETIME, 
                      fazenda VARCHAR(100), 
                      tecnico VARCHAR(100), 
                      cultura VARCHAR(50), 
                      safra VARCHAR(20), 
                      talhao VARCHAR(50), 
                      pragas INT, 
                      latitude FLOAT, 
                      longitude FLOAT, 
                      arquivo VARCHAR(255), 
                      fonte VARCHAR(50))''')
        conn.commit()
        c.close()
        conn.close()
    except Exception as e:
        st.error(f"Erro ao conectar no MySQL: {e}")

def salvar_no_banco(dados_lista):
    try:
        conn = get_mysql_connection()
        c = conn.cursor()
        sql = """INSERT INTO analises 
                 (id, data_hora, fazenda, tecnico, cultura, safra, talhao, pragas, latitude, longitude, arquivo, fonte) 
                 VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) 
                 ON DUPLICATE KEY UPDATE pragas=VALUES(pragas)"""
        
        for d in dados_lista:
            # Converte a string de data para objeto datetime do MySQL
            data_dt = datetime.strptime(d['data'], "%d/%m/%Y %H:%M")
            valores = (d['id'], data_dt, d['fazenda'], d['tecnico'], 
                       d['cultura'], d['safra'], d['talhao'], d['Pragas'], 
                       d['Latitude'], d['Longitude'], d['Amostra'], d['Fonte'])
            c.execute(sql, valores)
            
        conn.commit()
        c.close()
        conn.close()
    except Exception as e:
        st.error(f"Erro ao salvar no MySQL: {e}")

# Inicializa o banco de dados MySQL
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
    .report-section { background: #f8f9fa; padding: 20px; border-radius: 15px; margin-top: 20px; border: 1px solid #eee; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return YOLO('best.pt' if os.path.exists('best.pt') else 'yolov8n.pt')

model = load_model()

# --- 3. SIDEBAR (CENTRAL DE COMANDO) ---
st.sidebar.header("🕹️ Central de Comando")
with st.sidebar.expander("📋 Cadastro de Campo", expanded=True):
    propriedade = st.sidebar.text_input("Propriedade", "Fazenda Santa Fé")
    tecnico = st.sidebar.text_input("Responsável Técnico", "Anderson Silva")
    cultura_selecionada = st.sidebar.selectbox("Cultura Atual", ["Soja", "Milho", "Algodão", "Cana", "Outros"])
    safra_input = st.sidebar.text_input("Ciclo / Safra", "2025/2026")
    talhao_input = st.sidebar.text_input("Identificação do Talhão", "Talhão 01")

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
st.caption(f"Plataforma de Diagnóstico Digital | SaaS Ativo (MySQL)")
st.markdown("---")

uploaded_files = st.file_uploader("📂 Entrada de Dados", accept_multiple_files=True, type=['jpg', 'jpeg', 'png'])

if uploaded_files:
    novos = []
    nomes_existentes = st.session_state.dados_analise['Amostra'].tolist() if st.session_state.dados_analise is not None else []

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
                        "cultura": cultura_selecionada,
                        "safra": safra_input,
                        "talhao": talhao_input,
                        "Pragas": len(results[0].boxes),
                        "Latitude": lat,
                        "Longitude": lon,
                        "Amostra": file.name,
                        "Fonte": fonte,
                        "Maps_Link": f"https://www.google.com/maps?q={lat},{lon}" if lat != 0.0 else "#",
                        "_img_obj": img_plot
                    }
                    novos.append(item)
            except Exception as e:
                st.error(f"Erro no processamento {file.name}: {e}")
    
    if novos:
        salvar_no_banco(novos)
        df_novos = pd.DataFrame(novos)
        if st.session_state.dados_analise is None:
            st.session_state.dados_analise = df_novos
        else:
            st.session_state.dados_analise = pd.concat([st.session_state.dados_analise, df_novos], ignore_index=True)

# --- 6. RELATÓRIO DINÂMICO E BI ---
if st.session_state.dados_analise is not None and not st.session_state.dados_analise.empty:
    df = st.session_state.dados_analise
    media = df['Pragas'].mean()
    status_sanitario = "CRÍTICO" if media > 15 else "NORMAL"

    st.markdown(f"### 📊 BI - Dashboard Agrícola: {propriedade}")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Técnico", tecnico)
    k2.metric("Cultura", cultura_selecionada)
    k3.metric("Safra", safra_input)
    k4.metric("Total Pragas", f"{int(df['Pragas'].sum())} un")
    k5.metric("Status", status_sanitario, delta="ALERTA" if status_sanitario == "CRÍTICO" else "OK")

    st.markdown("---")
    
    c_mapa, c_intel = st.columns([1.6, 1])
    with c_mapa:
        st.subheader("📍 Georreferenciamento")
        df_geo = df[df['Latitude'] != 0.0]
        if not df_geo.empty:
            m = folium.Map(location=[df_geo['Latitude'].mean(), df_geo['Longitude'].mean()], zoom_start=17)
            for _, row in df_geo.iterrows():
                cor = 'red' if row['Pragas'] > 15 else 'orange' if row['Pragas'] > 5 else 'green'
                folium.CircleMarker([row['Latitude'], row['Longitude']], radius=12, color=cor, fill=True, popup=row['Amostra']).add_to(m)
            st_folium(m, use_container_width=True, height=480)
        else: st.warning("Sem dados de GPS para exibir no mapa.")

    with c_intel:
        st.subheader("📈 Inteligência")
        fig_gauge = go.Figure(go.Indicator(mode="gauge+number", value=media, 
            title={'text': "Média Pragas/Ponto"},
            gauge={'axis': {'range': [0, 50]}, 'bar': {'color': "#1b5e20"},
                   'steps': [{'range': [0, 15], 'color': "#c8e6c9"}, {'range': [15, 50], 'color': "#ffcdd2"}]}))
        fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=0))
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        df_top = df.nlargest(5, 'Pragas')
        fig_candle = go.Figure(data=[go.Candlestick(x=df_top['Amostra'], 
                                open=df_top['Pragas']*0.9, high=df_top['Pragas'], 
                                low=df_top['Pragas']*0.7, close=df_top['Pragas']*0.95)])
        fig_candle.update_layout(height=220, xaxis_rangeslider_visible=False, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_candle, use_container_width=True)

    st.markdown('<div class="report-section">', unsafe_allow_html=True)
    st.subheader("💡 Laudo e Recomendação")
    l1, l2 = st.columns([1, 3])
    with l1:
        if status_sanitario == "CRÍTICO": st.error("🚨 INFESTAÇÃO ALTA")
        else: st.success("✅ SITUAÇÃO SOB CONTROLE")
    with l2:
        st.info(f"O talhão **{talhao_input}** da propriedade **{propriedade}** apresenta média de **{media:.1f}** pragas/ponto. Diagnóstico realizado por **{tecnico}**.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    csv = df.drop(columns=['_img_obj', 'id']).to_csv(index=False, sep=';', encoding='utf-8-sig').encode('utf-8-sig')
    st.download_button("📥 Baixar Relatório Técnico (CSV/Excel)", data=csv, file_name=f"Relatorio_{propriedade}_{talhao_input}.csv", use_container_width=True)

    st.subheader("📸 Detalhes das Amostras")
    for idx, row in df.iterrows():
        g1, g2 = st.columns([1.5, 1])
        with g1: st.image(row['_img_obj'], use_container_width=True)
        with g2:
            st.markdown(f"""
            <div style="background: white; padding: 15px; border-radius: 12px; border: 1px solid #eee; margin-bottom:10px;">
                <span class="source-tag">{row['Fonte']}</span>
                <h4 style="margin:0;">🪲 {row['Pragas']} Detectadas</h4>
                <p style="font-size:12px;"><b>Arquivo:</b> {row['Amostra']}</p>
                <hr>
                {"<a href='"+row['Maps_Link']+"' target='_blank'><button class='loc-btn'>📍 GOOGLE MAPS</button></a>" if row['Latitude'] != 0.0 else "<i>Sem GPS</i>"}
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"🗑️ Remover {idx}", key=f"del_{row['id']}"):
                st.session_state.dados_analise = st.session_state.dados_analise.drop(idx).reset_index(drop=True)
                st.rerun()
else:
    st.info("Aguardando upload de fotos para gerar o diagnóstico e inteligência de dados.")
