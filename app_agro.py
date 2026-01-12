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
st.set_page_config(page_title="AgroVision Pro | Multi-Source", layout="wide")

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
        text-transform: uppercase; font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return YOLO('best.pt' if os.path.exists('best.pt') else 'yolov8n.pt')

model = load_model()

# 2. SIDEBAR COM CADASTRO
st.sidebar.header("🕹️ Comando Multi-Fonte")
modo_operacao = st.sidebar.radio("Fonte de Dados:", ["📂 Upload Híbrido (Drone/Celular)", "🛸 Drone Real-Time"])

with st.sidebar.expander("📋 Cadastro de Campo", expanded=True):
    nome_fazenda = st.text_input("Propriedade", "Fazenda Santa Fé")
    nome_tecnico = st.text_input("Responsável Técnico", "Anderson Silva")
    tipo_plantio = st.selectbox("Cultura Atual", ["Soja", "Milho", "Algodão", "Cana", "Outros"])
    safra = st.text_input("Ciclo / Safra", "2025/2026")
    talhao_id = st.text_input("Identificação do Talhão", "Talhão 01")

# 4. FUNÇÃO GPS ROBUSTA (DRONE & CELULAR)
def extrair_gps_hibrido(img_file):
    try:
        img = ExifImage(img_file)
        if img.has_exif and hasattr(img, 'gps_latitude'):
            lat = (img.gps_latitude[0] + img.gps_latitude[1]/60 + img.gps_latitude[2]/3600) * (-1 if img.gps_latitude_ref == 'S' else 1)
            lon = (img.gps_longitude[0] + img.gps_longitude[1]/60 + img.gps_longitude[2]/3600) * (-1 if img.gps_longitude_ref == 'W' else 1)
            return lat, lon, "🛰️ Drone/GPS"
    except:
        return None, None, "📱 Manual/Celular"
    return None, None, "📱 Manual/Celular"

# 5. LÓGICA DE PROCESSAMENTO
if modo_operacao == "🛸 Drone Real-Time":
    st.info("Modo de Transmissão Direta Ativo.")
    # (Mantém a lógica de CV2 anterior aqui...)
else:
    uploaded_files = st.file_uploader("📂 SUBIR FOTOS (DRONE OU CELULAR)", accept_multiple_files=True)

    if uploaded_files:
        dados_lavoura = []
        progresso = st.progress(0)
        
        for i, file in enumerate(uploaded_files):
            img = Image.open(file)
            results = model.predict(source=img, conf=0.25, verbose=False)
            img_plot = Image.fromarray(results[0].plot()[:, :, ::-1])
            
            file.seek(0)
            lat, lon, fonte = extrair_gps_hibrido(file)
            
            dados_lavoura.append({
                "Amostra": file.name, "Pragas": len(results[0].boxes),
                "Lat": lat, "Lon": lon, "Fonte": fonte,
                "Cultura": tipo_plantio, "Safra": safra,
                "_img": img_plot
            })
            progresso.progress((i + 1) / len(uploaded_files))

        if dados_lavoura:
            df = pd.DataFrame(dados_lavoura)
            
            # KPIs
            st.markdown(f"### 📊 Relatório Híbrido: {nome_fazenda}")
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Responsável", nome_tecnico)
            k2.metric("Cultura", tipo_plantio)
            k3.metric("Safra", safra)
            k4.metric("Total Pragas", int(df['Pragas'].sum()))
            k5.metric("Média/Ponto", f"{df['Pragas'].mean():.1f}")

            # MAPA COM TRATAMENTO DE ERROS
            st.markdown("---")
            col_map, col_info = st.columns([2, 1])
            
            with col_map:
                st.subheader("📍 Mapa de Calor Georreferenciado")
                df_geo = df.dropna(subset=['Lat', 'Lon'])
                if not df_geo.empty:
                    m = folium.Map(location=[df_geo['Lat'].mean(), df_geo['Lon'].mean()], zoom_start=16)
                    for _, r in df_geo.iterrows():
                        cor = 'red' if r['Pragas'] > 15 else 'green'
                        folium.CircleMarker([r['Lat'], r['Lon']], radius=8, color=cor, fill=True, 
                                          popup=f"{r['Pragas']} un - {r['Fonte']}").add_to(m)
                    st_folium(m, use_container_width=True, height=450)
                else:
                    st.warning("⚠️ Nenhuma foto possui coordenadas GPS (Drone). Exibindo apenas dados tabulares.")

            with col_info:
                st.subheader("💡 Recomendação")
                media = df['Pragas'].mean()
                if media > 15:
                    st.error(f"Nível Crítico! O técnico **{nome_tecnico}** recomenda aplicação imediata na safra **{safra}**.")
                else:
                    st.success("Nível Seguro. Manter monitoramento regular.")

            # GALERIA
            st.markdown("---")
            st.subheader("📸 Detalhes das Amostras")
            for _, row in df.iterrows():
                c1, c2 = st.columns([1, 2])
                with c1: st.image(row['_img'])
                with c2:
                    st.write(f"**Amostra:** {row['Amostra']}")
                    st.write(f"**Fonte:** {row['Fonte']}")
                    st.write(f"**Cultura:** {row['Cultura']} | **Safra:** {row['Safra']}")
                    if row['Fonte'] == "🛰️ Drone/GPS":
                        maps_url = f"https://www.google.com/maps?q={row['Lat']},{row['Lon']}"
                        st.markdown(f'<a href="{maps_url}" target="_blank" class="loc-btn">📍 VER NO MAPA</a>', unsafe_allow_html=True)
                st.markdown("---")
