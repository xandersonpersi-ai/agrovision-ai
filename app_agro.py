import streamlit as st
import pandas as pd
from ultralytics import YOLO
from exif import Image as ExifImage
import folium
from streamlit_folium import st_folium
import os
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# 1. CONFIGURAÇÃO DE INTERFACE
st.set_page_config(page_title="AgroVision Pro | Intelligence", layout="wide")

# Estilos Premium
st.markdown("""
    <style>
    .main { background-color: #f4f7f6; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 15px; border-top: 5px solid #2e7d32; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)

# 2. CABEÇALHO
st.title("AgroVision Pro AI 🛰️")
st.caption(f"Relatório de Diagnóstico Digital | Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
st.markdown("---")

# 3. FICHA TÉCNICA (SIDEBAR)
st.sidebar.header("📋 Cadastro de Campo")
with st.sidebar.expander("Identificação", expanded=True):
    nome_fazenda = st.text_input("Propriedade", "Fazenda Barretos")
    nome_tecnico = st.text_input("Responsável Técnico", "Anderson Silva")
    tipo_plantio = st.selectbox("Cultura Atual", ["Soja", "Milho", "Algodão", "Cana", "Outros"])
    safra = st.text_input("Ciclo / Safra", "2025/2026")
    talhao_id = st.text_input("Identificação do Talhão", "Área 1")

conf_threshold = st.sidebar.slider("Sensibilidade (Confidence)", 0.01, 1.0, 0.15)

# 4. FUNÇÃO GPS
def extrair_gps_st(img_file):
    try:
        img = ExifImage(img_file)
        if img.has_exif:
            lat = (img.gps_latitude[0] + img.gps_latitude[1]/60 + img.gps_latitude[2]/3600) * (-1 if img.gps_latitude_ref == 'S' else 1)
            lon = (img.gps_longitude[0] + img.gps_longitude[1]/60 + img.gps_longitude[2]/3600) * (-1 if img.gps_longitude_ref == 'W' else 1)
            return lat, lon
    except: return None
    return None

# 5. UPLOAD E PROCESSAMENTO
uploaded_files = st.file_uploader("📂 ARRASTE AS FOTOS PARA ANÁLISE", accept_multiple_files=True, type=['jpg', 'jpeg', 'png'])

if uploaded_files:
    model = YOLO('best.pt' if os.path.exists('best.pt') else 'yolov8n.pt')
    dados_lavoura = []
    progresso = st.progress(0)
    
    for i, file in enumerate(uploaded_files):
        try:
            img = Image.open(file)
            results = model.predict(source=img, conf=conf_threshold)
            img_com_caixas = results[0].plot() 
            img_com_caixas = Image.fromarray(img_com_caixas[:, :, ::-1])
            
            file.seek(0)
            coords = extrair_gps_st(file)
            num_pragas = len(results[0].boxes)
            
            # Inserindo todos os dados de cadastro em cada linha para um relatório rico
            dados_lavoura.append({
                "Amostra": file.name,
                "Pragas": num_pragas,
                "Latitude": coords[0] if coords else "N/A",
                "Longitude": coords[1] if coords else "N/A",
                "Data_Analise": datetime.now().strftime('%d/%m/%Y'),
                "Fazenda": nome_fazenda,
                "Tecnico": nome_tecnico,
                "Cultura": tipo_plantio,
                "Safra": safra,
                "Talhao": talhao_id
            })
            progresso.progress((i + 1) / len(uploaded_files))
            
            # Guardar a imagem processada apenas na memória do app (não vai pro CSV)
            dados_lavoura[-1]["_img_obj"] = img_com_caixas
        except: continue

    if dados_lavoura:
        df = pd.DataFrame(dados_lavoura)
        total_pragas = df['Pragas'].sum()
        media_ponto = df['Pragas'].mean()

        # 6. MÉTRICAS E DASHBOARD
        st.markdown(f"### 📊 Dashboard de Infestação: {nome_fazenda}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Propriedade", nome_fazenda)
        m2.metric("Talhão", talhao_id)
        m3.metric("Total Pragas", int(total_pragas))
        m4.metric("Média/Ponto", f"{media_ponto:.2f}")

        st.markdown("---")

        # 7. MAPA E GRÁFICOS
        col_mapa, col_intel = st.columns([1.6, 1])
        with col_mapa:
            st.subheader("📍 Georreferenciamento de Pragas")
            df_geo = df[df['Latitude'] != "N/A"]
            if not df_geo.empty:
                m = folium.Map(location=[df_geo['Latitude'].mean(), df_geo['Longitude'].mean()], zoom_start=18)
                for _, row in df_geo.iterrows():
                    cor = 'red' if row['Pragas'] > 15 else 'orange' if row['Pragas'] > 5 else 'green'
                    folium.CircleMarker([row['Latitude'], row['Longitude']], radius=10, color=cor, fill=True, popup=f"{row['Pragas']} un").add_to(m)
                st_folium(m, width="100%", height=400)

        with col_intel:
            st.subheader("📈 Análise Técnica")
            # Velocímetro (Gauge)
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number", value = media_ponto,
                gauge = {'axis': {'range': [0, 50]}, 'bar': {'color': "#2e7d32"},
                         'steps': [{'range': [0, 15], 'color': "#c8e6c9"}, {'range': [30, 50], 'color': "#ffcdd2"}]}
            ))
            fig_gauge.update_layout(height=250, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Velas (Candlestick) - Top 10
            df_top10 = df.nlargest(10, 'Pragas')
            fig_candle = go.Figure(data=[go.Candlestick(
                x=df_top10['Amostra'], open=df_top10['Pragas']*0.9, high=df_top10['Pragas'],
                low=df_top10['Pragas']*0.7, close=df_top10['Pragas']*0.95
            )])
            fig_candle.update_layout(height=250, xaxis_rangeslider_visible=False, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig_candle, use_container_width=True)

        # 8. DADOS BRUTOS E DOWNLOAD CORRIGIDO
        st.markdown("---")
        with st.expander("📊 Ver Dados Detalhados e Baixar Relatório", expanded=True):
            # Removemos a coluna da imagem para o Excel não travar
            df_excel = df.drop(columns=['_img_obj'])
            st.dataframe(df_excel, use_container_width=True)
            
            # O SEGREDO: sep=';' para o Excel abrir direto em colunas
            csv = df_excel.to_csv(index=False, sep=';', encoding='utf-8-sig').encode('utf-8-sig')
            
            st.download_button(
                label="📥 Baixar Relatório para Excel (CSV)",
                data=csv,
                file_name=f"Relatorio_{nome_fazenda}_{talhao_id}.csv",
                mime="text/csv"
            )

        # 9. GALERIA DE FOTOS
        st.subheader("📸 Galeria de Focos Detectados (IA)")
        for _, row in df.nlargest(10, 'Pragas').iterrows():
            st.image(row['_img_obj'], caption=f"{row['Amostra']} - {row['Pragas']} pragas detectadas", use_container_width=True)
            st.markdown("---")
