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

# 1. CONFIGURAÇÃO DE INTERFACE "PREMIUM"
st.set_page_config(page_title="AgroVision Pro | Intelligence", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f4f7f6; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 15px; border-top: 5px solid #2e7d32; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    div[data-testid="stExpander"] { background-color: #ffffff; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# 2. CABEÇALHO DINÂMICO
st.title("AgroVision Pro AI 🛰️")
st.caption(f"Plataforma de Diagnóstico Digital | Sessão: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
st.markdown("---")

# 3. FICHA TÉCNICA COMPLETA (SIDEBAR RECUPERADA)
st.sidebar.header("📋 Cadastro de Campo")
with st.sidebar.expander("Identificação", expanded=True):
    nome_fazenda = st.text_input("Propriedade", "Fazenda Santa Fé")
    nome_tecnico = st.text_input("Responsável Técnico", "Anderson Silva")
    tipo_plantio = st.selectbox("Cultura Atual", ["Soja", "Milho", "Algodão", "Cana", "Outros"])
    safra = st.text_input("Ciclo / Safra", "2025/2026")
    talhao_id = st.text_input("Identificação do Talhão", "Talhão 01")

with st.sidebar.expander("Configurações de IA"):
    conf_threshold = st.slider("Sensibilidade (Confidence)", 0.01, 1.0, 0.15)

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
uploaded_files = st.file_uploader("📂 ARRASTE AS FOTOS DA VARREDURA", accept_multiple_files=True, type=['jpg', 'jpeg', 'png'])

if uploaded_files:
    model = YOLO('best.pt' if os.path.exists('best.pt') else 'yolov8n.pt')
    dados_lavoura = []
    st.write("### ⚙️ Processando Inteligência Artificial...")
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
            
            dados_lavoura.append({
                "Amostra": file.name, 
                "Pragas": num_pragas,
                "Lat": coords[0] if coords else None, 
                "Lon": coords[1] if coords else None,
                "Imagem_Proc": img_com_caixas
            })
            progresso.progress((i + 1) / len(uploaded_files))
        except: continue

    if dados_lavoura:
        df = pd.DataFrame(dados_lavoura)
        total_encontrado = df['Pragas'].sum()
        media_ponto = df['Pragas'].mean()
        status_sanitario = "CRÍTICO" if media_ponto > 15 else "NORMAL"

        # 6. SUMÁRIO EXECUTIVO (KPIs RECUPERADOS)
        st.markdown(f"### 📊 Sumário Executivo: {nome_fazenda}")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Técnico", nome_tecnico)
        k2.metric("Cultura", tipo_plantio)
        k3.metric("Total Detectado", f"{int(total_encontrado)} un")
        k4.metric("Status", status_sanitario, delta="Alerta" if status_sanitario == "CRÍTICO" else "Ok")

        st.markdown("---")

        # 7. MAPA E CENTRO DE INTELIGÊNCIA
        col_mapa, col_intel = st.columns([1.6, 1])
        
        with col_mapa:
            st.subheader("📍 Georreferenciamento")
            df_geo = df.dropna(subset=['Lat', 'Lon'])
            if not df_geo.empty:
                m = folium.Map(location=[df_geo['Lat'].mean(), df_geo['Lon'].mean()], zoom_start=18)
                for _, row in df_geo.iterrows():
                    cor = 'red' if row['Pragas'] > 15 else 'orange' if row['Pragas'] > 5 else 'green'
                    folium.CircleMarker([row['Lat'], row['Lon']], radius=10, color=cor, fill=True).add_to(m)
                st_folium(m, width="100%", height=500)

        with col_intel:
            st.subheader("📈 Análise Técnica")
            # VOLTOU O VELOCÍMETRO ORIGINAL COM STEPS
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number", value = media_ponto,
                title = {'text': "Média Pragas / Ponto"},
                gauge = {
                    'axis': {'range': [0, 50]},
                    'bar': {'color': "#1b5e20"},
                    'steps': [
                        {'range': [0, 15], 'color': "#c8e6c9"},
                        {'range': [15, 30], 'color': "#fff9c4"},
                        {'range': [30, 50], 'color': "#ffcdd2"}]
                }
            ))
            fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

            # VELAS DOS 10 PONTOS CRÍTICOS (SUBSTITUIU O RANKING)
            st.write("**🕯️ Volatilidade: Top 10 Pontos**")
            df_top10 = df.nlargest(10, 'Pragas')
            fig_candle = go.Figure(data=[go.Candlestick(
                x=df_top10['Amostra'], open=df_top10['Pragas']*0.9, high=df_top10['Pragas'],
                low=df_top10['Pragas']*0.7, close=df_top10['Pragas']*0.95,
                increasing_line_color='#991b1b', decreasing_line_color='#991b1b'
            )])
            fig_candle.update_layout(height=250, xaxis_rangeslider_visible=False, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig_candle, use_container_width=True)

        # 8. RECOMENDAÇÃO TÉCNICA IA (RECUPERADA)
        st.markdown("---")
        st.subheader("💡 Recomendação de Manejo (IA)")
        rec_col1, rec_col2 = st.columns([1, 3])
        with rec_col1:
            if status_sanitario == "CRÍTICO": st.error("ALTA INFESTAÇÃO")
            else: st.success("BAIXA INFESTAÇÃO")
        with rec_col2:
            if status_sanitario == "CRÍTICO":
                st.write(f"**Atenção {nome_tecnico}:** O talhão **{talhao_id}** apresenta focos severos. Recomenda-se a aplicação localizada nos pontos vermelhos indicados no mapa para a cultura de {tipo_plantio}.")
            else:
                st.write(f"Os níveis em **{nome_fazenda}** estão controlados. Continue o monitoramento.")

        # 9. GALERIA DE EVIDÊNCIAS
        st.markdown("---")
        st.subheader("📸 Galeria de Focos Críticos (IA)")
        for _, row in df.nlargest(10, 'Pragas').iterrows():
            st.image(row['Imagem_Proc'], caption=f"{row['Amostra']} - {row['Pragas']} pragas", use_container_width=True)

        # 10. DADOS E DOWNLOAD (RECUPERADO)
        with st.expander("Ver Dados Brutos"):
            st.dataframe(df.drop(columns=['Imagem_Proc']), use_container_width=True)
            csv = df.drop(columns=['Imagem_Proc']).to_csv(index=False).encode('utf-8')
            st.download_button("📥 Exportar Relatório CSV", csv, f"Relatorio_{nome_fazenda}.csv", "text/csv")
