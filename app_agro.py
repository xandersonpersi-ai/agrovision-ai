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

st.markdown("""
    <style>
    .main { background-color: #f4f7f6; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 15px; border-top: 5px solid #2e7d32; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    div[data-testid="stExpander"] { background-color: #ffffff; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# 2. CABEÇALHO
st.title("AgroVision Pro AI 🛰️")
st.caption(f"Plataforma de Diagnóstico Digital | Sessão: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
st.markdown("---")

# 3. FICHA TÉCNICA (SIDEBAR COMPLETA)
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
                "Latitude": coords[0] if coords else "N/A",
                "Longitude": coords[1] if coords else "N/A",
                "Data": datetime.now().strftime('%d/%m/%Y'),
                "Fazenda": nome_fazenda,
                "Tecnico": nome_tecnico,
                "Cultura": tipo_plantio,
                "Safra": safra,
                "Talhao": talhao_id,
                "_img": img_com_caixas
            })
            progresso.progress((i + 1) / len(uploaded_files))
        except: continue

    if dados_lavoura:
        df = pd.DataFrame(dados_lavoura)
        total_pragas = df['Pragas'].sum()
        media_ponto = df['Pragas'].mean()
        status_sanitario = "CRÍTICO" if media_ponto > 15 else "NORMAL"

        # 6. SUMÁRIO EXECUTIVO (RESTRIÇÃO DE DELTA RECUPERADA)
        st.markdown(f"### 📊 Sumário Executivo: {nome_fazenda}")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Técnico", nome_tecnico)
        k2.metric("Cultura/Safra", f"{tipo_plantio} | {safra}")
        k3.metric("Total Detectado", f"{int(total_pragas)} un")
        k4.metric("Status", status_sanitario, delta="Ação Necessária" if status_sanitario == "CRÍTICO" else "Ok")

        st.markdown("---")

        # 7. MAPA E CENTRO DE INTELIGÊNCIA
        col_mapa, col_intel = st.columns([1.6, 1])
        with col_mapa:
            st.subheader("📍 Georreferenciamento")
            df_geo = df[df['Latitude'] != "N/A"]
            if not df_geo.empty:
                m = folium.Map(location=[df_geo['Latitude'].mean(), df_geo['Longitude'].mean()], zoom_start=18, tiles=None)
                folium.TileLayer('OpenStreetMap', control=False).add_to(m)
                for _, row in df_geo.iterrows():
                    cor = 'red' if row['Pragas'] > 15 else 'orange' if row['Pragas'] > 5 else 'green'
                    folium.CircleMarker([row['Latitude'], row['Longitude']], radius=10, color=cor, fill=True).add_to(m)
                st_folium(m, width="100%", height=500)

        with col_intel:
            st.subheader("📈 Análise Técnica")
            # Gauge Completo
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number", value = media_ponto,
                title = {'text': "Pressão Média"},
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

            # Candlestick Top 10
            st.write("**🕯️ Volatilidade: Top 10 Pontos**")
            df_top10 = df.nlargest(10, 'Pragas')
            fig_candle = go.Figure(data=[go.Candlestick(
                x=df_top10['Amostra'], open=df_top10['Pragas']*0.9, high=df_top10['Pragas'],
                low=df_top10['Pragas']*0.7, close=df_top10['Pragas']*0.95,
                increasing_line_color='#991b1b', decreasing_line_color='#991b1b'
            )])
            fig_candle.update_layout(height=250, xaxis_rangeslider_visible=False, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig_candle, use_container_width=True)

        # 8. RECOMENDAÇÃO TÉCNICA (RECUPERADA E MELHORADA)
        st.markdown("---")
        st.subheader("💡 Parecer Técnico Automático")
        rec_col1, rec_col2 = st.columns([1, 3])
        with rec_col1:
            if status_sanitario == "CRÍTICO": st.error("ALTA INFESTAÇÃO")
            else: st.success("BAIXA INFESTAÇÃO")
        with rec_col2:
            msg = f"Análise concluída para o talhão **{talhao_id}**. "
            if status_sanitario == "CRÍTICO":
                msg += f"A pressão média de {media_ponto:.1f} pragas/ponto exige intervenção imediata para proteger a cultura de {tipo_plantio}."
            else:
                msg += f"Os níveis na fazenda {nome_fazenda} estão dentro do limite de segurança. Manter monitoramento."
            st.write(msg)

        # 9. DADOS BRUTOS E DOWNLOAD (EXCEL-READY COM PONTO E VÍRGULA)
        st.markdown("---")
        with st.expander("📊 Ver Dados Detalhados e Exportar"):
            df_export = df.drop(columns=['_img'])
            st.dataframe(df_export, use_container_width=True)
            # UTF-8-SIG + Ponto e Vírgula = Perfeito para Excel em Português
            csv = df_export.to_csv(index=False, sep=';', encoding='utf-8-sig').encode('utf-8-sig')
            st.download_button("📥 Baixar Relatório para Excel", csv, f"Relatorio_{nome_fazenda}.csv", "text/csv")

        # 10. GALERIA DE FOTOS (FINAL)
        st.subheader("📸 Galeria de Focos Críticos")
        for _, row in df.nlargest(10, 'Pragas').iterrows():
            st.image(row['_img'], caption=f"{row['Amostra']} - {row['Pragas']} pragas", use_container_width=True)
            st.markdown("---")
