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

# 1. CONFIGURAÇÃO DE INTERFACE "PREMIUM" COM NEON AZUL #68CAED
st.set_page_config(page_title="AgroVision Pro | Intelligence", layout="wide")

st.markdown(f"""
    <style>
    @keyframes fadeInUp {{
        from {{ opacity: 0; transform: translateY(20px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    
    /* ANIMAÇÃO DO PULSO NEON EM AZUL #68CAED */
    @keyframes neonPulse {{
        0% {{ box-shadow: 0 0 5px #68CAED, 0 0 10px #68CAED; }}
        50% {{ box-shadow: 0 0 20px #68CAED, 0 0 30px #68CAED; }}
        100% {{ box-shadow: 0 0 5px #68CAED, 0 0 10px #68CAED; }}
    }}

    .main {{ background-color: #f4f7f6; }}
    
    .stMetric {{ 
        background-color: #ffffff; 
        padding: 20px; 
        border-radius: 15px; 
        border-top: 5px solid #2e7d32; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }}
    .stMetric:hover {{ transform: translateY(-5px); }}

    /* O BOTÃO COM SUA COR PERSONALIZADA #68CAED */
    .loc-btn {{
        display: inline-block;
        padding: 15px 25px;
        font-size: 16px;
        cursor: pointer;
        text-align: center;
        text-decoration: none;
        color: #fff;
        background: #68CAED; /* Sua cor aqui */
        border: none;
        border-radius: 12px;
        font-weight: bold;
        width: 100%;
        transition: all 0.3s ease;
        animation: neonPulse 2s infinite ease-in-out;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}
    
    .loc-btn:hover {{
        background: #4ab8db; /* Um tom levemente mais escuro para o hover */
        transform: scale(1.03);
        box-shadow: 0 0 40px #68CAED;
        color: #fff;
    }}

    .report-section {{ animation: fadeInUp 0.6s ease-out; }}
    </style>
    """, unsafe_allow_html=True)

# 2. CABEÇALHO
st.title("AgroVision Pro AI 🛰️")
st.caption(f"Plataforma de Diagnóstico Digital | Sessão: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
st.markdown("---")

# 3. FICHA TÉCNICA (SIDEBAR)
st.sidebar.header("📋 Cadastro de Campo")
with st.sidebar.expander("Identificação", expanded=True):
    nome_fazenda = st.text_input("Propriedade", "Fazenda Santa Fé")
    nome_tecnico = st.text_input("Responsável Técnico", "Anderson Silva")
    tipo_plantio = st.selectbox("Cultura Atual", ["Soja", "Milho", "Algodão", "Cana", "Outros"])
    safra = st.text_input("Ciclo / Safra", "2025/2026")
    talhao_id = st.text_input("Identificação do Talhão", "Talhão 01")

with st.sidebar.expander("Configurações de IA"):
    conf_threshold = st.slider("Sensibilidade (Confidence)", 0.01, 1.0, 0.15)

# 4. FUNÇÕES GPS
def extrair_gps_st(img_file):
    try:
        img = ExifImage(img_file)
        if img.has_exif:
            lat = (img.gps_latitude[0] + img.gps_latitude[1]/60 + img.gps_latitude[2]/3600) * (-1 if img.gps_latitude_ref == 'S' else 1)
            lon = (img.gps_longitude[0] + img.gps_longitude[1]/60 + img.gps_longitude[2]/3600) * (-1 if img.gps_longitude_ref == 'W' else 1)
            return lat, lon
    except: return None
    return None

def link_google_maps(lat, lon):
    if lat != "N/A":
        return f"https://www.google.com/maps/search/?api=1&query={lat},{lon}"
    return "#"

# 5. UPLOAD E IA
uploaded_files = st.file_uploader("📂 ARRASTE AS FOTOS PARA VARREDURA", accept_multiple_files=True, type=['jpg', 'jpeg', 'png'])

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
            lat, lon = (coords[0], coords[1]) if coords else ("N/A", "N/A")
            
            dados_lavoura.append({
                "Amostra": file.name, "Pragas": len(results[0].boxes),
                "Latitude": lat, "Longitude": lon,
                "Maps_Link": link_google_maps(lat, lon),
                "Fazenda": nome_fazenda, "Safra": safra, "Talhao": talhao_id,
                "Cultura": tipo_plantio, "Data": datetime.now().strftime('%d/%m/%Y'),
                "_img_obj": img_com_caixas
            })
            progresso.progress((i + 1) / len(uploaded_files))
        except: continue

    if dados_lavoura:
        df = pd.DataFrame(dados_lavoura)
        total_pragas = df['Pragas'].sum()
        media_ponto = df['Pragas'].mean()
        status_sanitario = "CRÍTICO" if media_ponto > 15 else "NORMAL"

        st.markdown('<div class="report-section">', unsafe_allow_html=True)

        # 6. SUMÁRIO
        st.markdown(f"### 📊 Sumário Executivo: {nome_fazenda}")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Técnico", nome_tecnico)
        k2.metric("Cultura/Safra", f"{tipo_plantio} | {safra}")
        k3.metric("Total Detectado", f"{int(total_pragas)} un")
        k4.metric("Status", status_sanitario, delta="Alerta" if status_sanitario == "CRÍTICO" else "Ok")

        st.markdown("---")

        # 7. MAPA E GRÁFICOS
        col_mapa, col_intel = st.columns([1.6, 1])
        with col_mapa:
            st.subheader("📍 Georreferenciamento")
            df_geo = df[df['Latitude'] != "N/A"]
            if not df_geo.empty:
                m = folium.Map(location=[df_geo['Latitude'].mean(), df_geo['Longitude'].mean()], zoom_start=18)
                for _, row in df_geo.iterrows():
                    cor = 'red' if row['Pragas'] > 15 else 'orange' if row['Pragas'] > 5 else 'green'
                    folium.CircleMarker([row['Latitude'], row['Longitude']], radius=10, color=cor, fill=True).add_to(m)
                st_folium(m, width="100%", height=500)

        with col_intel:
            st.subheader("📈 Análise Técnica")
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number", value = media_ponto,
                gauge = {'axis': {'range': [0, 50]}, 'bar': {'color': "#1b5e20"},
                         'steps': [{'range': [0, 15], 'color': "#c8e6c9"}, {'range': [15, 30], 'color': "#fff9c4"}, {'range': [30, 50], 'color': "#ffcdd2"}]}))
            fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

            # TOP 10 CANDLESTICK
            st.write("**🕯️ Volatilidade: Top 10 Pontos**")
            df_top10 = df.nlargest(10, 'Pragas')
            fig_candle = go.Figure(data=[go.Candlestick(
                x=df_top10['Amostra'], open=df_top10['Pragas']*0.9, high=df_top10['Pragas'],
                low=df_top10['Pragas']*0.7, close=df_top10['Pragas']*0.95,
                increasing_line_color='#68CAED', decreasing_line_color='#68CAED')])
            fig_candle.update_layout(height=250, xaxis_rangeslider_visible=False, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig_candle, use_container_width=True)

        # 8. EXPORTAR
        st.markdown("---")
        with st.expander("📊 Exportar Relatório"):
            df_export = df.drop(columns=['_img_obj'])
            csv = df_export.to_csv(index=False, sep=';', encoding='utf-8-sig').encode('utf-8-sig')
            st.download_button("📥 Baixar CSV para Excel", csv, f"Relatorio_{nome_fazenda}.csv", "text/csv")

        # 9. GALERIA COM BOTÃO NEON AZUL #68CAED
        st.subheader("📸 Galeria de Focos e Navegação GPS")
        for _, row in df.nlargest(10, 'Pragas').iterrows():
            g1, g2 = st.columns([1.5, 1])
            with g1:
                st.image(row['_img_obj'], use_container_width=True)
            with g2:
                st.markdown(f"""
                <div style="background: white; padding: 20px; border-radius: 15px; border: 1px solid #eee; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
                    <h3 style="margin-top:0;">🪲 {row['Pragas']} Pragas</h3>
                    <p><b>Amostra:</b> {row['Amostra']}</p>
                    <p><b>Lat/Lon:</b> {row['Latitude']}, {row['Longitude']}</p>
                    <hr>
                    <a href="{row['Maps_Link']}" target="_blank" style="text-decoration:none;">
                        <button class="loc-btn">📍 LOCALIZAR AGORA</button>
                    </a>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("---")
        
        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("💡 Aguardando fotos para processamento...")
