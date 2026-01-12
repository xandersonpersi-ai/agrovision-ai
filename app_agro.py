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
import webbrowser

# 1. CONFIGURAÇÃO DE INTERFACE
st.set_page_config(page_title="AgroVision Pro | Intelligence", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f4f7f6; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 15px; border-top: 5px solid #2e7d32; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .stButton>button { width: 100%; border-radius: 10px; background-color: #2e7d32; color: white; height: 3em; }
    </style>
    """, unsafe_allow_html=True)

# 2. CABEÇALHO
st.title("AgroVision Pro AI 🛰️")
st.caption(f"Inteligência de Campo & Navegação GPS | {datetime.now().strftime('%d/%m/%Y %H:%M')}")

# 3. FICHA TÉCNICA (SIDEBAR)
st.sidebar.header("📋 Cadastro de Campo")
with st.sidebar.expander("Identificação", expanded=True):
    nome_fazenda = st.text_input("Propriedade", "Fazenda Santa Fé")
    nome_tecnico = st.text_input("Responsável Técnico", "Anderson Silva")
    tipo_plantio = st.selectbox("Cultura", ["Soja", "Milho", "Algodão", "Cana", "Outros"])
    safra = st.text_input("Safra", "2025/2026")
    talhao_id = st.text_input("Talhão", "Talhão 01")

conf_threshold = st.sidebar.slider("Sensibilidade IA", 0.01, 1.0, 0.15)

# 4. FUNÇÃO GPS E LINKS
def extrair_gps_st(img_file):
    try:
        img = ExifImage(img_file)
        if img.has_exif:
            lat = (img.gps_latitude[0] + img.gps_latitude[1]/60 + img.gps_latitude[2]/3600) * (-1 if img.gps_latitude_ref == 'S' else 1)
            lon = (img.gps_longitude[0] + img.gps_longitude[1]/60 + img.gps_longitude[2]/3600) * (-1 if img.gps_longitude_ref == 'W' else 1)
            return lat, lon
    except: return None
    return None

def criar_link_maps(lat, lon):
    if lat != "N/A":
        return f"https://www.google.com/maps/search/?api=1&query={lat},{lon}"
    return "Sem GPS"

# 5. UPLOAD E PROCESSAMENTO
uploaded_files = st.file_uploader("📂 CARREGAR VARREDURA DO TALHÃO", accept_multiple_files=True, type=['jpg', 'jpeg', 'png'])

if uploaded_files:
    model = YOLO('best.pt' if os.path.exists('best.pt') else 'yolov8n.pt')
    dados_lavoura = []
    
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
                "Amostra": file.name,
                "Pragas": len(results[0].boxes),
                "Latitude": lat,
                "Longitude": lon,
                "Google_Maps": criar_link_maps(lat, lon),
                "Fazenda": nome_fazenda,
                "Talhao": talhao_id,
                "Cultura": tipo_plantio,
                "_img": img_com_caixas
            })
        except: continue

    if dados_lavoura:
        df = pd.DataFrame(dados_lavoura)
        media_ponto = df['Pragas'].mean()
        status = "CRÍTICO" if media_ponto > 15 else "NORMAL"

        # 6. DASHBOARD
        st.markdown("---")
        k1, k2, k3 = st.columns(3)
        k1.metric("Média de Pragas", f"{media_ponto:.1f}")
        k2.metric("Status Sanitário", status)
        k3.metric("Pontos Amostrados", len(df))

        # 7. MAPA E GRÁFICO
        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader("📍 Mapa de Infestação")
            df_geo = df[df['Latitude'] != "N/A"]
            if not df_geo.empty:
                m = folium.Map(location=[df_geo['Latitude'].mean(), df_geo['Longitude'].mean()], zoom_start=17)
                for _, row in df_geo.iterrows():
                    cor = 'red' if row['Pragas'] > 15 else 'green'
                    folium.CircleMarker([row['Latitude'], row['Longitude']], radius=8, color=cor, fill=True, popup=f"{row['Pragas']} pragas").add_to(m)
                st_folium(m, width="100%", height=400)

        with c2:
            st.subheader("📈 Pressão")
            fig = go.Figure(go.Indicator(mode="gauge+number", value=media_ponto, gauge={'axis': {'range': [0, 50]}, 'bar': {'color': "darkgreen"}}))
            fig.update_layout(height=250, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

        # 8. DOWNLOAD DO RELATÓRIO (COM LINK DO GPS)
        st.markdown("---")
        with st.expander("📊 Baixar Dados para Excel (CSV)"):
            df_export = df.drop(columns=['_img'])
            csv = df_export.to_csv(index=False, sep=';', encoding='utf-8-sig').encode('utf-8-sig')
            st.download_button("📥 Baixar Relatório com Links de GPS", csv, f"Relatorio_{talhao_id}.csv", "text/csv")

        # 9. GALERIA COM BOTÃO "LOCALIZAR PRAGA" (O SEU PEDIDO)
        st.markdown("---")
        st.subheader("📸 Galeria de Amostras e Localização em Tempo Real")
        
        for _, row in df.sort_values(by="Pragas", ascending=False).iterrows():
            col_img, col_info = st.columns([1.5, 1])
            with col_img:
                st.image(row['_img'], use_container_width=True)
            with col_info:
                st.write(f"### 🪲 {row['Pragas']} Pragas")
                st.write(f"**Amostra:** {row['Amostra']}")
                st.write(f"**Coordenadas:** {row['Latitude']}, {row['Longitude']}")
                
                # BOTÃO DE LOCALIZAR
                if row['Latitude'] != "N/A":
                    link = row['Google_Maps']
                    st.markdown(f"""
                        <a href="{link}" target="_blank">
                            <button style="
                                width: 100%;
                                background-color: #d32f2f;
                                color: white;
                                border: none;
                                padding: 15px;
                                border-radius: 10px;
                                cursor: pointer;
                                font-weight: bold;
                                font-size: 16px;
                                display: flex;
                                align-items: center;
                                justify-content: center;
                                gap: 10px;">
                                📍 LOCALIZAR PRAGA NO MAPA
                            </button>
                        </a>
                        """, unsafe_allow_html=True)
                else:
                    st.warning("GPS indisponível para esta foto.")
                
                st.info("O botão abrirá o Google Maps para navegação até o ponto exato.")
            st.markdown("---")

else:
    st.info("Aguardando varredura de fotos...")
