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

# 1. CONFIGURAÇÃO DE INTERFACE COM ANIMAÇÕES CSS
st.set_page_config(page_title="AgroVision Pro | Intelligence", layout="wide")

st.markdown("""
    <style>
    /* Animação de entrada suave */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .main { background-color: #f0f4f2; }
    
    /* Cards que flutuam ao passar o mouse */
    .stMetric {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 15px;
        border-top: 5px solid #2e7d32;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    .stMetric:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }

    /* Botão Localizar com Pulso e Brilho */
    .loc-btn {
        display: inline-block;
        padding: 16px 20px;
        font-size: 16px;
        cursor: pointer;
        text-align: center;
        text-decoration: none;
        color: #fff;
        background: linear-gradient(45deg, #d32f2f, #f44336);
        border: none;
        border-radius: 12px;
        font-weight: bold;
        width: 100%;
        transition: 0.3s;
        box-shadow: 0 4px 15px rgba(211, 47, 47, 0.3);
    }
    .loc-btn:hover {
        background: linear-gradient(45deg, #b71c1c, #d32f2f);
        transform: scale(1.02);
        box-shadow: 0 6px 20px rgba(211, 47, 47, 0.5);
    }
    
    /* Container de animação para as seções */
    .animated-section {
        animation: fadeIn 0.8s ease-out;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. CABEÇALHO
st.title("AgroVision Pro AI 🛰️")
st.caption(f"Inteligência Geográfica de Pragas | {datetime.now().strftime('%d/%m/%Y %H:%M')}")
st.markdown("---")

# 3. SIDEBAR (CADASTRO)
st.sidebar.header("📋 Monitoramento")
with st.sidebar.expander("📝 Dados da Área", expanded=True):
    nome_fazenda = st.text_input("Fazenda", "Santa Fé")
    nome_tecnico = st.text_input("Técnico", "Anderson Silva")
    tipo_plantio = st.selectbox("Cultura", ["Soja", "Milho", "Algodão", "Cana"])
    talhao_id = st.text_input("Talhão", "01")

conf_threshold = st.sidebar.slider("Precisão da IA", 0.1, 0.9, 0.25)

# 4. FUNÇÕES DE SUPORTE
def extrair_gps_st(img_file):
    try:
        img = ExifImage(img_file)
        if img.has_exif:
            lat = (img.gps_latitude[0] + img.gps_latitude[1]/60 + img.gps_latitude[2]/3600) * (-1 if img.gps_latitude_ref == 'S' else 1)
            lon = (img.gps_longitude[0] + img.gps_longitude[1]/60 + img.gps_longitude[2]/3600) * (-1 if img.gps_longitude_ref == 'W' else 1)
            return lat, lon
    except: return None
    return None

# 5. PROCESSAMENTO
uploaded_files = st.file_uploader("🚀 SOLTE AS FOTOS AQUI", accept_multiple_files=True, type=['jpg', 'jpeg', 'png'])

if uploaded_files:
    model = YOLO('best.pt' if os.path.exists('best.pt') else 'yolov8n.pt')
    dados = []
    
    with st.spinner('🤖 IA está varrendo o talhão...'):
        for i, file in enumerate(uploaded_files):
            img = Image.open(file)
            results = model.predict(source=img, conf=conf_threshold)
            img_res = Image.fromarray(results[0].plot()[:, :, ::-1])
            
            file.seek(0)
            coords = extrair_gps_st(file)
            lat, lon = (coords[0], coords[1]) if coords else ("N/A", "N/A")
            
            dados.append({
                "Amostra": file.name, "Pragas": len(results[0].boxes),
                "Lat": lat, "Lon": lon, "_img": img_res,
                "Maps": f"https://www.google.com/maps/search/?api=1&query={lat},{lon}" if lat != "N/A" else "#"
            })

    if dados:
        df = pd.DataFrame(dados)
        media = df['Pragas'].mean()
        
        # 6. DASHBOARD ANIMADO (KPIs)
        st.markdown('<div class="animated-section">', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Propriedade", nome_fazenda)
        c2.metric("Talhão", talhao_id)
        c3.metric("Total Pragas", int(df['Pragas'].sum()))
        c4.metric("Status", "CRÍTICO" if media > 15 else "OK", delta="- Alerta" if media > 15 else "+ Normal")
        
        st.markdown("---")

        # 7. MAPA E GRÁFICOS
        col_m, col_g = st.columns([1.5, 1])
        with col_m:
            st.subheader("📍 Mapa de Calor")
            df_geo = df[df['Lat'] != "N/A"]
            if not df_geo.empty:
                m = folium.Map(location=[df_geo['Lat'].mean(), df_geo['Lon'].mean()], zoom_start=18)
                for _, r in df_geo.iterrows():
                    folium.CircleMarker([r['Lat'], r['Lon']], radius=10, color='red' if r['Pragas']>15 else 'green', fill=True).add_to(m)
                st_folium(m, width="100%", height=450)

        with col_g:
            st.subheader("📊 Pressão de Pragas")
            fig = go.Figure(go.Indicator(mode="gauge+number", value=media, gauge={'axis': {'range': [0, 50]}, 'bar': {'color': "#2e7d32"}}))
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)

        # 8. DOWNLOAD
        st.markdown("---")
        with st.expander("📂 Exportar Dados para Excel"):
            df_ex = df.drop(columns=['_img'])
            csv = df_ex.to_csv(index=False, sep=';', encoding='utf-8-sig').encode('utf-8-sig')
            st.download_button("📥 Baixar Relatório Completo", csv, "agro_report.csv", "text/csv")

        # 9. GALERIA DE EVIDÊNCIAS COM BOTÃO ANIMADO
        st.subheader("📸 Evidências com Navegação GPS")
        for _, row in df.nlargest(10, 'Pragas').iterrows():
            g1, g2 = st.columns([1.6, 1])
            with g1:
                st.image(row['_img'], use_container_width=True)
            with g2:
                st.markdown(f"""
                <div style="background: white; padding: 20px; border-radius: 15px; border: 1px solid #ddd;">
                    <h3>🪲 {row['Pragas']} Pragas</h3>
                    <p><b>Amostra:</b> {row['Amostra']}</p>
                    <p><b>Ponto GPS:</b> {row['Lat']}, {row['Lon']}</p>
                    <hr>
                    <a href="{row['Maps']}" target="_blank" style="text-decoration: none;">
                        <div class="loc-btn">📍 LOCALIZAR NO MAPA</div>
                    </a>
                    <p style="font-size: 12px; color: #666; margin-top: 10px; text-align: center;">Clique para abrir rota GPS</p>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
