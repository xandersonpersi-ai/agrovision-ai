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
st.set_page_config(page_title="AgroVision Pro | Intelligence", layout="wide", page_icon="🌱")

st.markdown("""
    <style>
    .main { background-color: #F8FAFC; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); border-left: 6px solid #059669; }
    </style>
    """, unsafe_allow_html=True)

# 2. CABEÇALHO
st.title("AgroVision Pro AI 🛰️")
st.caption(f"Análise Técnica de Volatilidade de Pragas | {datetime.now().strftime('%d/%m/%Y %H:%M')}")

# 3. SIDEBAR
st.sidebar.header("⚙️ Parâmetros")
nome_fazenda = st.sidebar.text_input("Propriedade", "Fazenda Santa Fé")
conf_threshold = st.sidebar.slider("Sensibilidade IA", 0.01, 1.0, 0.20)

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

# 5. UPLOAD
uploaded_files = st.file_uploader("📂 SUBIR VARREDURA PARA ANÁLISE DE VELAS", accept_multiple_files=True, type=['jpg', 'jpeg', 'png'])

if uploaded_files:
    @st.cache_resource
    def load_yolo():
        return YOLO('best.pt' if os.path.exists('best.pt') else 'yolov8n.pt')
    
    model = load_yolo()
    dados_lavoura = []
    
    for i, file in enumerate(uploaded_files):
        try:
            img = Image.open(file)
            results = model.predict(source=img, conf=conf_threshold)
            num_pragas = len(results[0].boxes)
            
            # Criando "Sectores" fictícios para gerar as velas (ex: Setor A, B, C...)
            setor = f"Setor {chr(65 + (i // 5))}" 
            
            dados_lavoura.append({
                "Setor": setor,
                "Pragas": num_pragas,
                "Amostra": file.name
            })
        except: continue

    if dados_lavoura:
        df = pd.DataFrame(dados_lavoura)
        
        # Agrupando dados para criar a lógica de Candle (Vela)
        # Open = média inicial, Close = média final, High = max pragas, Low = min pragas
        df_candle = df.groupby('Setor')['Pragas'].agg(['mean', 'max', 'min', 'first', 'last']).reset_index()

        st.markdown("### 🕯️ Análise de Volatilidade por Setor")
        st.caption("Este gráfico mostra a variação e os picos de pragas em cada região da fazenda.")

        # 6. GRÁFICO DE VELAS (CANDLESTICK)
        fig_candle = go.Figure(data=[go.Candlestick(
            x=df_candle['Setor'],
            open=df_candle['first'],
            high=df_candle['max'],
            low=df_candle['min'],
            close=df_candle['last'],
            increasing_line_color= '#ef4444', # Vermelho se a infestação subiu no setor
            decreasing_line_color= '#059669'  # Verde se a infestação caiu ou é baixa
        )])

        fig_candle.update_layout(
            height=450,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_rangeslider_visible=False,
            template="plotly_white"
        )
        st.plotly_chart(fig_candle, use_container_width=True)

        # 7. MÉTRICAS BÁSICAS
        c1, c2 = st.columns(2)
        c1.metric("Setor mais Crítico", df_candle.loc[df_candle['max'].idxmax()]['Setor'])
        c2.metric("Pico de Pragas Encontrado", int(df['Pragas'].max()))

        st.markdown("---")
        st.info("💡 **Dica para o Investidor:** Velas vermelhas longas indicam setores com alta instabilidade e necessidade de intervenção pesada.")

else:
    st.info("Aguardando fotos para gerar análise de velas...")
