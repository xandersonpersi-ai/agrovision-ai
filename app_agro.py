import streamlit as st
import pandas as pd
from ultralytics import YOLO
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import folium
from streamlit_folium import folium_static
from exif import Image as ExifImage

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="AgroVision Pro AI", layout="wide", page_icon="🌱")

# --- ESTILO CSS PARA MOBILE ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# --- CARREGAMENTO DO MODELO ---
@st.cache_resource
def load_model():
    # Certifique-se que o arquivo best.pt está no seu GitHub
    return YOLO("best.pt")

model = load_model()

# --- TÍTULO E INPUTS ---
st.title("🌱 AgroVision Pro AI")
st.subheader("Sistema Inteligente de Monitoramento - Fazenda Santa Fé")

with st.sidebar:
    st.header("Configurações")
    nome_fazenda = st.text_input("Nome da Propriedade", "Fazenda Santa Fé")
    nome_tecnico = st.text_input("Responsável Técnico", "Anderson Silva")
    conf_threshold = st.slider("Sensibilidade da IA", 0.1, 1.0, 0.25)
    
uploaded_files = st.file_uploader("Suba as fotos do drone ou celular", accept_multiple_files=True, type=['jpg', 'jpeg', 'png'])

if uploaded_files:
    dados_coletados = []
    
    # Processamento das Imagens
    for i, file in enumerate(uploaded_files):
        img = Image.open(file)
        results = model.predict(source=img, conf=conf_threshold)
        qtd_pragas = len(results[0].boxes)
        
        # Simulação de Coordenadas (Caso a foto não tenha GPS)
        lat = -15.7 + (i * 0.001)
        lon = -47.8 + (i * 0.001)
        
        dados_coletados.append({
            "Amostra": f"Img_{i+1}",
            "Pragas": qtd_pragas,
            "Lat": lat,
            "Lon": lon,
            "Data": datetime.now().strftime("%d/%m/%Y")
        })

    df = pd.DataFrame(dados_coletados)

    # --- DASHBOARD PRINCIPAL ---
    col1, col2 = st.columns([1, 1])

    with col1:
        # 1. VELOCÍMETRO (KPI)
        media_pragas = df['Pragas'].mean()
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = media_pragas,
            title = {'text': "Média de Pragas / Ponto"},
            gauge = {
                'axis': {'range': [0, max(50, media_pragas*2)]},
                'bar': {'color': "darkred"},
                'steps': [
                    {'range': [0, 10], 'color': "lightgreen"},
                    {'range': [10, 30], 'color': "yellow"},
                    {'range': [30, 50], 'color': "orange"}
                ]
            }
        ))
        fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col2:
        # 2. GRÁFICO DE BARRAS INTELIGENTE (TOP 10)
        # Se houver mais de 10 fotos, filtramos para não tumultuar o mobile
        if len(df) > 10:
            df_plot = df.nlargest(10, 'Pragas')
            titulo = "Top 10 Pontos Críticos"
        else:
            df_plot = df
            titulo = "Infestação por Amostra"

        fig_bar = px.bar(
            df_plot, x='Amostra', y='Pragas', 
            color='Pragas', title=titulo,
            color_continuous_scale='Reds'
        )
        fig_bar.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20), xaxis_tickangle=-45)
        st.plotly_chart(fig_bar, use_container_width=True)

    # --- MAPA DE CALOR ---
    st.markdown("### 📍 Mapa de Calor de Infestação")
    m = folium.Map(location=[df['Lat'].mean(), df['Lon'].mean()], zoom_start=15)
    
    for _, row in df.iterrows():
        cor = 'red' if row['Pragas'] > 10 else 'green'
        folium.CircleMarker(
            location=[row['Lat'], row['Lon']],
            radius=row['Pragas'] + 5,
            color=cor,
            fill=True,
            popup=f"Amostra: {row['Amostra']} - Pragas: {row['Pragas']}"
        ).add_to(m)
    
    folium_static(m, width=700 if st.sidebar.checkbox("Ajustar mapa", False) else None)

    # --- TABELA DE DADOS DETALHADA (EXPANSÍVEL) ---
    with st.expander("🔍 Ver todos os dados brutos"):
        st.write(f"Total de amostras processadas: {len(df)}")
        st.dataframe(df, use_container_width=True)

    # --- RECOMENDAÇÃO DE IA ---
    st.info("🤖 **Recomendação AgroVision:**")
    if media_pragas > 20:
        st.error(f"ALERTA: Infestação alta na {nome_fazenda}. Recomendamos aplicação imediata nos pontos vermelhos do mapa.")
    else:
        st.success(f"Nível de infestação dentro do limite de controle na {nome_fazenda}. Continue monitorando.")

else:
    st.info("Aguardando o envio das imagens para gerar o diagnóstico...")
