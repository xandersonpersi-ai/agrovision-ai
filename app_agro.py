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

# 1. CONFIGURAÇÃO DE INTERFACE "PREMIUM" E RESPONSIVA
st.set_page_config(page_title="AgroVision Pro | Intelligence", layout="wide", page_icon="🌱")

# CSS Otimizado
st.markdown("""
    <style>
    .main { background-color: #f4f7f6; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 15px; border-top: 5px solid #2e7d32; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    div[data-testid="stExpander"] { background-color: #ffffff; border-radius: 10px; }
    [data-testid="stPlotlyChart"] { min-height: 300px; }
    </style>
    """, unsafe_allow_html=True)

# 2. CABEÇALHO DINÂMICO
st.title("AgroVision Pro AI 🛰️")
st.caption(f"Plataforma de Diagnóstico Digital | {datetime.now().strftime('%d/%m/%Y %H:%M')}")
st.markdown("---")

# 3. FICHA TÉCNICA E CONTROLE (SIDEBAR)
st.sidebar.header("📋 Cadastro de Campo")
nome_fazenda = st.sidebar.text_input("Propriedade", "Fazenda Santa Fé")
nome_tecnico = st.sidebar.text_input("Responsável Técnico", "Anderson Silva")
tipo_plantio = st.sidebar.selectbox("Cultura Atual", ["Soja", "Milho", "Algodão", "Cana", "Outros"])
talhao_id = st.sidebar.text_input("Identificação do Talhão", "Talhão 01")
conf_threshold = st.sidebar.slider("Sensibilidade IA", 0.01, 1.0, 0.15)

# 4. FUNÇÃO DE GEOLOCALIZAÇÃO
def extrair_gps_st(img_file):
    try:
        img = ExifImage(img_file)
        if img.has_exif:
            lat = (img.gps_latitude[0] + img.gps_latitude[1]/60 + img.gps_latitude[2]/3600) * (-1 if img.gps_latitude_ref == 'S' else 1)
            lon = (img.gps_longitude[0] + img.gps_longitude[1]/60 + img.gps_longitude[2]/3600) * (-1 if img.gps_longitude_ref == 'W' else 1)
            return lat, lon
    except: return None
    return None

# 5. ÁREA DE UPLOAD
uploaded_files = st.file_uploader(
    "📂 ARRASTE AS FOTOS DA VARREDURA", 
    accept_multiple_files=True, 
    type=['jpg', 'jpeg', 'png']
)

if uploaded_files:
    @st.cache_resource
    def load_yolo():
        return YOLO('best.pt' if os.path.exists('best.pt') else 'yolov8n.pt')
    
    model = load_yolo()
    dados_lavoura = []
    
    st.write("### ⚙️ Processando Inteligência Artificial...")
    progresso = st.progress(0)
    
    for i, file in enumerate(uploaded_files):
        try:
            img = Image.open(file)
            results = model.predict(source=img, conf=conf_threshold)
            
            # Captura a imagem com os quadrados (boxes) desenhados
            img_com_caixas = results[0].plot() 
            # Converte de BGR (OpenCV) para RGB (PIL/Streamlit)
            img_com_caixas = Image.fromarray(img_com_caixas[:, :, ::-1])
            
            file.seek(0)
            coords = extrair_gps_st(file)
            num_pragas = len(results[0].boxes)
            
            # MELHORIA: Nome limpo para o gráfico não tumultuar
            ponto_nome = f"Ponto {i+1:02d}"
            
            dados_lavoura.append({
                "Amostra": ponto_nome, 
                "Arquivo": file.name,
                "Pragas": num_pragas,
                "Lat": coords[0] if coords else None, 
                "Lon": coords[1] if coords else None,
                "Imagem_Proc": img_com_caixas # Guardamos para a galeria
            })
            progresso.progress((i + 1) / len(uploaded_files))
        except Exception: continue

    if dados_lavoura:
        df = pd.DataFrame(dados_lavoura)
        total_encontrado = df['Pragas'].sum()
        media_ponto = df['Pragas'].mean()

        # 6. MÉTRICAS DE IMPACTO
        st.markdown(f"### 📊 Sumário Executivo: {nome_fazenda}")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Amostras", f"{len(df)}")
        k2.metric("Total Pragas", f"{int(total_encontrado)}")
        k3.metric("Média/Ponto", f"{media_ponto:.1f}")
        status = "CRÍTICO" if media_ponto > 15 else "NORMAL"
        k4.metric("Status", status, delta="Alerta" if status == "CRÍTICO" else "Ok")

        st.markdown("---")

        # 7. MAPA E RANKING
        col_mapa, col_intel = st.columns([1.5, 1])
        
        with col_mapa:
            st.subheader("📍 Mapa de Infestação")
            df_geo = df.dropna(subset=['Lat', 'Lon'])
            if not df_geo.empty:
                m = folium.Map(location=[df_geo['Lat'].mean(), df_geo['Lon'].mean()], zoom_start=18)
                for _, row in df_geo.iterrows():
                    cor = 'red' if row['Pragas'] > 15 else 'orange' if row['Pragas'] > 5 else 'green'
                    folium.CircleMarker([row['Lat'], row['Lon']], radius=10, color=cor, fill=True).add_to(m)
                st_folium(m, width="100%", height=400)
            else:
                st.warning("⚠️ Fotos sem GPS.")

        with col_intel:
            st.subheader("📈 Ranking Crítico (Top 10)")
            # Gráfico Horizontal limpo
            df_top = df.nlargest(10, 'Pragas').sort_values('Pragas', ascending=True)
            fig_ranking = px.bar(df_top, x='Pragas', y='Amostra', orientation='h', 
                                 color='Pragas', color_continuous_scale='Reds', text='Pragas')
            fig_ranking.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="")
            st.plotly_chart(fig_ranking, use_container_width=True)

        # 8. NOVA GALERIA: AS 10 IMAGENS MAIS CRÍTICAS
        st.markdown("---")
        st.subheader("📸 Evidências: 10 Pontos com Maior Infestação")
        
        piores_amostras = df.nlargest(10, 'Pragas')
        
        # Cria uma grade de fotos (2 por linha para ficar bom no celular)
        cols = st.columns(2)
        for idx, (_, row) in enumerate(piores_amostras.iterrows()):
            with cols[idx % 2]:
                st.image(row['Imagem_Proc'], caption=f"{row['Amostra']} - {row['Pragas']} pragas detectadas", use_container_width=True)

        # 9. RECOMENDAÇÃO E DADOS
        st.markdown("---")
        if media_ponto > 15:
            st.error(f"**Atenção {nome_tecnico}:** O talhão **{talhao_id}** exige intervenção imediata.")
        else:
            st.success(f"Níveis de pragas em **{nome_fazenda}** estão sob controle.")

        with st.expander("📂 Ver Todos os Dados e Exportar"):
            st.dataframe(df.drop(columns=['Imagem_Proc']), use_container_width=True)
            st.download_button("📥 Baixar CSV", df.drop(columns=['Imagem_Proc']).to_csv(index=False).encode('utf-8'), f"Relatorio_{nome_fazenda}.csv")

else:
    st.info("💡 Arraste as fotos para gerar o diagnóstico em tempo real.")
