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

# CSS para garantir que os gráficos não "quebrem" no celular
st.markdown("""
    <style>
    .main { background-color: #f4f7f6; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 15px; border-top: 5px solid #2e7d32; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    div[data-testid="stExpander"] { background-color: #ffffff; border-radius: 10px; }
    /* Ajuste para mobile não achatar os gráficos */
    [data-testid="stPlotlyChart"] { min-height: 300px; }
    </style>
    """, unsafe_allow_html=True)

# 2. CABEÇALHO DINÂMICO
col_logo, col_tit = st.columns([1, 6])
with col_tit:
    st.title("AgroVision Pro AI 🛰️")
    st.caption(f"Plataforma de Diagnóstico Digital | Sessão: {datetime.now().strftime('%d/%m/%Y %H:%M')}")

st.markdown("---")

# 3. FICHA TÉCNICA E CONTROLE (SIDEBAR)
st.sidebar.header("📋 Cadastro de Campo")
with st.sidebar.expander("Identificação", expanded=True):
    nome_fazenda = st.text_input("Propriedade", "Fazenda Santa Fé")
    nome_tecnico = st.text_input("Responsável Técnico", "Anderson Silva")
    tipo_plantio = st.selectbox("Cultura Atual", ["Soja", "Milho", "Algodão", "Cana", "Outros"])
    safra = st.text_input("Ciclo / Safra", "2025/2026")
    talhao_id = st.text_input("Identificação do Talhão", "Talhão 01")

with st.sidebar.expander("Configurações de IA"):
    conf_threshold = st.slider("Sensibilidade (Confidence)", 0.01, 1.0, 0.15)
    st.info("Aumente a sensibilidade se a IA estiver ignorando pragas pequenas.")

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
    "📂 ARRASTE AS FOTOS DA VARREDURA (Apenas JPG, PNG)", 
    accept_multiple_files=True, 
    type=['jpg', 'jpeg', 'png']
)

if uploaded_files:
    # Carregamento do Modelo (Cacheado para ser rápido)
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
            
            file.seek(0)
            coords = extrair_gps_st(file)
            num_pragas = len(results[0].boxes)
            
            dados_lavoura.append({
                "Amostra": file.name, 
                "Pragas": num_pragas,
                "Lat": coords[0] if coords else None, 
                "Lon": coords[1] if coords else None
            })
            progresso.progress((i + 1) / len(uploaded_files))
        except Exception: continue

    if dados_lavoura:
        df = pd.DataFrame(dados_lavoura)
        total_encontrado = df['Pragas'].sum()
        media_ponto = df['Pragas'].mean()

        # 6. MÉTRICAS DE IMPACTO (KPIs)
        st.markdown(f"### 📊 Sumário Executivo: {nome_fazenda}")
        k1, k2, k3, k4 = st.columns([1,1,1,1])
        k1.metric("Amostras", f"{len(df)} fotos")
        k2.metric("Cultura", tipo_plantio)
        k3.metric("Total Pragas", f"{total_encontrado} un")
        status_sanitario = "CRÍTICO" if media_ponto > 15 else "NORMAL"
        k4.metric("Status", status_sanitario, delta="Alerta" if status_sanitario == "CRÍTICO" else "Ok")

        st.markdown("---")

        # 7. MAPA E CENTRO DE INTELIGÊNCIA
        col_mapa, col_intel = st.columns([1.6, 1])
        
        with col_mapa:
            st.subheader("📍 Mapa de Infestação")
            df_geo = df.dropna(subset=['Lat', 'Lon'])
            if not df_geo.empty:
                m = folium.Map(location=[df_geo['Lat'].mean(), df_geo['Lon'].mean()], zoom_start=18)
                for _, row in df_geo.iterrows():
                    cor_ponto = 'red' if row['Pragas'] > 15 else 'orange' if row['Pragas'] > 5 else 'green'
                    folium.CircleMarker(
                        location=[row['Lat'], row['Lon']],
                        radius=10 + (row['Pragas'] * 0.5), # Ajuste de escala do círculo
                        color=cor_ponto, fill=True, fill_opacity=0.7,
                        popup=f"Amostra: {row['Amostra']}<br>Detectado: {row['Pragas']} pragas"
                    ).add_to(m)
                st_folium(m, width="100%", height=400)
            else:
                st.warning("⚠️ Fotos sem GPS. O mapa não pode ser gerado.")

        with col_intel:
            st.subheader("📈 Análise de Pressão")
            
            # Gráfico de Velocímetro (Gauge) - Fica ótimo no mobile
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = media_ponto,
                title = {'text': "Média por Ponto", 'font': {'size': 16}},
                gauge = {
                    'axis': {'range': [0, 50]},
                    'bar': {'color': "#1b5e20"},
                    'steps': [
                        {'range': [0, 15], 'color': "#c8e6c9"},
                        {'range': [15, 30], 'color': "#fff9c4"},
                        {'range': [30, 50], 'color': "#ffcdd2"}]
                }
            ))
            fig_gauge.update_layout(height=250, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_gauge, use_container_width=True)

            # MELHORIA MOBILE: Ranking apenas do que importa
            st.write("**Top 5 Pontos mais Críticos**")
            # Se tiver muitas imagens, pegamos apenas as 5 piores para não tumultuar
            df_top = df.nlargest(5, 'Pragas')
            fig_ranking = px.bar(df_top, x='Pragas', y='Amostra', orientation='h', 
                                 color='Pragas', color_continuous_scale='Reds')
            fig_ranking.update_layout(height=250, showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_ranking, use_container_width=True)

        # 8. RECOMENDAÇÃO TÉCNICA AUTOMATIZADA
        st.markdown("---")
        with st.container():
            st.subheader("💡 Recomendação de Manejo (IA)")
            if media_ponto > 15:
                st.error(f"**Atenção {nome_tecnico}:** O talhão **{talhao_id}** apresenta pressão acima do nível de dano econômico. Recomenda-se controle químico localizado nos pontos vermelhos.")
            else:
                st.success(f"Nível de pragas em **{nome_fazenda}** está sob controle. Manter monitoramento rotineiro.")

        # 9. TABELA DE DADOS (Organizada para muitas imagens)
        with st.expander("📂 Ver Detalhes de Todas as Imagens"):
            st.write(f"Total de {len(df)} amostras analisadas.")
            st.dataframe(df.sort_values(by='Pragas', ascending=False), use_container_width=True)
            st.download_button("📥 Exportar CSV", df.to_csv(index=False).encode('utf-8'), f"Relatorio_{nome_fazenda}.csv")

else:
    st.info("💡 Pronto para começar! Arraste as fotos de inspeção para gerar o dashboard.")
