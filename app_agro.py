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

# 3. FICHA TÉCNICA E CONTROLE (SIDEBAR)
st.sidebar.header("📋 Cadastro de Campo")
with st.sidebar.expander("Identificação", expanded=True):
    nome_fazenda = st.text_input("Propriedade", "Fazenda Santa Fé")
    nome_tecnico = st.text_input("Responsável Técnico", "Anderson Silva")
    tipo_plantio = st.selectbox("Cultura Atual", ["Soja", "Milho", "Algodão", "Cana", "Outros"])
    safra = st.text_input("Ciclo / Safra", "2025/2026")
    talhao_id = st.text_input("Identificação do Talhão", "Talhão 01")

with st.sidebar.expander("Configurações de IA"):
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

# 5. UPLOAD E PROCESSAMENTO IA
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
                "Imagem_Proc": img_com_caixas,
                "Data": datetime.now().strftime('%d/%m/%Y'),
                "Fazenda": nome_fazenda,
                "Tecnico": nome_tecnico,
                "Talhao": talhao_id,
                "Cultura": tipo_plantio
            })
            progresso.progress((i + 1) / len(uploaded_files))
        except: continue

    if dados_lavoura:
        df = pd.DataFrame(dados_lavoura)
        total_encontrado = df['Pragas'].sum()
        media_ponto = df['Pragas'].mean()
        status_sanitario = "CRÍTICO" if media_ponto > 15 else "NORMAL"

        # 6. SUMÁRIO EXECUTIVO (KPIs)
        st.markdown(f"### 📊 Relatório de Infestação: {nome_fazenda}")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Técnico", nome_tecnico)
        k2.metric("Cultura/Safra", f"{tipo_plantio} | {safra}")
        k3.metric("Total Detectado", f"{int(total_encontrado)} pragas")
        k4.metric("Status Sanitário", status_sanitario, delta="Ação Necessária" if status_sanitario == "CRÍTICO" else "Sob Controle")

        st.markdown("---")

        # 7. MAPA E CENTRO DE INTELIGÊNCIA
        col_mapa, col_intel = st.columns([1.6, 1])
        
        with col_mapa:
            st.subheader("📍 Mapa de Calor e Localização")
            df_geo = df.dropna(subset=['Lat', 'Lon'])
            if not df_geo.empty:
                m = folium.Map(location=[df_geo['Lat'].mean(), df_geo['Lon'].mean()], zoom_start=18, tiles=None)
                folium.TileLayer('OpenStreetMap', name="Mapa Base", control=False).add_to(m)
                for _, row in df_geo.iterrows():
                    cor = 'red' if row['Pragas'] > 15 else 'orange' if row['Pragas'] > 5 else 'green'
                    folium.CircleMarker([row['Lat'], row['Lon']], radius=10, color=cor, fill=True, popup=f"{row['Pragas']} pragas").add_to(m)
                st_folium(m, width="100%", height=500)
            else:
                st.warning("⚠️ Atenção: Fotos sem GPS. O mapa de localização não pôde ser gerado.")

        with col_intel:
            st.subheader("📈 Análise de Pressão")
            # Velocímetro Original com Faixas de Risco
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number", value = media_ponto,
                title = {'text': "Pressão Média (Pragas/Foto)", 'font': {'size': 16}},
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

            # Gráfico de Velas (Candlestick) - Top 10 Pontos Críticos
            st.write("**🕯️ Volatilidade de Infestação (Top 10)**")
            df_top10 = df.nlargest(10, 'Pragas')
            fig_candle = go.Figure(data=[go.Candlestick(
                x=df_top10['Amostra'], 
                open=df_top10['Pragas']*0.85, 
                high=df_top10['Pragas'],
                low=df_top10['Pragas']*0.6, 
                close=df_top10['Pragas']*0.9,
                increasing_line_color='#b71c1c', 
                decreasing_line_color='#b71c1c'
            )])
            fig_candle.update_layout(height=250, xaxis_rangeslider_visible=False, margin=dict(l=0, r=0, t=0, b=0), template="plotly_white")
            st.plotly_chart(fig_candle, use_container_width=True)

        # 8. RECOMENDAÇÃO TÉCNICA ESTRUTURADA
        st.markdown("---")
        with st.container():
            st.subheader("💡 Parecer Técnico Automático")
            rec_col1, rec_col2 = st.columns([1, 2])
            with rec_col1:
                if status_sanitario == "CRÍTICO":
                    st.error("🚨 ALERTA: Nível de dano econômico atingido.")
                else:
                    st.success("✅ MONITORAMENTO: Nível de infestação tolerável.")
            with rec_col2:
                texto_rec = f"Relatório gerado para a unidade **{talhao_id}**. Com base na detecção por IA, a cultura de **{tipo_plantio}** apresenta uma média de **{media_ponto:.1f}** pragas por ponto amostral. "
                if status_sanitario == "CRÍTICO":
                    texto_rec += "Recomenda-se controle imediato nos focos de alta pressão (marcados em vermelho) para evitar perda de produtividade."
                else:
                    texto_rec += "Manter o cronograma de vistorias a cada 7 dias."
                st.write(texto_rec)

        # 9. DADOS BRUTOS ENRIQUECIDOS (PARA EXPORTAÇÃO)
        st.markdown("---")
        with st.expander("📊 Relatório Detalhado de Dados (Clique para Exportar)", expanded=False):
            # Criamos um DataFrame limpo para o CSV, contendo todos os campos de cadastro
            df_report = df.drop(columns=['Imagem_Proc'])
            st.write(f"**Dados consolidados do Talhão: {talhao_id}**")
            st.dataframe(df_report, use_container_width=True)
            
            csv_data = df_report.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Exportar Relatório Técnico (CSV)",
                data=csv_data,
                file_name=f"Relatorio_IA_{nome_fazenda}_{talhao_id}.csv",
                mime="text/csv",
                help="O arquivo CSV contém todas as coordenadas, contagens e dados de cadastro para integração em BI ou Excel."
            )

        # 10. GALERIA DE EVIDÊNCIAS
        st.subheader("📸 Evidências Visuais (Focos de Alta Pressão)")
        for _, row in df.nlargest(10, 'Pragas').iterrows():
            st.image(row['Imagem_Proc'], caption=f"Amostra: {row['Amostra']} | Detecção: {row['Pragas']} pragas | Talhão: {row['Talhao']}", use_container_width=True)
            st.markdown("---")

else:
    st.info("💡 Pronto para análise. Arraste as fotos para gerar o dashboard completo.")
