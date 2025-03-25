import streamlit as st

# Función de la página de bienvenida
def mostrar_bienvenida():
    # Configuración de estilos (puedes mover esto a un lugar común si lo usas en varias páginas)
    st.markdown("""
    <style>
        .title {
            font-size: 36px !important;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 20px;
            font-weight: bold;
        }
        .header {
            font-size: 24px !important;
            color: #2e8b57;
            margin-top: 30px;
            margin-bottom: 15px;
            border-bottom: 2px solid #2e8b57;
            padding-bottom: 5px;
        }
        .info-box {
            border-radius: 10px;
            padding: 15px;
            margin: 15px 0;
            background-color: #f0f8ff;
            border-left: 5px solid #1f77b4;
        }
        .emoji-container {
            text-align: center;
            font-size: 30px;
            margin: 10px 0 20px 0;
        }
        .feature-list {
            background-color: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            margin: 15px 0;
        }
        .feature-item {
            margin-bottom: 10px;
            font-size: 16px;
        }
        .contact-box {
            background-color: #e8f4f8;
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
            text-align: center;
            border-left: 5px solid #2e8b57;
        }
    </style>
    """, unsafe_allow_html=True)

    # Título principal
    st.markdown('<div class="title">Aplicación de Detección de Anomalías en Tiempo Real</div>', unsafe_allow_html=True)
    st.markdown('<div class="emoji-container">🚰🔍🔒</div>', unsafe_allow_html=True)
    
    # Introducción
    with st.container():
        st.markdown("""
        <div class="info-box">
            <strong>Esta aplicación tiene como objetivo detectar posibles anomalías y ataques en una planta de tratamiento de agua (SWaT) en tiempo real.</strong> 🕛<br><br>
            Utiliza modelos avanzados de aprendizaje automático para identificar comportamientos inusuales en los datos de sensores y actuadores.
        </div>
        """, unsafe_allow_html=True)
    
    # Cómo funciona
    st.markdown('<div class="header">¿Cómo funciona el sistema? 🛠️</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-list">
        <div class="feature-item">📂 <strong>Sube un archivo CSV</strong> con los datos de los sensores de la planta</div>
        <div class="feature-item">🤖 La aplicación <strong>procesará automáticamente</strong> tus datos utilizando modelos pre-entrenados</div>
        <div class="feature-item">🔍 <strong>Detectará anomalías</strong> y posibles ataques en los datos</div>
        <div class="feature-item">📊 Generará un <strong>informe detallado</strong> con visualizaciones interactivas</div>
        <div class="feature-item">🚨 Proporcionará <strong>alertas tempranas</strong> para acciones preventivas</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sección de contacto
    st.markdown("""
    <div class="contact-box">
        <p style="font-size: 18px; margin-bottom: 10px;">¿Tienes alguna duda o necesitas asistencia?</p>
        <p style="font-size: 16px;">✉️ Contacta conmigo en <strong>https://github.com/Osss04</strong></p>
    </div>
    """, unsafe_allow_html=True)






mostrar_bienvenida()