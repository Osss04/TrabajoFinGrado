import streamlit as st

def mostrar_bienvenida():
    #configuración de estilos
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
        .feature-subitem {
            margin-left: 30px;
            font-size: 15px;
            color: #555;
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

    #título
    st.markdown('<div class="title">Aplicación de Detección de Anomalías en Tiempo Real</div>', unsafe_allow_html=True)
    st.markdown('<div class="emoji-container">🚰🔍🔒</div>', unsafe_allow_html=True)
    
    #introducción
    with st.container():
        st.markdown("""
        <div class="info-box">
            <strong>Esta aplicación tiene como objetivo simular la tarea de detectar posibles anomalías y ataques en una planta de tratamiento de agua (SWaT) en tiempo real.</strong> 🕛<br><br>
            Se utiliza una red neuronal compuesta por LSTM y capas densas que actúa como regresor. Este modelo es entrenado con el comportamiento normal de los sensores de la planta de agua,
            por lo que para cada sensor predecirá dicho comportamiento. Cuando le llegue un registro que tenga mucha diferencia con el valor normal predicho, lo considerará anomalía.
        </div>
        """, unsafe_allow_html=True)
    

    st.markdown('<div class="header">¿Cómo funciona la detección de anomalías? 🛠️</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-list">
        <div class="feature-item">📖 Para ver cómo funciona la planta de tratamiento de agua y sus sensores, pulsa en la página <strong>Descripción del Sistema.</strong></div>
        <div class="feature-item">🔍 Para comenzar la simulación de la detección de anomalías en tiempo real, pulsa en <strong>Detección de Anomalías</strong>.
            <div class="feature-subitem">🚀 Una vez en la página, para iniciar el proceso, pulsa el botón <strong>Comenzar análisis</strong>.</div>
            <div class="feature-subitem">🚨 Se realizará la detección en tiempo real mostrando en qué <strong>parte del sistema</strong> se dan las anomalías y se llevará a cabo un <strong>registro</strong> de ellas.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
mostrar_bienvenida()