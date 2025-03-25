import streamlit as st
import pandas as pd
import time
import torch
import torch.nn as nn

# Definir el modelo LSTM
class LSTMPredictor(nn.Module):
    def __init__(self, input_size, lstm_neurons, dense_neurons, activation, output_size):
        super(LSTMPredictor, self).__init__()

        self.lstm_layers = nn.ModuleList()
        self.dense_layers = nn.ModuleList()
        self.dropout = nn.Dropout(0.1)  # dropout de 0.1 para evitar alto recall

        activations = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid()
        }
        self.activation = activations.get(activation, nn.ReLU())  # ReLU por defecto

        # Para agregar las capas LSTM
        for i in range(len(lstm_neurons)):
            input_dim = input_size if i == 0 else lstm_neurons[i-1]
            self.lstm_layers.append(nn.LSTM(input_dim, lstm_neurons[i], batch_first=True))

        # Para agregar las capas densas
        for i in range(len(dense_neurons)):
            input_dim = lstm_neurons[-1] if i == 0 else dense_neurons[i-1]
            self.dense_layers.append(nn.Linear(input_dim, dense_neurons[i]))

        # Para generar la capa de salida
        self.output_layer = nn.Linear(dense_neurons[-1], output_size)

    def forward(self, x):
        # Pasa por las capas LSTM
        for lstm in self.lstm_layers:
            x, _ = lstm(x)
            x = self.dropout(x)  # Después de cada capa LSTM hacemos dropout
        # Tomamos el último estado
        x = x[:, -1, :]

        # Pasa por las capas densas
        for dense in self.dense_layers:
            x = self.activation(dense(x))
            x = self.dropout(x)  # Después de cada capa densa hacemos dropout
        # Capa de salida
        x = self.output_layer(x)
        return x

# Cargar el modelo previamente entrenado
modelo = torch.load("Modelo/modelo_5.pth", weights_only=False)

# Estilos personalizados con HTML y CSS
st.markdown(
    """
    <style>
        .title {
            text-align: center;
            font-size: 36px;
            font-weight: bold;
            color: #030a83;
            margin-bottom: 10px;
        }
        .emoji {
            text-align: center;
            font-size: 30px;
            margin-bottom: 20px;
        }
        .container {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
            text-align: center;
            width: 80%; /* o un valor fijo como 600px */
            margin: 0 auto; /* esto lo centra horizontalmente */
        }
        .content {
            font-size: 18px;
            color: #333;
            line-height: 1.6;
        }
    </style>
    """,
    unsafe_allow_html=True
)

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


# Función para la página de descripción del sistema
def mostrar_descripcion_sistema():
    # Configuración inicial de estilo
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
        .centered-img {
            display: flex;
            justify-content: center;
            margin: 20px 0;
        }
        .info-box {
            border-radius: 10px;
            padding: 15px;
            margin: 15px 0;
            background-color: #f0f8ff;
            border-left: 5px solid #1f77b4;
        }
        .table-container {
            overflow-x: auto;
            margin: 20px 0;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0;
            font-size: 14px;
        }
        th {
            background-color: #1f77b4;
            color: white;
            padding: 12px;
            text-align: left;
        }
        td {
            padding: 10px;
            border-bottom: 1px solid #ddd;
        }
        tr:nth-child(even) {
            background-color: #f2f2f2;
        }
        tr:hover {
            background-color: #e6f7ff;
        }
        .emoji-container {
            text-align: center;
            font-size: 30px;
            margin: 10px 0 20px 0;
        }
    </style>
    """, unsafe_allow_html=True)

    # Título principal
    st.markdown('<div class="title">Descripción del Sistema SWaT</div>', unsafe_allow_html=True)
    st.markdown('<div class="emoji-container">🗒️🚰</div>', unsafe_allow_html=True)

    # Introducción
    with st.container():
        st.markdown("""
        <div class="info-box">
            <strong>En concreto, se analizará el comportamiento de la planta de tratamiento de agua conocida como <span style="color: #1f77b4;">SWaT (Secure Water Treatment)</span>.</strong><br><br>
            Esta planta de agua contiene seis procesos principales, y a continuación, se explican las fases del sistema:
        </div>
        """, unsafe_allow_html=True)

    # Diagrama de procesos
    st.markdown('<div class="header">Procesos de la Planta SWaT</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 6, 1])
    with col2:
        st.image("Imágenes/swat_Esquema.png", caption="Diagrama de los procesos principales de la planta SWaT", use_container_width=True)

    # Esquema de sensores
    st.markdown('<div class="header">Ubicación de Sensores</div>', unsafe_allow_html=True)
    with st.container():
        st.markdown("""
        <div class="info-box">
            A continuación se muestra dónde se ubica cada sensor en la planta:
        </div>
        """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 6, 1])
    with col2:
        st.image("Imágenes/esquema.png", caption="Esquema detallado de la planta SWaT mostrando ubicación de sensores", use_container_width=True)

    # Tabla de sensores
    st.markdown('<div class="header">Listado Completo de Sensores y Actuadores</div>', unsafe_allow_html=True)
    with st.container():
        st.markdown("""
        <div class="info-box">
            A continuación se muestran todos los sensores y actuadores del sistema con sus respectivas descripciones:
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="table-container">
        <table>
          <thead>
            <tr>
              <th>No.</th>
              <th>Nombre</th>
              <th>Tipo</th>
              <th>Descripción</th>
            </tr>
          </thead>
          <tbody>
            <tr><td>1</td><td>FIT-101</td><td>Sensor</td><td>Medidor de flujo; Mide el flujo de entrada al tanque de agua cruda.</td></tr>
            <tr><td>2</td><td>LIT-101</td><td>Sensor</td><td>Transmisor de nivel; Nivel del tanque de agua cruda.</td></tr>
            <tr><td>3</td><td>MV-101</td><td>Actuador</td><td>Válvula motorizada; Controla el flujo de agua al tanque de agua cruda.</td></tr>
            <tr><td>4</td><td>P-101</td><td>Actuador</td><td>Bomba; Bombea agua del tanque de agua cruda a la segunda etapa.</td></tr>
            <tr><td>5</td><td>P-102 (respaldo)</td><td>Actuador</td><td>Bomba; Bombea agua del tanque de agua cruda a la segunda etapa.</td></tr>
            <tr><td>6</td><td>AIT-201</td><td>Sensor</td><td>Analizador de conductividad; Mide el nivel de NaCl.</td></tr>
            <tr><td>7</td><td>AIT-202</td><td>Sensor</td><td>Analizador de pH; Mide el nivel de HCl.</td></tr>
            <tr><td>8</td><td>AIT-203</td><td>Sensor</td><td>Analizador ORP; Mide el nivel de NaOCl.</td></tr>
            <tr><td>9</td><td>FIT-201</td><td>Sensor</td><td>Transmisor de flujo; Controla las bombas dosificadoras.</td></tr>
            <tr><td>10</td><td>MV-201</td><td>Actuador</td><td>Válvula motorizada; Controla el flujo de agua al tanque de alimentación de UF.</td></tr>
            <tr><td>11</td><td>P-201</td><td>Actuador</td><td>Bomba dosificadora; Bomba dosificadora de NaCl.</td></tr>
            <tr><td>12</td><td>P-202 (respaldo)</td><td>Actuador</td><td>Bomba dosificadora; Bomba dosificadora de NaCl.</td></tr>
            <tr><td>13</td><td>P-203</td><td>Actuador</td><td>Bomba dosificadora; Bomba dosificadora de HCl.</td></tr>
            <tr><td>14</td><td>P-204 (respaldo)</td><td>Actuador</td><td>Bomba dosificadora; Bomba dosificadora de HCl.</td></tr>
            <tr><td>15</td><td>P-205</td><td>Actuador</td><td>Bomba dosificadora; Bomba dosificadora de NaOCl.</td></tr>
            <tr><td>16</td><td>P-206 (respaldo)</td><td>Actuador</td><td>Bomba dosificadora; Bomba dosificadora de NaOCl.</td></tr>
            <tr><td>17</td><td>DPIT-301</td><td>Sensor</td><td>Transmisor de presión diferencial; Controla el proceso de retro-lavado.</td></tr>
            <tr><td>18</td><td>FIT-301</td><td>Sensor</td><td>Medidor de flujo; Mide el flujo de agua en la etapa UF.</td></tr>
            <tr><td>19</td><td>LIT-301</td><td>Sensor</td><td>Transmisor de nivel; Nivel del tanque de alimentación de UF.</td></tr>
            <tr><td>20</td><td>MV-301</td><td>Actuador</td><td>Válvula motorizada; Controla el proceso de retro-lavado UF.</td></tr>
            <tr><td>21</td><td>MV-302</td><td>Actuador</td><td>Válvula motorizada; Controla el flujo de agua del proceso UF a la unidad de descloración.</td></tr>
            <tr><td>22</td><td>MV-303</td><td>Actuador</td><td>Válvula motorizada; Controla el drenaje de retro-lavado UF.</td></tr>
            <tr><td>23</td><td>MV-304</td><td>Actuador</td><td>Válvula motorizada; Controla el drenaje de UF.</td></tr>
            <tr><td>24</td><td>P-301 (respaldo)</td><td>Actuador</td><td>Bomba de alimentación UF; Bombea agua del tanque de alimentación UF al tanque de alimentación RO a través de la filtración UF.</td></tr>
            <tr><td>25</td><td>P-302</td><td>Actuador</td><td>Bomba de alimentación UF; Bombea agua del tanque de alimentación UF al tanque de alimentación RO a través de la filtración UF.</td></tr>
            <tr><td>26</td><td>AIT-401</td><td>Sensor</td><td>Medidor de dureza RO del agua.</td></tr>
            <tr><td>27</td><td>AIT-402</td><td>Sensor</td><td>Medidor ORP; Controla la dosificación de NaHSO3 (P203), dosificación de NaOCl (P205).</td></tr>
            <tr><td>28</td><td>FIT-401</td><td>Sensor</td><td>Transmisor de flujo; Controla el desclorador UV.</td></tr>
            <tr><td>29</td><td>LIT-401</td><td>Actuador</td><td>Transmisor de nivel; Nivel del tanque de alimentación RO.</td></tr>
            <tr><td>30</td><td>P-401 (respaldo)</td><td>Actuador</td><td>Bomba; Bombea agua del tanque de alimentación RO al desclorador UV.</td></tr>
            <tr><td>31</td><td>P-402</td><td>Actuador</td><td>Bomba; Bombea agua del tanque de alimentación RO al desclorador UV.</td></tr>
            <tr><td>32</td><td>P-403</td><td>Actuador</td><td>Bomba de bisulfito de sodio.</td></tr>
            <tr><td>33</td><td>P-404 (respaldo)</td><td>Actuador</td><td>Bomba de bisulfito de sodio.</td></tr>
            <tr><td>34</td><td>UV-401</td><td>Actuador</td><td>Desclorador; Elimina el cloro del agua.</td></tr>
            <tr><td>35</td><td>AIT-501</td><td>Sensor</td><td>Analizador de pH RO; Mide el nivel de HCl.</td></tr>
            <tr><td>36</td><td>AIT-502</td><td>Sensor</td><td>Analizador ORP de alimentación RO; Mide el nivel de NaOCl.</td></tr>
            <tr><td>37</td><td>AIT-503</td><td>Sensor</td><td>Analizador de conductividad de alimentación RO; Mide el nivel de NaCl.</td></tr>
            <tr><td>38</td><td>AIT-504</td><td>Sensor</td><td>Analizador de conductividad permeada RO; Mide el nivel de NaCl.</td></tr>
            <tr><td>39</td><td>FIT-501</td><td>Sensor</td><td>Medidor de flujo; Medidor de flujo de entrada a la membrana RO.</td></tr>
            <tr><td>40</td><td>FIT-502</td><td>Sensor</td><td>Medidor de flujo; Flujo permeado RO.</td></tr>
            <tr><td>41</td><td>FIT-503</td><td>Sensor</td><td>Medidor de flujo; Flujo rechazado RO.</td></tr>
            <tr><td>42</td><td>FIT-504</td><td>Sensor</td><td>Medidor de flujo; Medidor de flujo de recirculación RO.</td></tr>
            <tr><td>43</td><td>P-501</td><td>Actuador</td><td>Bomba; Bombea agua desclorada a RO.</td></tr>
            <tr><td>44</td><td>P-502 (respaldo)</td><td>Actuador</td><td>Bomba; Bombea agua desclorada a RO.</td></tr>
            <tr><td>45</td><td>PIT-501</td><td>Sensor</td><td>Medidor de presión; Presión de alimentación RO.</td></tr>
            <tr><td>46</td><td>PIT-502</td><td>Sensor</td><td>Medidor de presión; Presión de permeado RO.</td></tr>
            <tr><td>47</td><td>PIT-503</td><td>Sensor</td><td>Medidor de presión; Presión de rechazo RO.</td></tr>
            <tr><td>48</td><td>FIT-601</td><td>Sensor</td><td>Medidor de flujo; Medidor de flujo de retro-lavado UF.</td></tr>
            <tr><td>49</td><td>P-601</td><td>Actuador</td><td>Bomba; Bombea agua del tanque de permeado RO al tanque de agua cruda (no se utiliza para la recolección de datos).</td></tr>
            <tr><td>50</td><td>P-602</td><td>Actuador</td><td>Bomba; Bombea agua del tanque de retro-lavado UF al filtro UF para limpiar la membrana.</td></tr>
            <tr><td>51</td><td>P-603</td><td>Actuador</td><td>No implementado en SWaT todavía.</td></tr>
          </tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)
    
    

# Función para la página de detección de anomalías
def mostrar_deteccion_anomalias():
    # Configuración de estilos
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
        .upload-box {
            border: 2px dashed #1f77b4;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            margin: 20px 0;
            background-color: #f8f9fa;
        }
        .data-preview {
            border-radius: 10px;
            padding: 15px;
            margin: 15px 0;
            background-color: #f0f8ff;
        }
        .processing-box {
            border-radius: 10px;
            padding: 15px;
            margin: 15px 0;
            background-color: #fff4e6;
        }
        .result-box {
            border-radius: 10px;
            padding: 15px;
            margin: 15px 0;
            background-color: #f0fff0;
        }
        .anomaly {
            color: #d9534f;
            font-weight: bold;
        }
        .normal {
            color: #5cb85c;
            font-weight: bold;
        }
        .progress-container {
            margin: 20px 0;
        }
        .file-name {
            font-weight: bold;
            color: #1f77b4;
        }
        .emoji-container {
            text-align: center;
            font-size: 30px;
            margin: 10px 0 20px 0;
        }
    </style>
    """, unsafe_allow_html=True)

    # Título principal
    st.markdown('<div class="title">Detección de Anomalías en Tiempo Real</div>', unsafe_allow_html=True)
    st.markdown('<div class="emoji-container">🔍🕛</div>', unsafe_allow_html=True)
    
    # Sección de carga de archivos
    st.markdown('<div class="header">📤 Carga tus datos</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="upload-box">
        <p style="font-size: 18px;">Sube un archivo CSV con los datos de los sensores para analizar</p>
        <p style="font-size: 14px; color: #666;">Formatos soportados: .csv (máx. 200MB)</p>
    </div>
    """, unsafe_allow_html=True)
    
    archivo = st.file_uploader("Selecciona un archivo", type=["csv"], label_visibility="collapsed")

    if archivo:
        # Mostrar nombre del archivo
        st.markdown(f'<div style="margin: 10px 0;">Archivo seleccionado: <span class="file-name">{archivo.name}</span></div>', unsafe_allow_html=True)
        
        # Cargar datos
        try:
            df = pd.read_csv(archivo)
            
            # Vista previa de datos
            st.markdown('<div class="header">👀 Vista previa de los datos</div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="data-preview">
                <p>Primeras 5 filas del dataset cargado:</p>
            </div>
            """, unsafe_allow_html=True)
            st.dataframe(df.head().style.set_properties(**{'background-color': '#f8f9fa'}), use_container_width=True)
            
            # Procesamiento en tiempo real
            st.markdown('<div class="header">⚙️ Procesamiento en tiempo real</div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="processing-box">
                <p>Analizando datos fila por fila con el modelo LSTM...</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Barra de progreso
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            resultados = []
            total_filas = len(df)
            
            for i, row in df.iterrows():
                # Simulación de procesamiento
                features = row.values.reshape(1, -1)
                prediccion = modelo(features)  # Usar tu modelo real aquí
                anomalia = prediccion > 0.5  # Ajustar umbral según necesidad
                resultados.append({"Índice": i, "Anomalía": "Sí" if anomalia else "No"})
                
                # Actualizar progreso
                percent_complete = int((i + 1) / total_filas * 100)
                progress_bar.progress(percent_complete)
                status_text.markdown(f"""
                <div style="margin: 5px 0;">
                    Fila {i}: 
                    <span class="{'anomaly' if anomalia else 'normal'}">
                        {'🚨 Anomalía detectada' if anomalia else '✅ Comportamiento normal'}
                    </span>
                </div>
                """, unsafe_allow_html=True)
                
                # Pequeña pausa para simular tiempo real
                time.sleep(0.1)
            
            # Resultados finales
            st.markdown('<div class="header">📊 Resultados del análisis</div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="result-box">
                <p>Resumen de detección de anomalías:</p>
            </div>
            """, unsafe_allow_html=True)
            
            resultados_df = pd.DataFrame(resultados)
            anomalias_count = resultados_df[resultados_df["Anomalía"] == "Sí"].shape[0]
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total de filas analizadas", total_filas)
            with col2:
                st.metric("Anomalías detectadas", anomalias_count, delta=f"{anomalias_count/total_filas*100:.1f}%")
            
            st.dataframe(
                resultados_df.style.applymap(
                    lambda x: 'background-color: #ffcccc' if x == 'Sí' else 'background-color: #ccffcc', 
                    subset=['Anomalía']
                ),
                use_container_width=True
            )
            
            # Botón para descargar resultados
            csv = resultados_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Descargar resultados",
                data=csv,
                file_name='resultados_anomalias.csv',
                mime='text/csv'
            )
            
        except Exception as e:
            st.error(f"Error al procesar el archivo: {str(e)}")

# Índice de la aplicación (selección entre las dos páginas) en la barra lateral
pagina_seleccionada = st.sidebar.selectbox("Selecciona una página", ["Inicio", "Descripción del Sistema","Detección de Anomalías"], label_visibility="collapsed")

# Cambiar apariencia del índice
st.sidebar.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        background-color: #f1f1f1;
        padding: 20px;
        border-radius: 10px;
    }
    .stSelectbox {
        font-size: 16px;
        padding: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

# Mostrar la página seleccionada
if pagina_seleccionada == "Inicio":
    mostrar_bienvenida()
elif pagina_seleccionada == "Detección de Anomalías":
    mostrar_deteccion_anomalias()
elif pagina_seleccionada == "Descripción del Sistema":
    mostrar_descripcion_sistema()