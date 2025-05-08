import streamlit as st
from PIL import Image
import os

img_path = "imagenes/swat_Esquema.png"

def mostrar_descripcion_sistema():
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

    #título
    st.markdown('<div class="title">Descripción del Sistema SWaT</div>', unsafe_allow_html=True)
    st.markdown('<div class="emoji-container">🗒️🚰</div>', unsafe_allow_html=True)

    #introducción
    with st.container():
        st.markdown("""
        <div class="info-box">
            <strong>En concreto, se analizará el comportamiento de la planta de tratamiento de agua conocida como <span style="color: #1f77b4;">SWaT (Secure Water Treatment)</span>.</strong><br><br>
            Esta planta de agua contiene seis procesos principales, y a continuación, se explican las fases del sistema:
        </div>
        """, unsafe_allow_html=True)

    #procesos
    st.markdown('<div class="header">Procesos de la Planta SWaT</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 6, 1])
    with col2:
        img = Image.open("../Imagenes/swat_Esquema.png")
        st.image(img, caption="Diagrama de los procesos principales de la planta SWaT", use_container_width=True)

    #sensores
    st.markdown('<div class="header">Ubicación de Sensores</div>', unsafe_allow_html=True)
    with st.container():
        st.markdown("""
        <div class="info-box">
            A continuación se muestra dónde se ubica cada sensor en la planta:
        </div>
        """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 6, 1])
    with col2:
        img2 = Image.open("../Imagenes/Esquema.png")
        st.image(img2, caption="Esquema detallado de la planta SWaT mostrando ubicación de sensores", use_container_width=True)

    #tabla
    st.markdown('<div class="header">Listado Completo de Sensores y Actuadores</div>', unsafe_allow_html=True)
    with st.container():
        st.markdown("""
        <div class="info-box">
            También se muestran todos los sensores y actuadores del sistema con sus respectivas descripciones:
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


mostrar_descripcion_sistema()