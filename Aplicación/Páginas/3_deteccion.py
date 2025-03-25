import streamlit as st
import pandas as pd
import time
import torch
import torch.nn as nn

import os
print(os.getcwd())  # Muestra el directorio actual
 

 
# Cargar el modelo previamente entrenado
modelo = torch.load("Modelo/modelo_5.pth", weights_only=False)

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

mostrar_deteccion_anomalias()