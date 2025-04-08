import streamlit as st
import pandas as pd
import time
import torch
import numpy as np
from torch.utils.data import Dataset
from datetime import datetime
from PIL import Image 

 ########################################################
 #GENERADOR DE DATOS
 ########################################################
class TimeSeriesDataset(Dataset):
    def __init__(self, data, n, h, m, overlap = 1):
        """
        Parámetros:
            data: Serie temporal, es un DataFrame.
            n: Tamaño de la ventana, es un entero.
            h: Horizonte de predicción, es un entero.
            m: Número de predicciones futuras, es un entero.
        """

        #convertir a float32 para que el entrenamiento sea más rápido
        if isinstance(data, np.ndarray):
            self.data = torch.tensor(data, dtype=torch.float32)
        else:
            self.data = torch.tensor(data.values, dtype=torch.float32)

        self.n = n
        self.h = h
        self.m = m
        self.overlap = overlap
        
        #cantidad de muestras posibles del dataset
        self.num_samples = (len(self.data) - (n + h + m) + 1)// self.overlap

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        getitem: Devuelve la muestra correspondiente al índice dado.

        Devuelve:
            - x: ventana de tamaño n.
            - y: un valor futuro después de un horizonte h (tamaño 1).
        """
        real_idx = idx*self.overlap
        #ventana de entrada de tamaño n, con lo que se entrena
        x = self.data[real_idx:real_idx+self.n]

        #se quiere predecir el valor[n+m+h], por lo que se compara con el real
        y = self.data[real_idx+self.n+self.h:real_idx+self.n+self.h+self.m].reshape(-1)

        return x, y
    

 ########################################################
 #EVALUACION DEL MODELO
 ########################################################
def computa_error(y_true, y_pred):
    """
    computa_error: Calcula el error absoluto entre las predicciones y los valores verdaderos.

    Parámetros:
    y_true: Vector que contien los valores reales del dataset, es un array de NumPy.
    y_pred: Vector que contiene las predicciones realiadas por la red neuronal, es un array de NumPy.
    
    Devuelve:
        -error absoluto entre el vector de predicciones y el de valores reales.
    """
    error = torch.abs(y_true - y_pred).detach().cpu().numpy() #error por variable
    return error

def evaluate_model(model_path, test_loader, y_test_true, X_test, device='cpu'):
    """
    evaluate_model: Obtiene los errores con los datos de entrenamiento, calcula el umbral con los datos
    de validación y obtiene los resultados del modelo con los datos del test.
    
    Parámetros:
    model_path: ruta para cargar el modelo a evualuar. , es una string.  
    test_loader: Es el DataLoader que hemos creado para cargar los datos de test, es un DataLoader.
    y_test_true: Vector que contiene los valores de la etiqueta clase de los datos de test, es un vector de NumPy.
    
    """
    #cargar el modelo guardado
    model = torch.jit.load(model_path, map_location=device)
    print(f"Modelo cargado desde {model_path}")
    
    #activar el modo de evaluación
    model.eval()
    
    #se lee la media la s.d. guardados en el modelo
    train_mean = model.train_mean.cpu().numpy()
    train_std = model.train_std.cpu().numpy()
    val_errors = model.val_errors.cpu().numpy()


    #calcular los z-score en el conjunto de validación
    val_z_scores = (val_errors - train_mean) / train_std

    #definir el umbral para detectar anomalías: el máximo
    feature_thresholds = np.max(val_z_scores, axis=0)#umbral por variable

    #definir array vacío para los errores en el test
    test_errors = []
    test_predictions = []
    y_test_real = []

    #crear un DataFrame vacío para almacenar los resultados
    df_results = pd.DataFrame(columns=['📆 Fecha', '📈 Predicción', '🚨 Anomalía', '⚠️ Variables Anómalas'])

    columns = ['📆 Fecha Anomalía', '⚠️ Variables Anómalas', '📏 Desviación respecto al umbral']
    anomaly_history_df = pd.DataFrame(columns=columns)

    #crear un contenedor en Streamlit para la tabla
    table_placeholder = st.empty()

    #inicializar el marcador de posición para la imagen
    image_placeholder = st.empty()

    anomaly_message_placeholder = st.empty()  #placeholder para el mensaje

    #contenedor en Streamlit para la tabla
    table_placeholder_anomaly = st.empty()

    #desactivar el cálculo de gradientes
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device).float()
            y_batch = y_batch.to(device).float()
            y_pred = model(X_batch)
            #calcular el error en el test con la función computa_error
            error = computa_error(y_batch, y_pred)
            test_errors.append(error)
            test_predictions.append(y_pred.cpu().numpy()) #guardar predicciones
            y_test_real.append(y_batch.cpu().numpy())
            time.sleep(1) #para la simulacion

            #mostrar en Streamlit la predicción actualizada
            #detectar anomalías
            test_z_score = (error - train_mean) / train_std
            anomalies_per_feature = test_z_score > feature_thresholds
            #crear una lista para almacenar los nombres de los sensores anómalos
            anomaly_list = []

            #iterar sobre cada fila (cada predicción)
            for row in anomalies_per_feature:
                #obtener los índices de las columnas donde hay anomalías (True)
                anomaly_indices = np.where(row)[0]  # Esto devolverá todos los índices de características anómalas
                #mapear los índices de las anomalías a los nombres de las características (sensores)
                anomaly_list.append([X_test.columns[idx] for idx in anomaly_indices])  #convertir índices en nombres de sensores

            anomaly_flag = int(anomalies_per_feature.any())
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            #elegir imagen a mostrar en tiempo real
            state_image_path = elige_imagen(anomaly_list)
            #mostrar imagen del estado del sistema
            estado_sistema(state_image_path, image_placeholder) 

            #agregar nueva predicción a la tabla con nombres de sensores
            new_row = pd.DataFrame({
                '📆 Fecha': [current_time],
                '📈 Predicción': [np.round(y_pred.cpu().numpy(),4).tolist()],  #convertir a lista para evitar errores de formato
                '🚨 Anomalía': ["Sí" if anomaly_flag == 1 else "No"],  #convertir 1 -> "Sí" y 0 -> "No"
                '⚠️ Variables Anómalas': anomaly_list  #convertir índices a nombres de sensores
            })

            df_results = pd.concat([df_results, new_row], ignore_index=True)
            df_results['📈 Predicción'] = df_results['📈 Predicción'].apply( #para aproximar la predicción a dos decimales
            lambda x: np.round(np.array(x, dtype=float), 2).tolist()
            )
            table_placeholder.dataframe(df_results, use_container_width=True)

            if anomaly_list[0]:
                #mostrar el mensaje solo cuando hay una anomalía
                anomaly_message_placeholder.error("¡Anomalía detectada! Revisa los sensores y el estado del sistema.")
                
                errors_for_anomalies = error[0][anomaly_indices]

                thresholds_for_anomalies = feature_thresholds[anomaly_indices]  # Usamos el umbral más alto de los sensores

                mensajes_desvio = []

                #para poner un mensaje de cuánto se desvía la anomalía del umbral de anomalías
                for sensor, error, umbral in zip(anomaly_list[0], errors_for_anomalies, thresholds_for_anomalies):
                    if umbral != 0:  #para evitar división por cero
                        desviacion = (abs(error) / umbral) * 100
                        mensaje = f'El registro "{sensor}" se desvía un {desviacion:.2f}% del umbral'
                    else:
                        mensaje = f'El registro "{sensor}" tiene umbral cero (no se puede calcular desviación)'
                    mensajes_desvio.append(mensaje)

                #añadir información a la tabla de anomalías
                new_row = {
                    '📆 Fecha Anomalía': current_time,
                    '⚠️ Variables Anómalas': anomaly_list,
                    '📏 Desviación respecto al umbral': mensajes_desvio
                }

                #convertir el diccionario a un DataFrame de pandas
                new_row_df = pd.DataFrame([new_row])

                #agregar la nueva fila al DataFrame usando pd.concat
                anomaly_history_df = pd.concat([anomaly_history_df, new_row_df], ignore_index=True)

                #mostrar el DataFrame actualizado con las anomalías
                table_placeholder_anomaly.dataframe(anomaly_history_df, use_container_width=True)
            else:
                #si no hay anomalía, podemos borrar el mensaje de alerta
                anomaly_message_placeholder.empty()
                        


    return df_results

############################################
#MOSTRAR LA IMAGEN DE ESTADO DEL SISTEMA
############################################
def estado_sistema(path_imagen, image_placeholder):
    """
    estado_sistema: Muestra una imagen que representa el estado del sistema en Streamlit.

    Parámetros:
    state_image_path: Ruta de la imagen que representa el estado actual del sistema.
    image_placeholder: Placeholder del Streamlit dónde se actualiza la imagen.
    """
    image = Image.open(path_imagen)  #abre la imagen desde la ruta
    image_placeholder.image(image, caption="Estado del Sistema", use_container_width=True)

def elige_imagen(anomaly_list):
    """
    elige_imagen: En función de la anomalía, elige la imagen correspondiente.

    Parámetros:
    anomaly_list: Lista en la que aparecen los sensores anómalos.

    Devuelve:
        state_image_path: Ruta a la imagen que toca mostrar en función de la lista de anomalías.
    """
    #si hay anomalía
    if anomaly_list:
        if 'FIT101' in anomaly_list[0]:
            state_image_path = "Imágenes/EstadoSistema/FIT101.png"  #imagen cuando 'FIT101' tiene una anomalía
        elif 'LIT101' in anomaly_list[0]:
            state_image_path = "Imágenes/EstadoSistema/lIT101.png"  #imagen cuando 'LIT101' tiene una anomalía
        elif 'DPIT301' in anomaly_list[0]:
            state_image_path = "Imágenes/EstadoSistema/DPIT301.png"  #imagen cuando 'DPIT301' tiene una anomalía
        elif 'FIT201' in anomaly_list[0]:
            state_image_path = "Imágenes/EstadoSistema/FIT201.png"  #imagen cuando 'FIT201' tiene una anomalía
        elif 'FIT601' in anomaly_list[0]:
            state_image_path = "Imágenes/EstadoSistema/FIT601.png"  #imagen cuando 'FIT601' tiene una anomalía
        elif 'LIT301' in anomaly_list[0]:
            state_image_path = "Imágenes/EstadoSistema/LIT301.png"  #imagen cuando 'LIT301' tiene una anomalía
        elif 'LIT401' in anomaly_list[0]:
            state_image_path = "Imágenes/EstadoSistema/LIT401.png"  #imagen cuando 'LIT401' tiene una anomalía
        elif any(sensor.startswith("MV101") for sensor in anomaly_list[0]):  #cualquier 'MV101'
            state_image_path = "Imágenes/EstadoSistema/MV101.png"
        elif any(sensor.startswith("MV201") for sensor in anomaly_list[0]):  #cualquier 'MV201'
            state_image_path = "Imágenes/EstadoSistema/MV201.png"
        elif any(sensor.startswith("MV301") for sensor in anomaly_list[0]):  #cualquier 'MV301'
            state_image_path = "Imágenes/EstadoSistema/MV301.png"
        elif any(sensor.startswith("MV302") for sensor in anomaly_list[0]):  #cualquier 'MV302'
            state_image_path = "Imágenes/EstadoSistema/MV302.png"
        elif any(sensor.startswith("MV303") for sensor in anomaly_list[0]):  #cualquier 'MV303'
            state_image_path = "Imágenes/EstadoSistema/MV303.png"
        elif any(sensor.startswith("MV304") for sensor in anomaly_list[0]):  #cualquier 'MV304'
            state_image_path = "Imágenes/EstadoSistema/MV304.png"
        elif any(sensor.startswith("P101") for sensor in anomaly_list[0]):  #cualquier 'P101'
            state_image_path = "Imágenes/EstadoSistema/P101.png"
        elif any(sensor.startswith("P203") for sensor in anomaly_list[0]):  #cualquier 'P203'
            state_image_path = "Imágenes/EstadoSistema/P203.png"
        elif any(sensor.startswith("P205") for sensor in anomaly_list[0]):  #cualquier 'P205'
            state_image_path = "Imágenes/EstadoSistema/P205.png"
        elif any(sensor.startswith("P302") for sensor in anomaly_list[0]):  #cualquier 'P302'
            state_image_path = "Imágenes/EstadoSistema/P302.png"
        elif any(sensor.startswith("P602") for sensor in anomaly_list[0]):  #cualquier 'P602'
            state_image_path = "Imágenes/EstadoSistema/P602.png"
        else:
            state_image_path = "Imágenes/EstadoSistema/Normal.png"  # Imagen cuando no hay anomalías
    return state_image_path

def mostrar_deteccion_anomalias():
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

    #título
    st.markdown('<div class="title">Detección de Anomalías en Tiempo Real</div>', unsafe_allow_html=True)
    st.markdown('<div class="emoji-container">🔍🕛</div>', unsafe_allow_html=True)
    
    #cargar el test
    test = pd.read_csv("Datos/test_modificado.csv")

    st.markdown('<div class="header">📊 Datos cargados</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="upload-box">
        <p style="font-size: 18px;">Los datos de los sensores ya han sido cargados.</p>
        <p style="font-size: 14px; color: #666;">Haz clic en el botón para comenzar el análisis.</p>
    </div>
    """, unsafe_allow_html=True)
    
    #para poner el botón en el centro
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        boton = st.button("🚀 Comenzar análisis", use_container_width=True)
    
    if boton:
        #vista previa de datos
        st.markdown('<div class="header">👀 Vista previa de los datos</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="data-preview">
            <p>Primeras 5 filas del dataset de los sensores:</p>
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(test.head().style.set_properties(**{'background-color': '#f8f9fa'}), use_container_width=True)
        st.success("Iniciando análisis del archivo...")
        try:
            #procesamiento en tiempo real
            st.markdown('<div class="header">⚙️ Procesamiento en tiempo real</div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="processing-box">
                <p>Analizando datos fila por fila con el modelo LSTM...</p>
            </div>
            """, unsafe_allow_html=True)
            

            X_test = test.drop(columns = "Normal/Attack")
            y_test = test["Normal/Attack"]

            #instanciar el generador de datos
            #creación de los datasets tanto para entrenamiento como para test
            test_dataset = TimeSeriesDataset(X_test, 120, 10, 1)

            #creación de los dataloaders


            #crear el dataloader para los datos de test
            test_loader = torch.utils.data.DataLoader(
                test_dataset, 
                num_workers=0,      #para cargar los datos más rápido
                pin_memory=True     #optimiza la transferencia de los datos a la GPU
            )

            y_real =y_test.values[130:]

            resultados_df = evaluate_model("Modelo/modelo_completo.pt",test_loader,y_real,X_test, device='cpu')
            
            #botón para descargar resultados
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