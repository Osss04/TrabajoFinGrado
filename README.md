# Trabajo de Fin de Grado: Técnicas Avanzadas de Aprendizaje Automático para la Detección de Intrusos en Sistemas Ciber-Físicos 🎓🔐

---

## Información del Repositorio 📂

Este repositorio contiene todo el trabajo correspondiente a mi **Trabajo de Fin de Grado**. A continuación, se describen los directorios y archivos que conforman este repositorio:

### Estructura de Directorios:

- **Aplicación**: Contiene todos los archivos necesarios para ejecutar la aplicación de detección de anomalías. Contiene a su vez los siguientes archivos:
  - **Imágenes**:
    Directorio que contiene las imágenes en la aplicación. 🖼️
  - **Modelo**:
    Directorio que contiene el archivo `modelo_completo.pt`, que guarda toda la información del modelo entrenado. 🎯
  - `pagina_principal.py`:
    Archivo de python que contiene todo el código de `streamlit` en el que se especifican aspectos visuales de la aplicación de detección de anomalías. 🎨

- **Creación Modelo**: Contiene todos los archivos relacionados con el modelo creado, y a su vez, tiene las siguientes carpetas:
  - **Resultados**: Contiene los resultados del **Grid Search**, del **entrenamiento** y de la **evaluación** del modelo. 📈
  - **Libretas**: Directorio que contiene todas las libretas `ipynb` que se han creado para realizar el modelo. Se encuentran las siguientes libretas:
    - `Preprocesamiento de los datos.ipynb`: Libreta que explica cómo se ha realizado el **Grid Search**, el **entrenamiento** y la **evaluación** del modelo. 🧑‍💻

---

## Dependencias ⚙️

Para ejecutar la libreta `Preprocesamiento de los datos.ipynb`, se ha utilizado **Python 3.10.16** y las siguientes librerías:

- `pandas` 📊
- `matplotlib` 📉
- `sklearn` 📚
- `seaborn` 🌈
- `numpy` 🔢
- `scipy` 🧪
- `tqdm` ⏳


Para ejecutar el archivo `pagina_principal.py`, se ha utilizado **Python 3.12.7** y las siguientes librerías:

- `pandas` 📊
- `streamlit` 🌐
- `time` ⏳
- `torch` 🔥

---

## Ejecución del Modelo 🚀

Para ejecutar el modelo, sigue los pasos detallados a continuación:

### Paso 1: Descargar los Archivos Originales del Dataset SWaT 📥

Accede a los siguientes enlaces para descargar los datos:

- [SWaT_Dataset_Normal.xlsx](https://pruebasaluuclm-my.sharepoint.com/:x:/r/personal/oscar_alcarria_alu_uclm_es/Documents/Archivos%20TFG/Datos%20Originales%20SWaT/SWaT_Dataset_Normal.xlsx?d=w9a72d4f689c246538b404ae29ee1f5a5&csf=1&web=1&e=7afD8A)
- [SWaT_Dataset_Attack_v0.xlsx](https://pruebasaluuclm-my.sharepoint.com/:x:/r/personal/oscar_alcarria_alu_uclm_es/Documents/Archivos%20TFG/Datos%20Originales%20SWaT/SWaT_Dataset_Attack_v0.xlsx?d=w48b3d7413b314499985f8ed7bf5c8be4&csf=1&web=1&e=eMIMfm)

### Paso 2: Cargar los Archivos en la Libreta `Preprocesamiento de los datos.ipynb` 📤

Una vez descargados los archivos, cárgalos en la libreta para comenzar con el preprocesamiento.

### Paso 3: Ejecutar la Libreta en el Orden Establecido ▶️

Sigue el orden de ejecución dentro de la libreta para asegurar que el preprocesamiento se realice correctamente.

### Paso 4: Guardar los Archivos Preprocesados 💾

Guarda los archivos preprocesados como `train.csv`, `val.csv` y `test.csv` para cargarlos en la libreta `desarrollo_del_modelo.ipynb`.

### Paso 5: Cargar los Archivos Preprocesados en la Libreta `desarrollo_del_modelo.ipynb` 🔄

Una vez guardados los archivos preprocesados, cárgalos en la libreta destinada al desarrollo del modelo.

### Paso 6: Ejecutar la Libreta en el Orden Establecido ▶️

Ejecuta la libreta `desarrollo_del_modelo.ipynb` siguiendo el orden establecido para realizar el **Grid Search**, el **entrenamiento** y la **evaluación** del modelo.

### Paso 7: Guardar el Archivo del Modelo Entrenado 💾

Al finalizar el entrenamiento, guarda el archivo del modelo entrenado.

---

## Ejecución de la Aplicación 🚀  

Para ejecutar la aplicación, sigue estos pasos:  

### Paso 1: Abrir la terminal  
Una vez descargados los archivos en local, abre la terminal en el directorio donde guardaste el archivo.  

### Paso 2: Ejecutar la aplicación  
Asegúrate de que **Streamlit** está instalado en tu ordenador. Luego, ejecuta el siguiente comando en la terminal:  

```bash
streamlit run pagina_principal.py
```

---

## Contacto 📧

Si tienes alguna pregunta o sugerencia, no dudes en contactarme.

---
