# Trabajo de Fin de Grado: Técnicas Avanzadas de Aprendizaje Automático para la Detección de Intrusos en Sistemas Ciber-Físicos 🎓🔐

---

## Información del Repositorio 📂

Este repositorio contiene todo el trabajo correspondiente a mi **Trabajo de Fin de Grado**. A continuación, se describen los directorios y archivos que conforman este repositorio:

### Estructura de Directorios:

- **📂 Aplicación**: Contiene todos los archivos necesarios para ejecutar la aplicación de detección de anomalías. Contiene a su vez los siguientes archivos:
  - **📂 Imágenes**:
    Directorio que contiene las imágenes en la aplicación.
  - **📂 Modelo**:
    Directorio que contiene el archivo `modelo_completo.pt`, que guarda toda la información del modelo entrenado.
  - **📂 Páginas**: Directorio que contiene todas las páginas que aparecen en la aplicación de `streamlit`. Se encuentran los archivos:
    - 🗒️ **`1_inicio.py`**: Archivo de python con `streamlit` de la página de inicio de la aplicación.
    - 🗒️ **`2_descripcion.py`**: Archivo de python con `streamlit` de la página de descripción del sistema.
    - 🗒️ **`3_deteccion.py`**: Archivo de python con `streamlit` de la página de detección de anomalías en tiempo real.
  - 🗒️ **`app.py`**: Archivo de python que contiene la clase en python del modelo y toda la navegación del `streamlit`, por lo que es la página principal.

- **📂 Creación Modelo**: Contiene todos los archivos relacionados con el modelo creado, y a su vez, tiene las siguientes carpetas:
  - **📂 Resultados**: Contiene los resultados del **Grid Search**, del **Entrenamiento** y de la **Evaluación** del modelo en el archivo `README.md`.
  - **📂 Libretas**: Directorio que contiene todas las libretas `ipynb` que se han creado para realizar y analizar el modelo. Se encuentran las siguientes libretas:
    - 🗒️ **`Análisis_y_Preprocesamiento_de_los_Datos.ipynb`**: Libreta que explica el **Estudio de los datos** y el **Preprocesamiento** que se ha realizado.
    - 🗒️ **`Modelado_y_Evaluacion.ipynb`**: Libreta que explica cómo se ha realizado el **Grid Search**, el **Entrenamiento** y la **Evaluación** del modelo.
    - 🗒️ **`Estudio_Salida_del_Modelo.ipynb`**: Libreta que estudia el **Tiempo de Detección del Modelo** y el **Funcionamiento del Regresor**.
    - 🗒️🐍 **`grid_search.py`**: Archivo de Python que explica el **grid_search** que se ha realizado.
---

## Dependencias ⚙️

Para las **Libretas**, se ha utilizado **`Python 3.10.16`** y las siguientes librerías:

| Librería       | Versión   |
|----------------|-----------|
| pandas         | 2.2.3     |
| matplotlib     | 3.10.0    |
| scikit-learn   | 1.6.1     |
| seaborn        | 0.13.2    |
| numpy          | 1.26.4    |
| torch (PyTorch)| 1.12.0    |
| tqdm           | 4.64.1    |

Para la **Aplicación**, se ha utilizado **`Python 3.12`** y las siguientes librerías:

| Librería       | Versión   |
|----------------|-----------|
| torch (PyTorch)| 2.2.0     |
| streamlit      | 1.37.1    |
| pandas         | 2.2.2     |
| numpy          | 1.26.4    |
| Pillow         | 10.3.0    |
| matplotlib     | 3.8.4     |


---


## Ejecución de la Aplicación 🚀  

Para ejecutar la aplicación, sigue estos pasos:  

### Paso 1: Abrir la terminal  
Una vez descargados los archivos en local, abre la terminal en el directorio donde guardaste el archivo.  

### Paso 2: Ejecutar la aplicación  
Asegúrate de que todas las dependencias estén instaladas en tu ordenador. Luego, ejecuta el siguiente comando en la terminal:  

```bash
streamlit run app.py
```

---