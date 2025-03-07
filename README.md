# TrabajoFinGrado
Este repositorio contiene todos los archivos para la realización del Trabajo de Fin de Grado:

## Técnicas Avanzadas de Aprendizaje Automáticopara la Detección de Intrusos en Sistemas Ciber-Físicos

---

# Ejecución de las Libretas

Esta sección describe los pasos necesarios para ejecutar las libretas contenidas en este repositorio.

## Estructura del Proyecto
Los archivos relacionados con el modelo se encuentran en el directorio `Libretas`. 

Para ejecutar la libreta **`Creación del modelo.ipynb`**, primero es necesario preprocesar los datos del dataset SWaT siguiendo los pasos indicados en la libreta **`Preprocesamiento de los datos.ipynb`**.

---

## 📌 Ejecución de la libreta `Preprocesamiento de los datos.ipynb`

### 1️⃣ Instalar dependencias
Se requiere Python `3.10.16` y las siguientes librerías:
```bash
pip install pandas matplotlib scikit-learn seaborn numpy scipy
```

### 2️⃣ Descargar y cargar los datos
Obtener los siguientes archivos del dataset SWaT y configurar su ruta en la libreta:
- [SWaT_Dataset_Normal.xlsx](https://pruebasaluuclm-my.sharepoint.com/:x:/r/personal/oscar_alcarria_alu_uclm_es/Documents/Archivos%20TFG/Datos%20Originales%20SWaT/SWaT_Dataset_Normal.xlsx?d=w9a72d4f689c246538b404ae29ee1f5a5&csf=1&web=1&e=7afD8A)
- [SWaT_Dataset_Attack_v0](https://pruebasaluuclm-my.sharepoint.com/:x:/r/personal/oscar_alcarria_alu_uclm_es/Documents/Archivos%20TFG/Datos%20Originales%20SWaT/SWaT_Dataset_Attack_v0.xlsx?d=w48b3d7413b314499985f8ed7bf5c8be4&csf=1&web=1&e=eMIMfm)


### 3️⃣ Ejecutar la libreta
Ejecutar las celdas en el orden establecido. Al finalizar, se generarán tres archivos `.csv` correspondientes a los datos preprocesados:
- `train.csv` (entrenamiento)
- `validation.csv` (validación)
- `test.csv` (pruebas)

Estos archivos serán utilizados en la siguiente libreta.

---

## 📌 Ejecución de la libreta `Creación del modelo.ipynb`

### 1️⃣ Instalar dependencias
Se requiere Python `3.10.16` y las siguientes librerías:
```bash
pip install pandas torch numpy scikit-learn tqdm seaborn matplotlib
```

### 2️⃣ Cargar los datos preprocesados
Configurar la ruta de los archivos generados en la libreta de preprocesamiento:
- `train.csv`
- `validation.csv`
- `test.csv`

### 3️⃣ Verificar GPU
Si se dispone de una GPU, asegurarse de que está activada:
```python
import torch
torch.cuda.is_available()  # Debe devolver True
```
*Ejemplo de GPU utilizada: NVIDIA GeForce RTX 3080 Ti.*

### 4️⃣ Ejecutar la libreta
Ejecutar las celdas en el orden indicado. Al finalizar:
- Se guardarán los modelos entrenados en archivos `.pth`.
- Se evaluará el modelo con los datos de prueba.
- Se generará el archivo `salida_modelo.csv` con las predicciones del modelo.

*Como referencia, con la NVIDIA GeForce RTX 3080 Ti el tiempo de entrenamiento para 5 epochs es de unas 30 horas.*

---



https://pruebasaluuclm-my.sharepoint.com/:f:/r/personal/oscar_alcarria_alu_uclm_es/Documents/Archivos%20TFG/Datos%20Originales%20SWaT?csf=1&web=1&e=ZtQXjF