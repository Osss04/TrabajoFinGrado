# RESULTADOS GRID SEARCH 📊

## param_grid1:

### Probando configuración:  
**`{'activation': 'relu', 'dense_neurons': [704], 'epochs': 3, 'lstm_neurons': [512, 256, 130]}`**  
**Entrenando con epochs = 3**  

- **Epoch 1/3** - Loss: 0.0157  
- **Epoch 2/3** - Loss: 0.0145  
- **Epoch 3/3** - Loss: 0.0111  

**Evaluando en conjunto de validación...**  
📉 **MAE en validación: 0.0920**  

---

### Probando configuración:  
**`{'activation': 'relu', 'dense_neurons': [704], 'epochs': 3, 'lstm_neurons': [256, 130]}`**  
**Entrenando con epochs = 3**  

- **Epoch 1/3** - Loss: 0.0103  
- **Epoch 2/3** - Loss: 0.0087  
- **Epoch 3/3** - Loss: 0.0072  

**Evaluando en conjunto de validación...**  
📉 **MAE en validación: 0.0577**  

---

### Probando configuración:  
**`{'activation': 'relu', 'dense_neurons': [704, 200], 'epochs': 3, 'lstm_neurons': [512, 256, 130]}`**  
**Entrenando con epochs = 3**  

- **Epoch 1/3** - Loss: 0.0222  
- **Epoch 2/3** - Loss: 0.0250  
- **Epoch 3/3** - Loss: 0.0158  

**Evaluando en conjunto de validación...**  
📉 **MAE en validación: 0.0648**  

---

### Probando configuración:  
**`{'activation': 'relu', 'dense_neurons': [704, 200], 'epochs': 3, 'lstm_neurons': [256, 130]}`**  
**Entrenando con epochs = 3**  

- **Epoch 1/3** - Loss: 0.0171  
- **Epoch 2/3** - Loss: 0.0130  
- **Epoch 3/3** - Loss: 0.0131  

**Evaluando en conjunto de validación...**  
📉 **MAE en validación: 0.0916**  

---

## param_grid2:

### Probando configuración:  
**`{'activation': 'sigmoid', 'dense_neurons': [704], 'epochs': 3, 'lstm_neurons': [512, 256, 130]}`**  
**Entrenando con epochs = 3**  

- **Epoch 1/3** - Loss: 0.0199  
- **Epoch 2/3** - Loss: 0.0165  
- **Epoch 3/3** - Loss: 0.0120  

**Evaluando en conjunto de validación...**  
📉 **MAE en validación: 0.0914**  

---

### Probando configuración:  
**`{'activation': 'sigmoid', 'dense_neurons': [704], 'epochs': 3, 'lstm_neurons': [256, 130]}`**  
**Entrenando con epochs = 3**  

- **Epoch 1/3** - Loss: 0.0183
- **Epoch 2/3** - Loss: 0.0136  
- **Epoch 3/3** - Loss: 0.0094  

**Evaluando en conjunto de validación...**  
📉 **MAE en validación: 0.0913**  

---

### Probando configuración:  
**`{'activation': 'sigmoid', 'dense_neurons': [704, 200], 'epochs': 3, 'lstm_neurons': [512, 256, 130]}`**  
**Entrenando con epochs = 3**  

- **Epoch 1/3** - Loss: 0.0175  
- **Epoch 2/3** - Loss: 0.0196  
- **Epoch 3/3** - Loss: 0.0180  

**Evaluando en conjunto de validación...**  
📉 **MAE en validación: 0.1953**  

---

### Probando configuración:  
**`{'activation': 'sigmoid', 'dense_neurons': [704, 200], 'epochs': 3, 'lstm_neurons': [256, 130]}`**  
**Entrenando con epochs = 3**  

- **Epoch 1/3** - Loss: 0.0182  
- **Epoch 2/3** - Loss: 0.0172  
- **Epoch 3/3** - Loss: 0.0183  

**Evaluando en conjunto de validación...**  
📉 **MAE en validación: 0.1691**  




### ✅ Mejor configuración general:

| Parámetro        | Valor             |
|-----------------|------------------|
| **Activación**  | `relu`           |
| **Neuronas densas** | `[704]`     |
| **Neuronas LSTM** | `[256, 130]` |
| **Mejor MAE**   | `0.0577`        |


<hr style="height:4px;border-width:0;color:gray;background-color:gray">

# Resultados del Entrenamiento


## 📈 Progreso del Entrenamiento

| Época   | Loss     | Checkpoint               |
|---------|----------|--------------------------|
| 1/9     | `0.0102` |                          |
| 2/9     | `0.0083` |                          |
| 3/9     | `0.0068` |                          |
| 4/9     | `0.0064` |                          |
| 5/9     | `0.0061` | ✅ `model_5.pt`          |
| 6/9     | `0.0062` | ✅ `model_6.pt`          |
| 7/9     | `0.0065` | ✅ `model_7.pt`          |
| 8/9     | `0.0058` | ✅ `model_8.pt`          |
| 9/9     | `0.0072` | ✅ `model_9.pt`          |
| 10/10   | `0.0054` | ✅ `model_10.pt`         |

### 🔍 Análisis
- **Mejor pérdida**: Época 10 (`0.0054`)
- **Modelos guardados**: 6 (épocas 5, 6, 7, 8, 9, 10)



<hr style="height:4px;border-width:0;color:gray;background-color:gray">

# Resultados del Modelo de Detección de Anomalías

## ⚡ Progreso de la evaluación:
100%|████████████████████████████████████████████████████| 449789/449789 [1:32:16<00:00, 81.25it/s]

## 📊 Métricas de Rendimiento

| Métrica     | Valor   |
|-------------|---------|
| Precision   | 0.7726  |
| Recall      | 0.7208  |
| F1-Score    | 0.7458  |
| Accuracy    | 0.9403  |