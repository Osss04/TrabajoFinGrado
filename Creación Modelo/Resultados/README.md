# RESULTADOS GRID SEARCH 📊

## param_grid:


### Probando configuración:
**`{'activation': 'relu', 'dense_neurons': [256], 'epochs': 5, 'lstm_neurons': [300, 200, 130]}`** 

**Entrenando con epochs = 5** 
                                                                    
- **Epoch 1/5** - Loss: 0.0165                                                                   
- **Epoch 2/5** - Loss: 0.0145                                                                   
- **Epoch 3/5** - Loss: 0.0091                                                                   
- **Epoch 4/5** - Loss: 0.0077                                                                   
- **Epoch 5/5** - Loss: 0.0079

**Evaluando en conjunto de validación...**  
 📉 **MAE en validación: 0.0397**

---

### Probando configuración:
**`{'activation': 'relu', 'dense_neurons': [256], 'epochs': 5, 'lstm_neurons': [200, 130]}`** 

**Entrenando con epochs = 5** 
                                                                   
- **Epoch 1/5** - Loss: 0.0121                                                                 
- **Epoch 2/5** - Loss: 0.0106                                                                 
- **Epoch 3/5** - Loss: 0.0086                                                                
- **Epoch 4/5** - Loss: 0.0076                                                               
- **Epoch 5/5** - Loss: 0.0068

**Evaluando en conjunto de validación...**  
 📉 **MAE en validación: 0.0251**

---

### Probando configuración:  
**`{'activation': 'relu', 'dense_neurons': [256, 128], 'epochs': 5, 'lstm_neurons': [300, 200, 130]}`**  

**Entrenando con epochs = 5**  

- **Epoch 1/5** - Loss: 0.0193  
- **Epoch 2/5** - Loss: 0.0200  
- **Epoch 3/5** - Loss: 0.0176  
- **Epoch 4/5** - Loss: 0.0150  
- **Epoch 5/5** - Loss: 0.0133  

**Evaluando en conjunto de validación...**  
📉 **MAE en validación: 0.0572**  

---

### Probando configuración:
**`{'activation': 'relu', 'dense_neurons': [256, 128], 'epochs': 5, 'lstm_neurons': [200, 130]}`** 

**Entrenando con epochs = 5** 
                                                                   
 - **Epoch 1/5** - Loss: 0.0181                                         
 - **Epoch 2/5** - Loss: 0.0157                                         
 - **Epoch 3/5** - Loss: 0.0113                                                      
 - **Epoch 4/5** - Loss: 0.0092                                                                 
 - **Epoch 5/5** - Loss: 0.0084

**Evaluando en conjunto de validación...**  
 📉 **MAE en validación: 0.0521**

---

### ✅ Mejor configuración general:

| Parámetro        | Valor             |
|-----------------|------------------|
| **Activación**  | `relu`           |
| **Neuronas densas** | `[256]`     |
| **Neuronas LSTM** | `[200, 130]` |
| **Mejor MAE**   | `0.0251`        |

<hr style="height:4px;border-width:0;color:gray;background-color:gray">

# Resultados del Entrenamiento


## 📈 Progreso del Entrenamiento

| Época | Loss     | Checkpoint             |
|-------|----------|------------------------|
| 1/6   | `0.0104` | ✅ `model_epoch_1.pt`   |
| 2/6   | `0.0090` | ✅ `model_epoch_2.pt`   |
| 3/6   | `0.0087` | ✅ `model_epoch_3.pt`   |
| 4/6   | `0.0084` | ✅ `model_epoch_4.pt`   |
| 5/6   | `0.0077` | ✅ `model_epoch_5.pt`   |
| 6/6   | `0.0070` | 🏆 **`model_epoch_6.pt`** |

### 🔍 Análisis
- **Mejor pérdida**: Época 6 (`0.0070`)
- **Checkpoints guardados**: 6 (épocas 1, 2, 3, 4, 5, 6)




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