# RESULTADOS GRID SEARCH 📊

---

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

---


### ✅ Mejor configuración general:

| Parámetro        | Valor             |
|-----------------|------------------|
| **Activación**  | `relu`           |
| **Neuronas densas** | `[704]`     |
| **Neuronas LSTM** | `[256, 130]` |
| **Mejor MAE**   | `0.0577`        |