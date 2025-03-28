# RESULTADOS GRID SEARCH

# RESULTADOS GRID SEARCH

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

### ✅ Mejor configuración:  
```json
{'activation': 'relu', 'dense_neurons': [704], 'epochs': 3, 'lstm_neurons': [256, 130]}