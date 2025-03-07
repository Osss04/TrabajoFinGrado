# RESULTADOS GRID SEARCH

## param1:
----------------------------------------------
### Probando configuración: 
`{'activation': 'relu', 'dense_neurons': [100, 704], 'epochs': 3, 'lstm_neurons': [512 ,256, 130]}`
Entrenando con epochs = 3

- **Epoch 1/3** - Loss: 0.0753
- **Epoch 2/3** - Loss: 0.0892
- **Epoch 3/3** - Loss: 0.0930

Evaluando en conjunto de validación...

- **MAE en validación**: 0.3470

### Probando configuración: 
`{'activation': 'relu', 'dense_neurons': [100, 704], 'epochs': 3, 'lstm_neurons': [256, 130]}`
Entrenando con epochs = 3

- **Epoch 1/3** - Loss: 0.0745
- **Epoch 2/3** - Loss: 0.0757
- **Epoch 3/3** - Loss: 0.0925

Evaluando en conjunto de validación...

- **MAE en validación**: 0.3298

### Probando configuración: 
`{'activation': 'relu', 'dense_neurons': [704], 'epochs': 3, 'lstm_neurons': [512 ,256, 130]}`
Entrenando con epochs = 3

- **Epoch 1/3** - Loss: 0.0696
- **Epoch 2/3** - Loss: 0.0726
- **Epoch 3/3** - Loss: 0.0684

Evaluando en conjunto de validación...

- **MAE en validación**: 0.3090

### Probando configuración: 
`{'activation': 'relu', 'dense_neurons': [704], 'epochs': 3, 'lstm_neurons': [256, 130]}`
Entrenando con epochs = 3

- **Epoch 1/3** - Loss: 0.0690
- **Epoch 2/3** - Loss: 0.0653
- **Epoch 3/3** - Loss: 0.0637

Evaluando en conjunto de validación...

- **MAE en validación**: 0.3067

### Mejor configuración: 
`{'activation': 'relu', 'dense_neurons': [704], 'epochs': 3, 'lstm_neurons': [256, 130]}`
- **Mejor MAE**: 0.3067

----------------------------------------------
## param2:
----------------------------------------------