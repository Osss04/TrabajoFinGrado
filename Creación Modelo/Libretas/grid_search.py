import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
from torch.amp import GradScaler, autocast

class TimeSeriesDataset(Dataset):
    def __init__(self, data, n, h, m, overlap = 1):
        """
        Parámetros:
            data: Serie temporal, es un DataFrame.
            n: Tamaño de la ventana, es un entero.
            h: Horizonte de predicción, es un entero.
            m: Número de predicciones futuras, es un entero.
            overlap: es el solapamiento. Como para cada segundo se requieren
            los 120 segundos anteriores, se usa el overlap para no pasar de 120 en 120. 
            Por defecto, su valor es 1.
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

class LSTMPredictor(nn.Module):
        def __init__(self, input_size, lstm_neurons, dense_neurons, activation, output_size):
            super(LSTMPredictor, self).__init__()
    
            self.lstm_layers = nn.ModuleList()
            self.dense_layers = nn.ModuleList()
    
            activations = {
                'relu': nn.ReLU(),
                'sigmoid': nn.Sigmoid()
            }
            self.activation = activations.get(activation, nn.ReLU())  # ReLU por defecto
    
            #para agregar las capas LSTM
            for i in range(len(lstm_neurons)):
                input_dim = input_size if i == 0 else lstm_neurons[i-1]
                self.lstm_layers.append(nn.LSTM(input_dim, lstm_neurons[i], batch_first=True))
    
            #para agregar las capas densas
            for i in range(len(dense_neurons)):
                input_dim = lstm_neurons[-1] if i == 0 else dense_neurons[i-1]
                self.dense_layers.append(nn.Linear(input_dim, dense_neurons[i]))
    
            #para generar la capa de salida
            self.output_layer = nn.Linear(dense_neurons[-1], output_size)

            #guardar la media, la desviación estándar y los errores en validación
            self.register_buffer("train_mean", torch.tensor(0.0))
            self.register_buffer("train_std", torch.tensor(1.0))
            self.register_buffer("val_errors", torch.tensor([]))
    
        def forward(self, x):
            #pasa por las capas LSTM
            for lstm in self.lstm_layers:
                x, _ = lstm(x)
            #se toma el último estado
            x = x[:, -1, :]
    
            #pasa por las capas densas
            for dense in self.dense_layers:
                x = self.activation(dense(x))
            #capa de salida
            x = self.output_layer(x)
            return x
        

def build_model(input_size, lstm_neurons, dense_neurons, activation, output_size):
    """
    build_model: Construye un modelo LSTM en PyTorch para regresión de series temporales.

    Parámetros:
        input_size: Número de características de entrada, es un entero.
        lstm_neurons: Lista con el número de neuronas en cada capa LSTM, es una lista.
        dense_neurons: Lista con el número de neuronas en cada capa densa, es una lista.
        activation: Función de activación de las capas densas ('relu', 'sigmoid'), es una lista.
        output_size: Número de características de salida, es un entero.

    Devuelve:
        Modelo preparado para entrenar.
    """
    return LSTMPredictor(input_size, lstm_neurons, dense_neurons, activation, output_size)


def grid_search(build_model, param_grid, train_loader, val_loader,output_size, device='cuda'):
    """
    grid_search: Función que realizar un grid search de manera manual. Entrena 3 épocas cada posible modelo en función
    de los hiperparámetros dados, para posteriormente evaluarlo. Esta función guarda cual es la mejor combinación de
    hiperparámetros y los devuelve.

    Parámetros:
    build_model: Función diseñada para construir el modelo de forma dinámica. Recibe la lista con las
    neuronas correspondientes y devuelve el modelo, es una función.
    param_grid: Diccionario que contiene los hiperparámetros a probar en la búsqueda, es un diccionario.
    train_loader: Es el DataLoader creado para cargar los datos de entrenamiento, es un DataLoader.
    val_loader: Es el DataLoader creado para cargar los datos de validación, es un DataLoader.
    device='cuda': Dispositivo dónde se entrena el modelo, por defecto es la gpu (cuda).
    output_size: Tamaño de la capa de salida, es un entero.
    
    Devuelve:
        - best_params: diccionario que contiene la mejor combinación de hiperparámetros que ha encontrado el modelo.
        - best_score: valor entero que indica el mejor score del mejor modelo.
    
    """

    #se comienza instanciando el mejor score a valor infinito,
    #ya que la métrica es el MAE y hay que quedarse con el modelo
    #que ofrezca el menor error absoluto medio
    best_score = float('inf')
    best_params = None

    #batch de los datos de train loader
    X_sample, _ = next(iter(train_loader))
    #el tamaño de la entrada de la red es la cantidad de columnas del 
    #dataset, X_sample tiene un tamaño [120, X_columns]
    input_size = X_sample.shape[2]

    #recorrer todas las configuraciones posibles del diccionario param_grid
    for params in ParameterGrid(param_grid):
        print(f"\n Probando configuración: {params}")

        #llamar a nuestra función build_model para crear el modelo
        #con los hiperparámetros de la iteración actual del param_grid
        model = build_model(
            input_size=input_size,
            lstm_neurons=params['lstm_neurons'],
            dense_neurons=params['dense_neurons'],
            activation=params['activation'],
            output_size = output_size
        ).to(device)

        #utilizar el optimizador Adam con learning rate 0.0001
        #para evitar el problea de explosión de gradiente
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        #establecer que la función de pérdida a usar
        #es el error absoluto medio (MAE)
        loss_fn = nn.L1Loss()

        print(f" Entrenando con epochs = {params['epochs']}")
        scaler = GradScaler(device="cuda")
        #realizar tantas épocas como se establezca en el diccionario de param_grid,
        for epoch in range(params['epochs']):
            model.train() #activar el modo de entrenamiento
            train_loss = 0.0 #inicializar la pérdida

            #utilizar progress_bar para tener un poco más de información durante el entrenamiento
            progress_bar = tqdm(train_loader, desc=f" Epoch {epoch + 1}/{params['epochs']}", leave=False)

            #para todos los batches durante esta época, convertir los datos a float32 para
            #que el entrenamiento sea más rápido
            
            for X_batch, y_batch in progress_bar:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                optimizer.zero_grad()
            
                with autocast(device_type='cuda'):
                    y_pred = model(X_batch)
                    y_batch = y_batch.view(y_batch.shape[0], -1)
                    loss = loss_fn(y_pred, y_batch)
            
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                #aumentar la pérdida del entrenamiento
                train_loss += loss.item()
                #actualizar la barra de progreso
                progress_bar.set_postfix(loss=f"{train_loss / len(train_loader):.4f}")

            print(f" Epoch {epoch + 1}/{params['epochs']} - Loss: {train_loss / len(train_loader):.4f}")

        model.eval() #activar el modo de evaluación
        #preparar listas vacías para guardar las predicciones y valores reales
        predictions, true_values = [], []
        print(" Evaluando en conjunto de validación...")
        #desactivar el cálculo de graidentes, para hacer predicciones
        #no necesita hacer retropropagación
        with torch.no_grad():
            #recorrer todo el dataset de validación por batches:
            for X_val, y_val in val_loader:
                
                #convertir los datos a float32 para que el entrenamiento
                #sea más rápido
                X_val, y_val = X_val.to(device).float(), y_val.to(device).float()
                #reducir en 1 la dimensión innecesaria como en el entrenamiento
                #y_val = y_val.squeeze(1)
                y_val = y_val.view(y_val.shape[0], -1)
                #obtener la predicción del modelo
                y_pred = model(X_val)
                #añadir las predicciones a las listas definidas
                #hacerlo como array de NumPy para después
                #calcular mean_absolute_error
                predictions.append(y_pred.cpu().numpy())
                true_values.append(y_val.cpu().numpy())
                
        #concatenar todas las predicciones y los valores reales
        predictions = np.concatenate(predictions, axis=0)
        true_values = np.concatenate(true_values, axis=0)

        #obtener el score del modelo utilizando la función de
        #NumPy para calcular el Error Cuadrático Medio
        score = mean_absolute_error(true_values, predictions)
        print(f" MAE en validación: {score:.4f}")

        #si el score actual es menor que el mejor score guardado,
        #lo almacena
        #actualiza también los mejores hiperparámetros
        if score < best_score:
            best_score = score
            best_params = params

        #elimina el modelo de la memoria para no tener problemas de memoria
        del model
        #libera la memoria de la GPU para evitar guardar datos no utilizados
        torch.cuda.empty_cache()

    print(f"\n Mejor configuración: {best_params}")
    print(f" Mejor MAE: {best_score:.4f}")

    #devuelve los mejores parámetros y el mejor score
    return best_params, best_score

def main():
    print("REALIZAR GRID_SEARCH")
    train = pd.read_csv("train.csv")
    validation = pd.read_csv("val.csv")
    test = pd.read_csv("test.csv")
    print(test)
    X_train = train.drop(columns = "Normal/Attack")
    y_train = train["Normal/Attack"]
    X_validation = validation.drop(columns = "Normal/Attack")
    y_validation = validation["Normal/Attack"]
    X_test = test.drop(columns = "Normal/Attack")
    y_test = test["Normal/Attack"]

    n = 120  #tamaño de la ventana
    h = 10   #horizonte de predicción
    m = 1    #número de predicciones futuras

    #creación de los datasets
    train_dataset = TimeSeriesDataset(X_train, n, h, m)
    val_dataset = TimeSeriesDataset(X_validation, n, h, m)

    #creación de los dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        num_workers=2,      #para cargar los datos más rápido
        pin_memory=torch.cuda.is_available(),     #optimiza la transferencia de los datos a la GPU
        persistent_workers=True,
        prefetch_factor=4
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        num_workers=2,      #para cargar los datos más rápido
        pin_memory=torch.cuda.is_available(),    #optimiza la transferencia de los datos a la GPU
        persistent_workers=True,
        prefetch_factor=4
    )

    param_grid1 = {
    'lstm_neurons': [[300, 200, 130], [200, 130]],
    'dense_neurons':  [[256], [256, 128]],
    'activation': ['relu'],
    'epochs': [5] #5 épocas para realizar el grid_search
    }
    
    run_grid = True

    torch.backends.cudnn.benchmark = True
    
    if run_grid:
        params1, score1 = grid_search(build_model, param_grid1, train_loader, val_loader, X_train.shape[1])
    

if __name__ == "__main__":
    main()