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



def train_model(build_model, best_grid, train_loader, val_loader, output_size, device='cpu'):
    """
    train_model: Función que entrena el modelo de detección de anomalías en Pytorch y que guarda dicho modelo.

    Parámetros:
    build_model: Función diseñada para construir el modelo de forma dinámica. Recibe la lista con las neuronas
    correspondientes y devuelve el modelo, es una función.
    best_grid: Diccionario que contiene la mejor combinación de hiperparámetros, es un diccionario.
    train_loader: Es el DataLoader creado para cargar los datos de entrenamiento, es un DataLoader.
    val_loader: Es el DataLoader creado para cargar los datos de validación, es un DataLoader.
    output_size: Es el tamaño de la capa de salida, es un entero.
    device='cpu': Dispositivo dónde se entrena el modelo, por defecto es la cpu.
    
    Devuelve:
        Entrena el modelo y guarda checkpoints en las épocas 2,4,5,6 y el modelo final.
    """
    #obtener un batch de los datos de train loader
    X_sample, _ = next(iter(train_loader))
    #establecer el tamaño de la entrada de la red a la cantidad
    #de columnas del dataset, al igual que en la función
    #del grid_search
    input_size = X_sample.shape[2]

    #crear el modelo con la función build_model y el diccionario
    #de best_params
    model = build_model(
            input_size=input_size,
            lstm_neurons=best_grid['lstm_neurons'],
            dense_neurons=best_grid['dense_neurons'],
            activation=best_grid['activation'],
            output_size=output_size
    ).to(device)

    #utilizar el optimizador Adam con learning rate 0.0001
    #para evitar el problea de explosión de gradiente
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    #establecer que la función de pérdida a usar
    #es el error absoluto medio (MAE)
    loss_fn = nn.L1Loss()
    scaler = GradScaler(device='cuda')
    
    #realizar tantas épocas como indique el diccionario best_grid
    for epoch in range(best_grid['epochs']):
        #activar el modo de entrenamiento
        model.train()
        train_loss = 0.0 #inicializar la pérdida
        train_errors = [] #lista para errores del train
        #progress_bar para tener un pco más de información durante el entrenamiento
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{best_grid['epochs']}",
                            leave=False, mininterval=10)

        #para todos los batches durante esta época, convertir los datos a float32 para
        #que el entrenamiento sea más rápido
        for X_batch, y_batch in progress_bar:
            X_batch = X_batch.to(device).float()
            y_batch = y_batch.to(device).float()
            #resetear los gradientes para realizar el entrenamiento
            #en la epoch actual
            optimizer.zero_grad()
            #obtener las predicciones del modelo
            with autocast(device_type='cuda'):
                y_pred = model(X_batch)
                y_batch = y_batch.view(y_batch.shape[0], -1)
                loss = loss_fn(y_pred, y_batch)
            
            #usar el scaler para backpropagation
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            #aumentar la pérdida del entrenamiento
            train_loss += loss.item()

            #actualizar la barra de progreso
            #progress_bar.set_postfix(loss=f"{train_loss / len(train_loader):.4f}")
        print(f"Epoch {epoch + 1}/{best_grid['epochs']} - Train Loss: {train_loss / len(train_loader):.4f}")
        
        #para guardar resultados
        if epoch + 1 <= 10:
            train_errors, val_errors = evalua_y_guarda_results(model, train_loader, val_loader, device)
            # Guardar estadísticas de entrenamiento y el modelo final
            model.train_mean = torch.tensor(np.mean(train_errors)).to(device)
            model.train_std = torch.tensor(np.std(train_errors)).to(device)
            model.val_errors = torch.tensor(val_errors).to(device)
            model_path = f"model_epoch_{epoch + 1}.pt"
            scripted_model = torch.jit.script(model)
            torch.jit.save(scripted_model, model_path)
            print(f"Modelo guardado en: {model_path}")

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


def evalua_y_guarda_results(model, train_loader, val_loader, device = "cpu"):
    """
    evalua_y_guarda_results: Calcula los errores en la última epoch tanto para train como para validación.

    Parámetros
        model: Modelo sobre el que se quieren guardar los datos. Es un modelo de torch.
        train_loader: Es el DataLoader creado para cargar los datos de entrenamiento, es un DataLoader.
        val_loader: Es el DataLoader creado para cargar los datos de validación, es un DataLoader.
        device='cpu': Dispositivo dónde se entrena el modelo, por defecto es la cpu.
        
    
    Devuelve:
        -train_errors: Lista de errores del train, es un array de NumPy.
        -val_errors_tensor: Lista de errores de validación, es un tensor de torch.
    """
    model.eval()
    train_errors = []
    print("Calculando los errores en el entrenamiento...")
    with torch.no_grad():
        for X_batch, y_batch in tqdm(train_loader, desc="Calculando errores en entrenamiento", leave= False):
            X_batch = X_batch.to(device).float()
            y_batch = y_batch.to(device).float()
            y_pred = model(X_batch)
            error = computa_error(y_batch, y_pred)
            train_errors.append(error)
    
    train_errors = np.concatenate(train_errors, axis = 0)
    print("Calculando los errores en validación...")
    val_errors = []
    with torch.no_grad():
        for X_batch, y_batch in tqdm(val_loader, desc="Calculando errores en validación", leave= False):
            X_batch = X_batch.to(device).float()
            y_batch = y_batch.to(device).float()
            y_pred = model(X_batch)
            error = computa_error(y_batch, y_pred)
            error_tensor = torch.tensor(error).to(device)  #convertir el error a tensor de PyTorch
            val_errors.append(error_tensor)

    #concatenar todos los errores de validación
    val_errors_tensor = torch.cat(val_errors, dim=0)

    return train_errors, val_errors_tensor




def main():
    print("REALIZAR ENTRENAMIENTO")
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

    #creación del dataset para el conjunto de test
    test_dataset = TimeSeriesDataset(X_test, n, h, m)

    #dataloader para los datos de test
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        num_workers=2,      #para cargar los datos más rápido
        pin_memory=True     #optimiza la transferencia de los datos a la GPU
    )

    best_grid = {
    'lstm_neurons': [200, 130],
    'dense_neurons': [256],
    'activation': 'relu',
    'epochs': 10 #10 epochs de entrenamiento para el modelo
    }
    train_model(build_model, best_grid, train_loader, val_loader, X_train.shape[1], device="cuda")


if __name__ == "__main__":
    main()