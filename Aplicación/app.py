import streamlit as st
import torch
import torch.nn as nn

# Definir el modelo LSTM
class LSTMPredictor(nn.Module):
    def __init__(self, input_size, lstm_neurons, dense_neurons, activation, output_size):
        super(LSTMPredictor, self).__init__()

        self.lstm_layers = nn.ModuleList()
        self.dense_layers = nn.ModuleList()
        self.dropout = nn.Dropout(0.1)  # dropout de 0.1 para evitar alto recall

        activations = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid()
        }
        self.activation = activations.get(activation, nn.ReLU())  # ReLU por defecto

        # Para agregar las capas LSTM
        for i in range(len(lstm_neurons)):
            input_dim = input_size if i == 0 else lstm_neurons[i-1]
            self.lstm_layers.append(nn.LSTM(input_dim, lstm_neurons[i], batch_first=True))

        # Para agregar las capas densas
        for i in range(len(dense_neurons)):
            input_dim = lstm_neurons[-1] if i == 0 else dense_neurons[i-1]
            self.dense_layers.append(nn.Linear(input_dim, dense_neurons[i]))

        # Para generar la capa de salida
        self.output_layer = nn.Linear(dense_neurons[-1], output_size)

    def forward(self, x):
        # Pasa por las capas LSTM
        for lstm in self.lstm_layers:
            x, _ = lstm(x)
            x = self.dropout(x)  # Después de cada capa LSTM hacemos dropout
        # Tomamos el último estado
        x = x[:, -1, :]

        # Pasa por las capas densas
        for dense in self.dense_layers:
            x = self.activation(dense(x))
            x = self.dropout(x)  # Después de cada capa densa hacemos dropout
        # Capa de salida
        x = self.output_layer(x)
        return x


st.set_page_config(page_title="Detección de Anomalías SWaT", layout="wide")

pg = st.navigation([
    st.Page("Páginas/1_inicio.py", title="🏠 Inicio"),
    st.Page("Páginas/2_descripcion.py", title="📖 Descripción del Sistema"),
    st.Page("Páginas/3_deteccion.py", title="🔍 Detección de Anomalías"),
])
pg.run()