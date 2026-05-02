import numpy as np
#import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt


#pendiente de separa en bloques para validacion 

# 1. Preparación de los datos
datos = np.genfromtxt('..\..\data\processed\dataset_normalizado.csv', delimiter=',', dtype=None, encoding='utf-8',skip_header=1)


#se separan los prarametros a determinar de los determinantes
# Función para estructurar los datos para LSTM (X=entrada, y=salida)
def crear_datos(datos, n_steps):
    X, y = [], []
    for i in range(len(datos) - n_steps):
        # Tomamos todos los datos (características) para los n_pasos_atras
        X.append(datos[i : (i + n_steps), :])
        
        # Tomamos solo las primeras 4 columnas (niveles de los ríos) del día siguiente
        y.append(datos[i + n_steps, 0:4]) 
        
    return np.array(X), np.array(y)

# Definir pasos de tiempo (cuántos datos anteriores usar para predecir el siguiente)
n_steps = 365
X, y = crear_datos(datos, n_steps)

# Reajustar X a [muestras, pasos de tiempo, características]
n_features = datos.shape[1]
X = X.reshape((X.shape[0], X.shape[1], n_features))

# 2. Definir el Modelo LSTM
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
#numero de neuronas
model.add(Dense(4))
model.compile(optimizer='adam', loss='mse')

# 3. Entrenar el modelo
model.fit(X, y, epochs=200, verbose=0)

# 4. Comprobación con el último dato del dataset
# Tomamos la última secuencia de 365 días generada (X[-1])
x_input = X[-1]
# Le damos la forma requerida por el modelo: [1 muestra, n_steps, n_features]
x_input = x_input.reshape((1, n_steps, n_features))

# Hacemos la predicción
prediccion = model.predict(x_input)

# Comparamos la predicción con el valor real
print("Valores REALES del último día registrado (y[-1]):")
print(y[-1])
print("\nValores PREDICHOS por el modelo:")
print(prediccion[0])
