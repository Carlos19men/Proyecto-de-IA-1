import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Reajustar X a [muestras, pasos de tiempo, características]
n_features = datos.shape[1]
X = X.reshape((X.shape[0], X.shape[1], n_features))

# Convertir los datos de Numpy a Tensores de PyTorch
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# 2. Definir el Modelo LSTM en PyTorch
class ModeloLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ModeloLSTM, self).__init__()
        # Definimos la capa LSTM (batch_first=True asegura que la dimensión 0 sea la cantidad de muestras)
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        # Capa de salida con 4 neuronas
        self.linear = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Pasar por el LSTM
        lstm_out, _ = self.lstm(x)
        # Tomar la salida del ÚLTIMO paso de tiempo (el día 365) para predecir
        ultimo_paso = lstm_out[:, -1, :]
        # Pasar a la capa final (densa)
        prediccion = self.linear(ultimo_paso)
        return prediccion

# Crear la instancia del modelo (50 neuronas ocultas, 4 salidas)
model = ModeloLSTM(input_size=n_features, hidden_size=50, output_size=4)

# Definir Optimizador (Adam) y Función de pérdida (MSE)
optimizer = optim.Adam(model.parameters())
criterion = nn.MSELoss()

# 3. Entrenar el modelo
epochs = 200
print("Iniciando entrenamiento...")
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad() # Limpiar gradientes
    
    # Forward pass (Predicción)
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    
    # Backward pass (Ajustar pesos)
    loss.backward()
    optimizer.step()
    
    # Imprimir progreso cada 50 epochs para ver que sí está haciendo algo
    if (epoch+1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Error (Loss): {loss.item():.4f}')

# 4. Comprobación con el último dato del dataset
model.eval() # Poner el modelo en modo evaluación (no entrenamiento)

# Tomamos la última secuencia generada (X[-1])
x_input = X[-1]
x_input = x_input.reshape((1, n_steps, n_features))
x_input_tensor = torch.tensor(x_input, dtype=torch.float32)

# Hacemos la predicción sin calcular gradientes (ahorra memoria)
with torch.no_grad():
    prediccion = model(x_input_tensor)

# Comparamos la predicción con el valor real
print("\nValores REALES del último día registrado (y[-1]):")
print(y[-1])
print("\nValores PREDICHOS por el modelo:")
print(prediccion.numpy()[0]) # Convertimos el tensor de vuelta a Numpy para imprimirlo
