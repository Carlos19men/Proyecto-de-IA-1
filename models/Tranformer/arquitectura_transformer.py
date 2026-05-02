#aqui construimos el cerebro y lo guardamos en Keras_Backend para que PyTorch lo ejecute
import os
os.environ["KERAS_BACKEND"] = "torch"
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Forzar CPU

import keras
from keras import layers
import numpy as np

# 1. Función para la Matemática de Posición (Positional Encoding)
def obtener_codificacion_posicional(longitud_secuencia, d_model):
    """
    Genera las ondas de seno y coseno para inyectar el tiempo en los datos.
    """
    posiciones = np.arange(longitud_secuencia)[:, np.newaxis]
    divisores = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

    pe = np.zeros((longitud_secuencia, d_model))
    pe[:, 0::2] = np.sin(posiciones * divisores)
    pe[:, 1::2] = np.cos(posiciones * divisores)

    # Keras 3 usa keras.ops para ser compatible con PyTorch, TF y JAX
    return keras.ops.cast(pe, dtype="float32")

# 2. El Bloque del Transformer (Encoder)
def bloque_transformer(inputs, dimension_cabezal, num_cabezales, dimension_densa, dropout=0.1):
    """
    Construye una capa de Multi-Head Attention seguida de redes densas.
    """
    # Capa de Normalización: Estabiliza el entrenamiento matemático
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)

    # Capa de Atención: "Qué día del pasado importa más"
    atencion_salida = layers.MultiHeadAttention(
        key_dim=dimension_cabezal, num_heads=num_cabezales, dropout=dropout
    )(x, x) # Q=x, K=x, V=x (Auto-atención)

    x = layers.Dropout(dropout)(atencion_salida)

    # Conexión Residual (Suma la entrada original con la salida de atención)
    res = x + inputs

    # Feed Forward: Red Neuronal Densa para procesar lo que descubrió la atención
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=dimension_densa, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)

    return x + res

# 3. Construcción Completa del Modelo
def construir_modelo_orinoco(longitud_secuencia, num_variables):
    # La entrada será (60 días, cantidad de variables como precipitación, nivel, etc)
    inputs = keras.Input(shape=(longitud_secuencia, num_variables))

    # Proyectamos nuestras variables iniciales a un espacio más grande (ej. 64 dimensiones)
    # para que la codificación posicional tenga espacio para sumarse matemáticamente
    x = layers.Dense(32)(inputs)

    # Generamos y sumamos la Codificación Posicional
    pe = obtener_codificacion_posicional(longitud_secuencia, 32)
    x = x + pe

    # Apilamos 2 bloques de Transformer (puedes poner más si es necesario)
    x = bloque_transformer(x, dimension_cabezal=32, num_cabezales=4, dimension_densa=32)
    x = bloque_transformer(x, dimension_cabezal=32, num_cabezales=4, dimension_densa=32)

    # Extraemos la información importante de todos los días (Global Average Pooling)
    x = layers.GlobalAveragePooling1D()(x)

    # Capa Densa final
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.1)(x)

    # Capa de Salida: 1 solo número (El caudal futuro del río en UNA ciudad)
    outputs = layers.Dense(4, activation="linear")(x)

    # Compilamos el grafo
    modelo = keras.Model(inputs=inputs, outputs=outputs, name="Transformer_Orinoco")
    return modelo

# 4. Prueba Rápida de la Arquitectura
if __name__ == "__main__":
    # Suponiendo que tu ventana es de 60 días y tienes 8 variables en tu CSV
    LONGITUD = 60
    VARIABLES = 4

    modelo = construir_modelo_orinoco(LONGITUD, VARIABLES)
    modelo.summary()