import os
os.environ["KERAS_BACKEND"] = "torch"
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Forzar CPU


import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


modelo = keras.saving.load_model('predictinador3000.keras')

print("1. Cargando y preparando datos...")
df = pd.read_csv('../data/raw/data_set_1/dataset_orinoco_multivariado_final_copy.csv', parse_dates=['fecha']).sort_values('fecha').set_index('fecha')
df = df.ffill().bfill()

# === AGREGAR: Características cíclicas (día, mes, año) ===
def agregar_codificacion_ciclica(df):
    """
    Agrega codificación cíclica para capturar:
    - Día del año (estaciones: lluvia/sequía)
    - Día del mes (inicio/fin de mes)
    - Mes del año
    - Año (tendencias a largo plazo)
    """
    # Día del año (1-365) → estaciones
    dia_del_año = df.index.dayofyear
    df['dia_año_seno'] = np.sin(2 * np.pi * dia_del_año / 365)
    df['dia_año_coseno'] = np.cos(2 * np.pi * dia_del_año / 365)
    
    # Día del mes (1-31) → inicio/fin de mes
    dia_del_mes = df.index.day
    df['dia_mes_seno'] = np.sin(2 * np.pi * dia_del_mes / 31)
    df['dia_mes_coseno'] = np.cos(2 * np.pi * dia_del_mes / 31)
    
    # Mes del año (1-12)
    mes = df.index.month
    df['mes_seno'] = np.sin(2 * np.pi * mes / 12)
    df['mes_coseno'] = np.cos(2 * np.pi * mes / 12)
    
    # Año (para tendencias a largo plazo) - normalizado
    año = df.index.year
    df['año'] = (año - año.min()) / (año.max() - año.min())
    
    return df

df = agregar_codificacion_ciclica(df)
print(f"   → Columnas actuales: {list(df.columns)}")
print(f"   → Forma del dataset: {df.shape}")



# Escalamos
scaler = MinMaxScaler()
datos_escalados = scaler.fit_transform(df)

#modificacion del tiempo sen/con


# Índices de las 4 ciudades
ciudades = ['ciudad_bolivar', 'caicara', 'palua', 'ayacucho']
indice_target = [df.columns.get_loc(col) for col in ciudades]

# Función de ventanas - para TODAS las ciudades a la vez
def crear_ventanas_multi(datos_np, seq_length, target_indices):
    X, y = [], []
    for i in range(len(datos_np) - seq_length):
        X.append(datos_np[i : i + seq_length, :])
        # Extraer los 4 valores objetivo simultáneamente
        y.append(datos_np[i + seq_length, target_indices])
    return np.array(X), np.array(y)

LONGITUD = 30
VARIABLES = df.shape[1]

print("2. Creando ventanas para las 4 ciudades...")
X, y = crear_ventanas_multi(datos_escalados, LONGITUD, indice_target)
print(f"   X shape: {X.shape}, y shape: {y.shape}")

# Separar train 60%, test 20% y pruebas posteriores 20%
# Dividir en 3 partes
train_split = int(len(X) * 0.6)
val_split = int(len(X) * 0.8)

X_train = X[:train_split]      # 60%
X_val = X[train_split:val_split]  # 20%
X_test = X[val_split:]          # 20%

y_train = y[:train_split]
y_val = y[train_split:val_split]
y_test = y[val_split:]

#X_train, X_test = X[:split_index], X[split_index:]
#y_train, y_test = y[:split_index], y[split_index:]
fechas_test = df.index[LONGITUD + val_split:LONGITUD + val_split + len(X_test)]

print(f"   Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]} muestras")



# Evaluar en test set (datos nunca vistos durante entrenamiento)
print("5. Evaluando en test set...")
predicciones = modelo.predict(X_test)
print(f"   Predicciones shape: {predicciones.shape}")  # (n, 4)

# Desescalar cada ciudad
resultados = {}
for i, ciudad in enumerate(ciudades):
    idx = indice_target[i]
    
    # Valores reales
    dummy_real = np.zeros((len(y_test), VARIABLES))
    dummy_real[:, idx] = y_test[:, i]
    y_real = scaler.inverse_transform(dummy_real)[:, idx]
    
    # Predicciones
    dummy_pred = np.zeros((len(predicciones), VARIABLES))
    dummy_pred[:, idx] = predicciones[:, i]
    y_pred = scaler.inverse_transform(dummy_pred)[:, idx]
    
    resultados[ciudad] = {
        'real': y_real,
        'pred': y_pred,
        'fechas': fechas_test
    }


# --- 6. Filtrado por fechas, métricas y gráficos ---
print("\n6. Generando métricas y gráficos...")

# Parámetros para establecer fechas de prueba (puedes cambiar estas fechas)
# Por defecto usamos todo el rango disponible en test
fecha_inicio_prueba = fechas_test[0].strftime('%Y-%m-%d')
fecha_fin_prueba = fechas_test[-1].strftime('%Y-%m-%d')

print(f"Rango total disponible en test: {fecha_inicio_prueba} a {fecha_fin_prueba}")

# Puedes descomentar y modificar estas líneas para probar un rango de fechas específico:
fecha_inicio_prueba = '2024-11-30'
fecha_fin_prueba = '2024-12-31'
print(f"Evaluando el periodo: {fecha_inicio_prueba} a {fecha_fin_prueba}")

for ciudad in ciudades:
    # Crear un DataFrame con los resultados para facilitar el filtrado
    df_res = pd.DataFrame({
        'real': resultados[ciudad]['real'],
        'pred': resultados[ciudad]['pred']
    }, index=resultados[ciudad]['fechas'])
    
    # Filtrar por el rango de fechas establecido
    df_filtrado = df_res.loc[fecha_inicio_prueba:fecha_fin_prueba]
    
    if df_filtrado.empty:
        print(f"\nAdvertencia: No hay datos para {ciudad} en el rango de fechas especificado.")
        continue

    # Calcular métricas e indicar rango de éxito
    errores = np.abs(df_filtrado['real'] - df_filtrado['pred'])
    mae = np.mean(errores)
    rmse = np.sqrt(np.mean(errores**2))
    
    # Definimos el "rango de éxito" como el porcentaje de predicciones con error <= 0.5m y <= 1.0m
    exito_05m = np.mean(errores <= 0.5) * 100
    exito_10m = np.mean(errores <= 1.0) * 100

    print(f"\n=== Resultados para {ciudad.upper()} ({len(df_filtrado)} días) ===")
    print(f"  - MAE (Error promedio absoluto): {mae:.2f} m")
    print(f"  - RMSE: {rmse:.2f} m")
    print(f"  - RANGO DE ÉXITO (error <= 0.5m): {exito_05m:.1f}% de las predicciones")
    print(f"  - RANGO DE ÉXITO (error <= 1.0m): {exito_10m:.1f}% de las predicciones")

    # Generar gráfico para el rango seleccionado
    plt.figure(figsize=(15, 6))
    plt.plot(df_filtrado.index, df_filtrado['real'], label=f'Nivel Real ({ciudad})', color='blue')
    plt.plot(df_filtrado.index, df_filtrado['pred'], label=f'Predicción Transformer', color='red', linestyle='--')
    plt.title(f'Predicción del Río Orinoco - {ciudad.upper()}\nPeriodo: {fecha_inicio_prueba} a {fecha_fin_prueba} | Éxito(<=0.5m): {exito_05m:.1f}%')
    plt.xlabel('Fecha')
    plt.ylabel('Nivel (m)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'prediccion_orinoco_{ciudad}.png')
    plt.close()
    print(f"   → Gráfico guardado: prediccion_orinoco_{ciudad}.png")

#
#print("7. Guardando modelo...")
#modelo.save('modelo_orinoco_multi.keras')
#print("   → Modelo guardado: modelo_orinoco_multi.keras")
#
print("\n=== Entrenamiento completado ===")
print("Archivos generados:")
for ciudad in ciudades:
    print(f"  - prediccion_orinoco_{ciudad}.png")
print("  - modelo_orinoco_multi.keras")