import pandas as pd
from calendar import monthrange
import os

input_file = "ElNiño.csv"
output_file = "ElNiño_diario.csv"

print(f"Procesando {input_file}...")

# Leer los datos separados por ;
df = pd.read_csv(input_file, sep=";", encoding="latin1")
df = df.dropna(subset=[df.columns[0]])

daily_data = []

# Iterar sobre las filas
for index, row in df.iterrows():
    year = int(row.iloc[0])
    # Iterar sobre los 12 meses
    for month in range(1, 13):
        value = row.iloc[month]
        # Determinar cuántos días tiene el mes en ese año
        _, num_days = monthrange(year, month)
        for day in range(1, num_days + 1):
            date_str = f"{year}-{month:02d}-{day:02d}"
            daily_data.append({"fecha": date_str, "valor_del_niño": value})

df_daily = pd.DataFrame(daily_data)
df_daily.to_csv(output_file, index=False)
print(f"¡Listo! Se ha creado el archivo {output_file} con {len(df_daily)} registros diarios.")
