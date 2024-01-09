import os
import numpy as np
import pandas as pd

negative_folder = r'C:\caminho\para\sua\pasta\Control'
positive_folder = r'C:\caminho\para\sua\pasta\Dementia'

data = []
labels = []  # Para almacenar si es positivo o negativo

# Función para llenar con NaN hasta 4296 columnas
def pad_with_nan(data_row):
    if len(data_row) < 4296:
        return np.pad(data_row, (0, 4296 - len(data_row)), 'constant', constant_values=(np.nan,))
    else:
        return data_row[:4296]

# Cargar datos de negativos
for filename in os.listdir(negative_folder):
    if filename.endswith(".npy"):
        data_row = np.load(os.path.join(negative_folder, filename))
        data_row = pad_with_nan(data_row)  # Llenar con NaN si es necesario
        data.append(data_row)
        labels.append(0)  # 0 para negativos

# Cargar datos de positivos
for filename in os.listdir(positive_folder):
    if filename.endswith(".npy"):
        data_row = np.load(os.path.join(positive_folder, filename))
        data_row = pad_with_nan(data_row)  # Llenar con NaN si es necesario
        data.append(data_row)
        labels.append(1)  # 1 para positivos

# Crear DataFrame
column_names = [f"feature_{i}" for i in range(4296)]  # Asegúrate de tener 4296 columnas
df = pd.DataFrame(data, columns=column_names)
df["label"] = labels

ruta_actual = os.path.dirname(os.path.abspath(__file__))

# Combina la ruta actual con el nombre del archivo CSV
nombre_archivo_csv = "dataframe.csv"
ruta_csv = os.path.join(ruta_actual, nombre_archivo_csv)

# Guarda el DataFrame como un archivo CSV en la misma carpeta
df.to_csv(ruta_csv, index=False)