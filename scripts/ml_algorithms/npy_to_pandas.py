import os
import numpy as np
import pandas as pd
from sktime.transformations.panel.dwt import DWTTransformer

negative_folder = r"C:\Users\Lenovo\Desktop\IC\[99] Database Final (16_4296)\cookie_c"
positive_folder = r"C:\Users\Lenovo\Desktop\IC\[99] Database Final (16_4296)\cookie_d"

data = []
labels = []  # Para armazenar se é positivo ou negativo

# Carregar dados de negativos
for filename in os.listdir(negative_folder):
    if filename.endswith(".npy"):
        data_row = np.load(os.path.join(negative_folder, filename))
        data.append(data_row)
        labels.append(0)  # 0 para negativos

# Carregar dados de positivos
for filename in os.listdir(positive_folder):
    if filename.endswith(".npy"):
        data_row = np.load(os.path.join(positive_folder, filename))
        data.append(data_row)
        labels.append(1)  # 1 para positivos

# Criar DataFrame
column_names = [f"feature_{i}" for i in range(data[0].shape[0])]
df = pd.DataFrame(data, columns=column_names)
df["label"] = labels

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

num_levels = 5  #Ajustar segundo o requerido
dwt_transformer = DWTTransformer(num_levels=num_levels)
X_dwt = dwt_transformer.fit_transform(X)

print(X_dwt)