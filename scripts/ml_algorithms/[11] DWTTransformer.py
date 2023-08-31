import os
import numpy as np
from sktime.transformations.panel.dwt import DWTTransformer

# Carregar seus dados
folder_dementia = r"C:\Users\Lenovo\Desktop\IC\[99] Database Final (16_4296)\cookie_d"
folder_control = r"C:\Users\Lenovo\Desktop\IC\[99] Database Final (16_4296)\cookie_c/"

data_dementia = []
for npy_file in os.listdir(folder_dementia):
    if npy_file.endswith(".npy"):
        data = np.load(os.path.join(folder_dementia, npy_file))
        data_dementia.append(data)

data_control = []
for npy_file in os.listdir(folder_control):
    if npy_file.endswith(".npy"):
        data = np.load(os.path.join(folder_control, npy_file))
        data_control.append(data)

X = np.concatenate((data_dementia, data_control), axis=0)
y = np.concatenate((np.ones(len(data_dementia)), np.zeros(len(data_control))), axis=0)

num_levels = 5  #Ajustar segundo o requerido
dwt_transformer = DWTTransformer(num_levels=num_levels)
X_dwt = dwt_transformer.fit_transform(X)