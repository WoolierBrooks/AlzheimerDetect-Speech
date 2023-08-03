import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sktime.classification.kernel_based import RocketClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sktime.datatypes._panel._convert import from_2d_array_to_nested

# Carregar dados de pacientes com diagnóstico de demência positivo
folder_dementia = r"C:\Users\Lenovo\Desktop\IC\[99] Database Final (160)\cookie_d"
data_dementia = []
for npy_file in os.listdir(folder_dementia):
    if npy_file.endswith(".npy"):
        data = np.load(os.path.join(folder_dementia, npy_file))
        data_dementia.append(data)
data_dementia = np.array(data_dementia, dtype=object)

# Carregar dados de pacientes con diagnóstico de demência control (controle)
folder_control = r"C:\Users\Lenovo\Desktop\IC\[99] Database Final (160)\cookie_c/"
data_control = []
for npy_file in os.listdir(folder_control):
    if npy_file.endswith(".npy"):
        data = np.load(os.path.join(folder_control, npy_file))
        data_control.append(data)
data_control = np.array(data_control, dtype=object)

# Combinar e etiquetar os dados
X = np.concatenate((data_dementia, data_control), axis=0)
y = np.concatenate((np.ones(len(data_dementia)), np.zeros(len(data_control))), axis=0)

# Dividir os dados em conjuntos de treinamento e prova
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Realizar padding e normalização ao mesmo tempo
max_vector_size = max(len(vector) for vector in X)
equalized_matrix = []

for vector in X:
    # Passo 1: Normalizar o vetor
    normalized_vector = normalize(vector[:, np.newaxis], axis=0).ravel()
    
    # Passo 2: Preencher os vetores menores com valores padrão (zero) até atingirem o tamanho máximo.
    diff = max_vector_size - len(normalized_vector)
    equalized_vector = np.concatenate((normalized_vector, np.zeros(diff)))
    equalized_matrix.append(equalized_vector)

# Resultado como um array 2D
X_normalized = np.array(equalized_matrix)

# Converter o array 2D en dados no formato nested (panel)
X_train_nested = from_2d_array_to_nested(X_normalized)

# Ajustar os rótulos `y_train` para ter o mesmo tamanho que `X_train_nested`
y_train_adjusted = y_train[:X_train_nested.shape[0]]    

# Crear el clasificador Rocket 
clf = RocketClassifier()
print(1)
# Entrenar el clasificador
clf.fit(X_train_nested, y_train_adjusted)
print(2)
# Normalizar los datos de prueba y convertir a formato nested
X_test_norm = normalize(X_test, axis=1)
X_test_nested = from_2d_array_to_nested(X_test_norm)

# Hacer predicciones en el conjunto de prueba
y_pred = clf.predict(X_test_nested)

# Calcular la precisión del clasificador
accuracy = accuracy_score(y_test, y_pred)
print("Precisión:", accuracy)