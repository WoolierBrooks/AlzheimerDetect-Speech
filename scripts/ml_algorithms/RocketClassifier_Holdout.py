import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sktime.classification.kernel_based import RocketClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sktime.datatypes._panel._convert import from_2d_array_to_nested

# Carregar dados de pacientes com diagnóstico de demência positivo
folder_positivo = r"C:\Users\Lenovo\Desktop\IC\[99] Database Final (160)\cookie_d" # mudar nome para demência
data_positivo = []
for npy_file in os.listdir(folder_positivo):
    if npy_file.endswith(".npy"):
        data = np.load(os.path.join(folder_positivo, npy_file))
        data_positivo.append(data)
data_positivo = np.array(data_positivo, dtype=object)

# Carregar dados de pacientes con diagnóstico de demência negativo (controle)
folder_negativo = r"C:\Users\Lenovo\Desktop\IC\[99] Database Final (160)\cookie_c/"
data_negativo = []
for npy_file in os.listdir(folder_negativo):
    if npy_file.endswith(".npy"):
        data = np.load(os.path.join(folder_negativo, npy_file))
        data_negativo.append(data)
data_negativo = np.array(data_negativo, dtype=object)

# Combinar e etiquetar os dados
X = np.concatenate((data_positivo, data_negativo), axis=0)
y = np.concatenate((np.ones(len(data_positivo)), np.zeros(len(data_negativo))), axis=0)

# Dividir os dados em conjuntos de treinamento e prova
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar os dados de treinamento
normalized_sequences = []
for sequence in X:
    sequence_array = np.array(sequence)
    normalized_sequence = normalize(sequence_array[:, np.newaxis], axis=0).ravel()
    normalized_sequences.append(normalized_sequence)
X_normalized = np.array(normalized_sequences, dtype=object)

# Converter o array 2D en dados no formato nested (panel)
X_train_nested = from_2d_array_to_nested(X_normalized)

# Crear el clasificador Rocket 
clf = RocketClassifier()

# Entrenar el clasificador
clf.fit(X_train_nested, y_train)

# Normalizar los datos de prueba y convertir a formato nested
X_test_norm = normalize(X_test, axis=1)
X_test_nested = from_2d_array_to_nested(X_test_norm)

# Hacer predicciones en el conjunto de prueba
y_pred = clf.predict(X_test_nested)

# Calcular la precisión del clasificador
accuracy = accuracy_score(y_test, y_pred)
print("Precisión:", accuracy)
