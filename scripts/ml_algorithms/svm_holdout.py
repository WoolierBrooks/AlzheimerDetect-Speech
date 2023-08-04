import os
import numpy as np
from tslearn.utils import to_time_series_dataset
from tslearn.svm import TimeSeriesSVC
from sklearn.model_selection import train_test_split

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
X_train = np.concatenate((data_dementia, data_control), axis=0)
y_train = np.concatenate((np.ones(len(data_dementia)), np.zeros(len(data_control))), axis=0)

# Dividir os dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Converter para o formato correto de séries temporais
X_train = to_time_series_dataset(X_train)
X_test = to_time_series_dataset(X_test)

# Treinar o modelo SVM
clf = TimeSeriesSVC(C=1.0, kernel="gak")
clf.fit(X_train, y_train)

# Fazer a previsão com o modelo treinado
y_pred = clf.predict(X_test)

# Avaliar a acurácia do modelo
accuracy = (y_pred == y_test).mean()
print("Acurácia:", accuracy)
