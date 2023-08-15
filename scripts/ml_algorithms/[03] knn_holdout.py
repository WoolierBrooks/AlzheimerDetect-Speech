import os
import numpy as np
from tslearn.utils import to_time_series_dataset
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Carregar dados de pacientes com diagnóstico de demência positivo
folder_dementia = r"C:\Users\Lenovo\Desktop\IC\[99] Database Final (16_287)\cookie_d"
data_dementia = []
for npy_file in os.listdir(folder_dementia):
    if npy_file.endswith(".npy"):
        data = np.load(os.path.join(folder_dementia, npy_file))
        data_dementia.append(data)
data_dementia = np.array(data_dementia, dtype=object)

# Carregar dados de pacientes con diagnóstico de demência control (controle)
folder_control = r"C:\Users\Lenovo\Desktop\IC\[99] Database Final (16_287)\cookie_c/"
data_control = []
for npy_file in os.listdir(folder_control):
    if npy_file.endswith(".npy"):
        data = np.load(os.path.join(folder_control, npy_file))
        data_control.append(data)
data_control = np.array(data_control, dtype=object)

# Combinar e etiquetar os dados
X = np.concatenate((data_dementia, data_control), axis=0)
y = np.concatenate((np.ones(len(data_dementia)), np.zeros(len(data_control))), axis=0)

# Converter para o formato correto de séries temporais
X = to_time_series_dataset(X)

# Definir validação cruzada com 5 folds estratificados
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Definir as métricas a serem avaliadas
scoring_metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]

# Loop sobre as métricas
for metric in scoring_metrics:
    scores = cross_val_score(KNeighborsTimeSeriesClassifier(n_neighbors=20), X, y, cv=cv, scoring=metric)
    print(f"{metric.capitalize()} Scores: {scores}")
    print(f"Mean {metric.capitalize()}: {scores.mean()}")
    print(f"Standard Deviation {metric.capitalize()}: {scores.std()}")