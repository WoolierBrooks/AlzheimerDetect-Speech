import os
import numpy as np
from tslearn.utils import to_time_series_dataset
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

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
scoring_metrics = {
    "accuracy": make_scorer(accuracy_score),
    "precision": make_scorer(precision_score),
    "recall": make_scorer(recall_score),
    "f1": make_scorer(f1_score),
    "roc_auc": make_scorer(roc_auc_score)
}

# Realizar validação cruzada com todas as métricas em uma única chamada
results = cross_validate(KNeighborsTimeSeriesClassifier(n_neighbors=10), X, y, cv=cv, scoring=scoring_metrics)

# Exibir os resultados para cada métrica
for metric, scores in results.items():
    print(f"{metric.capitalize()} Scores: {scores}")
    print(f"Mean {metric.capitalize()}: {np.mean(scores)}")
    print(f"Standard Deviation {metric.capitalize()}: {np.std(scores)}")