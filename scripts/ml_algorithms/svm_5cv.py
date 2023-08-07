import os
import numpy as np
from tslearn.utils import to_time_series_dataset
from tslearn.svm import TimeSeriesSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
print("Carregando os dados")
# Carregar dados de pacientes com diagnóstico de demência positivo
folder_dementia = r"C:\Users\Lenovo\Desktop\IC\[99] Database Final (1)\cookie_d"
data_dementia = []
for npy_file in os.listdir(folder_dementia):
    if npy_file.endswith(".npy"):
        data = np.load(os.path.join(folder_dementia, npy_file))
        data_dementia.append(data)
data_dementia = np.array(data_dementia, dtype=object)

# Carregar dados de pacientes con diagnóstico de demência control (controle)
folder_control = r"C:\Users\Lenovo\Desktop\IC\[99] Database Final (1)\cookie_c/"
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

# Criar o modelo SVM
clf = TimeSeriesSVC(C=1.0, kernel="gak")

# Definir as métricas a serem calculadas
scoring_metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
print("Iniciando a validação cruzada")
# Realizar a validação cruzada com 5 folds para cada métrica
for metric in scoring_metrics:
    scores = cross_val_score(clf, X, y, cv=5, scoring=metric)
    print(f"{metric.capitalize()} Scores: {scores}")
    print(f"Mean {metric.capitalize()}: {scores.mean()}")
    print(f"Standard Deviation {metric.capitalize()}: {scores.std()}")

#@ 1
#Accuracy Scores: [0.55454545 0.62727273 0.54545455 0.48181818 0.52293578]
#Mean Accuracy: 0.5464053377814846
#Standard Deviation Accuracy: 0.04760046440237219

#Precision Scores: [0.57647059 0.64705882 0.56790123 0.52941176 0.56521739]
#Mean Precision: 0.5772119604685674
#Standard Deviation Precision: 0.038456603941097595

#Recall Scores: [0.79032258 0.72131148 0.75409836 0.59016393 0.63934426]
#Mean Recall: 0.6990481226864093
#Standard Deviation Recall: 0.07386139675004359

#F1 Scores: [0.66666667 0.68217054 0.64788732 0.55813953 0.6       ]
#Mean F1: 0.6309728136259418
#Standard Deviation F1: 0.04570636921958017

#Roc_auc Scores: [0.56922043 0.66008699 0.55035129 0.43860823 0.53756831]
#Mean Roc_auc: 0.5511670479931793
#Standard Deviation Roc_auc: 0.070773661955492