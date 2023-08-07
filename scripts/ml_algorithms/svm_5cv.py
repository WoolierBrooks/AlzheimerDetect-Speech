import os
import numpy as np
from tslearn.utils import to_time_series_dataset
from tslearn.svm import TimeSeriesSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
#import warnings
#warnings.filterwarnings("ignore", category=RuntimeWarning)

# Carregar dados de pacientes com diagnóstico de demência positivo
folder_dementia = r"C:\Users\Lenovo\Desktop\IC\[99] Database Final (1)\cookie_d"
data_dementia = []
for npy_file in os.listdir(folder_dementia):
    if npy_file.endswith(".npy"):
        data = np.load(os.path.join(folder_dementia, npy_file))
        data_dementia.append(data)
data_dementia = np.array(data_dementia, dtype=object)

# Carregar dados de pacientes com diagnóstico de demência control (controle)
folder_control = r"C:\Users\Lenovo\Desktop\IC\[99] Database Final (1)\cookie_c/"
data_control = []
for npy_file in os.listdir(folder_control):
    if npy_file.endswith(".npy"):
        data = np.load(os.path.join(folder_control, npy_file))
        data_control.append(data)
data_control = np.array(data_control, dtype=object)
print("Terminei de carregar")
# Combinar e etiquetar os dados
X_train = np.concatenate((data_dementia, data_control), axis=0)
y_train = np.concatenate((np.ones(len(data_dementia)), np.zeros(len(data_control))), axis=0)

# Converter para o formato correto de séries temporais
X_train = to_time_series_dataset(X_train)

# Verificar se há valores NaN nos dados e substituir por zero
X_train = np.nan_to_num(X_train)

# Remover amostras com valores NaN do conjunto de treinamento
valid_samples_mask = ~np.isnan(X_train).any(axis=(1, 2))
X_train = X_train[valid_samples_mask]
y_train = y_train[valid_samples_mask]

print("Vou criar o modelo")
# Criar o modelo SVM
clf = TimeSeriesSVC(C=1.0, kernel="gak")
print("Inicio do 5cv")
# 5-fold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
accuracies = []
precisions = []
recalls = []
f1_scores = []
confusion_matrices = []

# Counter to keep track of the current fold
fold_counter = 1

for train_index, test_index in cv.split(X_train, y_train):
    X_train_fold, X_test_fold = X_train[train_index], X_train[test_index]
    y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]

    # Verificar se há valores NaN nos dados e substituir por zero
    X_train_fold = np.nan_to_num(X_train_fold)
    X_test_fold = np.nan_to_num(X_test_fold)  # Tratar o conjunto de teste

    # Verificar se ainda existem valores NaN após substituição por zero
    print("Contém NaN no conjunto de treinamento:", np.isnan(X_train_fold).any())
    print("Contém NaN no conjunto de teste:", np.isnan(X_test_fold).any())
    print("Contém NaN no conjunto de treinamento:", np.isnan(X_train).any())
    print("Contém NaN no conjunto de treinamento:", np.isnan(y_train_fold).any())
    print("Contém NaN no conjunto de treinamento:", np.isnan(y_test_fold).any())
    print("Contém NaN no conjunto de treinamento:", np.isnan(y_train).any())

    print(f"Fold {fold_counter} complete\n")
    fold_counter += 1

    print(X_train_fold)
    print("#######################################################################################################################")
    print(y_train_fold)
    # Treinar o modelo SVM com o fold atual
    clf.fit(X_train_fold, y_train_fold)
    print("agora foi")
    # Fazer a previsão com o modelo treinado
    y_pred_fold = clf.predict(X_test_fold)

    # Calcular as métricas do fold atual e armazenar
    accuracy_fold = accuracy_score(y_test_fold, y_pred_fold)
    precision_fold = precision_score(y_test_fold, y_pred_fold, zero_division=1)
    recall_fold = recall_score(y_test_fold, y_pred_fold)
    f1_score_fold = f1_score(y_test_fold, y_pred_fold)
    confusion_matrix_fold = confusion_matrix(y_test_fold, y_pred_fold)

    accuracies.append(accuracy_fold)
    precisions.append(precision_fold)
    recalls.append(recall_fold)
    f1_scores.append(f1_score_fold)
    confusion_matrices.append(confusion_matrix_fold)

print("Fazendo estatísticas")
# Calcular a média e desvio padrão das métricas dos folds
mean_accuracy = np.mean(accuracies)
std_accuracy = np.std(accuracies)

mean_precision = np.mean(precisions)
std_precision = np.std(precisions)

mean_recall = np.mean(recalls)
std_recall = np.std(recalls)

mean_f1_score = np.mean(f1_scores)
std_f1_score = np.std(f1_scores)

mean_confusion_matrix = np.mean(confusion_matrices, axis=0)
std_confusion_matrix = np.std(confusion_matrices, axis=0)

print("Acurácia média:", mean_accuracy)
print("Desvio padrão da acurácia:", std_accuracy)

print("Precisão média:", mean_precision)
print("Desvio padrão da precisão:", std_precision)

print("Recall médio:", mean_recall)
print("Desvio padrão do recall:", std_recall)

print("F1-score médio:", mean_f1_score)
print("Desvio padrão do F1-score:", std_f1_score)

print("Matriz de Confusão média:")
print(mean_confusion_matrix)
print("Desvio padrão da matriz de confusão:")
print(std_confusion_matrix)