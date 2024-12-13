import os
import chardet
import re
import gzip
import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import confusion_matrix, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

def load_texts_from_directory(directory, label=None):
    texts = []
    filenames = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith(".cha"):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'rb') as file:
                raw_data = file.read()
                result = chardet.detect(raw_data)
                text = raw_data.decode(result['encoding'])
                par_text = extract_par_text(text)
                texts.append(par_text)
                filenames.append(filename)
                if label is not None:
                    labels.append(label)
    return texts, filenames, labels

def extract_par_text(text):
    par_lines = re.findall(r'\*PAR:\t(.*?)(?=\n\S)', text, re.DOTALL)
    concatenated_text = " ".join(par_lines)
    cleaned_text = re.sub(r'\\d+_\d+\', '', concatenated_text)
    return cleaned_text.strip()

define_directories = lambda: ("train_negative_dir", "train_positive_dir")
negative_dir, positive_dir = define_directories()

x_neg, filenames_neg, y_neg = load_texts_from_directory(negative_dir, 0)
x_pos, filenames_pos, y_pos = load_texts_from_directory(positive_dir, 1)

x_train = x_neg + x_pos
filenames_train = filenames_neg + filenames_pos
y_train = y_neg + y_pos

def ncd(x, x2):
    x_compressed = len(gzip.compress(x.encode()))
    x2_compressed = len(gzip.compress(x2.encode()))
    xx2 = len(gzip.compress((" ".join([x, x2])).encode()))
    return (xx2 - min(x_compressed, x2_compressed)) / max(x_compressed, x2_compressed)

train_ncd = np.array([[ncd(x_train[i], x_train[j]) for j in range(len(x_train))] for i in range(len(x_train))])

def knn_classification(train_ncd, y_train, k_values):
    loo = LeaveOneOut()
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        accuracies = []
        tp_total, fp_total, fn_total, tn_total = 0, 0, 0, 0

        for train_index, test_index in loo.split(train_ncd):
            X_train, X_test = train_ncd[train_index], train_ncd[test_index]
            y_train_fold, y_test_fold = np.array(y_train)[train_index], np.array(y_train)[test_index]

            knn.fit(X_train, y_train_fold)
            y_pred = knn.predict(X_test)

            accuracy = knn.score(X_test, y_test_fold)
            accuracies.append(accuracy)

            cm = confusion_matrix(y_test_fold, y_pred)
            tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

            tp_total += tp
            fp_total += fp
            fn_total += fn
            tn_total += tn

        mean_accuracy = np.mean(accuracies)
        precision_alzheimer = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
        recall_control = tn_total / (tn_total + fp_total) if (tn_total + fp_total) > 0 else 0

        print(f'Mean Leave-One-Out CV Accuracy for k={k}: {mean_accuracy}')
        print(f'Precision (Alzheimer - Class 1) for k={k}: {precision_alzheimer}')
        print(f'Recall (Control - Class 0) for k={k}: {recall_control}')

def knn_regression(train_ncd, y_train, k_values):
    loo = LeaveOneOut()
    for k in k_values:
        knn = KNeighborsRegressor(n_neighbors=k)
        y_true_all, y_pred_all = [], []

        for train_index, test_index in loo.split(train_ncd):
            X_train_fold, X_test_fold = train_ncd[train_index], train_ncd[test_index]
            y_train_fold, y_test_fold = np.array(y_train)[train_index], np.array(y_train)[test_index]

            knn.fit(X_train_fold, y_train_fold)
            y_pred = knn.predict(X_test_fold)

            y_true_all.extend(y_test_fold)
            y_pred_all.extend(y_pred)

        mean_rmse = np.sqrt(mean_squared_error(y_true_all, y_pred_all))
        mean_mae = mean_absolute_error(y_true_all, y_pred_all)
        corr, _ = pearsonr(y_true_all, y_pred_all) if len(y_true_all) > 1 else (float('nan'), None)

        print(f'Mean Leave-One-Out CV RMSE for k={k}: {mean_rmse}')
        print(f'Mean Leave-One-Out CV MAE for k={k}: {mean_mae}')
        print(f'Mean Leave-One-Out CV Correlation for k={k}: {corr}')

k_values = [1, 3, 5]
knn_classification(train_ncd, y_train, k_values)
knn_regression(train_ncd, y_train, k_values)
