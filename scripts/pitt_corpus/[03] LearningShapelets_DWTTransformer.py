import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tslearn.shapelets import LearningShapelets, grabocka_params_to_shapelet_size_dict
from tslearn.utils import ts_size
from sktime.transformations.panel.dwt import DWTTransformer

# Carregar seus dados
folder_dementia = "<diretório da pasta>"
folder_control = "<diretório da pasta>"

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
print(len(data_dementia), len(data_control))
y = np.concatenate((np.ones(len(data_dementia)), np.zeros(len(data_control)-2)), axis=0)

num_levels = 5  #Ajustar segundo o requerido
dwt_transformer = DWTTransformer(num_levels=num_levels)
X_dwt = dwt_transformer.fit_transform(X)
#Aqui o x_dwt está diminuindo as amostras de 549 a 547

# Get statistics of the dataset
n_ts, ts_sz = X_dwt.shape[:2]
n_classes = len(set(y))

# Set the number of shapelets per size as done in the original paper
shapelet_sizes = grabocka_params_to_shapelet_size_dict(n_ts=n_ts,
                                                       ts_sz=ts_sz,
                                                       n_classes=n_classes,
                                                       l=0.01, #0.8
                                                       r=1)

# Define the model using parameters provided by the authors (except that we
# use fewer iterations here)
shp_clf = LearningShapelets(n_shapelets_per_size=shapelet_sizes,
                            optimizer=tf.optimizers.Adam(.01),
                            batch_size=549,
                            weight_regularizer=.01,
                            max_iter=300,
                            random_state=1,
                            verbose=0)
shp_clf.fit(X_dwt, y)

# Make predictions and calculate accuracy score
pred_labels = shp_clf.predict(X_dwt)
print("Correct classification rate:", accuracy_score(y, pred_labels))

# Plot the different discovered shapelets
plt.figure()
for i, sz in enumerate(shapelet_sizes.keys()):
    plt.subplot(len(shapelet_sizes), 1, i + 1)
    plt.title("%d shapelets of size %d" % (shapelet_sizes[sz], sz))
    for shp in shp_clf.shapelets_:
        if ts_size(shp) == sz:
            plt.plot(shp.ravel())
    plt.xlim([0, max(shapelet_sizes.keys()) - 1])

plt.tight_layout()
plt.show()

# The loss history is accessible via the `model_` that is a keras model
plt.figure()
plt.plot(np.arange(1, shp_clf.n_iter_ + 1), shp_clf.history_["loss"])
plt.title("Evolution of cross-entropy loss during training")
plt.xlabel("Epochs")
plt.show()

# Resultados
# 0.5594149908592322