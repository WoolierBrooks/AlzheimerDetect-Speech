# Carregar as bibliotecas necessárias
from sktime.transformations.series.boxcox import BoxCoxTransformer
from sklearn.metrics import accuracy_score
from tslearn.shapelets import LearningShapelets, grabocka_params_to_shapelet_size_dict
from tslearn.utils import ts_size
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

# Criar uma instância do BoxCoxTransformer
boxcox_transformer = BoxCoxTransformer()

# Caminhos das pastas com os dados
folder_dementia = "<diretório da pasta>"
folder_control = "<diretório da pasta>"

# Lista para armazenar os dados transformados
transformed_data_dementia = []
transformed_data_control = []

# Carregar e aplicar a transformação Box-Cox aos dados de demência
for npy_file in os.listdir(folder_dementia):
    if npy_file.endswith(".npy"):
        data = np.load(os.path.join(folder_dementia, npy_file))
        transformed_data = boxcox_transformer.fit_transform(data)
        transformed_data_dementia.append(transformed_data)

# Carregar e aplicar a transformação Box-Cox aos dados de controle
for npy_file in os.listdir(folder_control):
    if npy_file.endswith(".npy"):
        data = np.load(os.path.join(folder_control, npy_file))
        transformed_data = boxcox_transformer.fit_transform(data)
        transformed_data_control.append(transformed_data)

# transformed_data_dementia e transformed_data_control contêm as séries de tempo transformadas
X = np.concatenate((transformed_data_dementia, transformed_data_control), axis=0)
print(len(transformed_data_dementia), len(transformed_data_control))
print(X.shape)
print(X)
y = np.concatenate((np.ones(len(transformed_data_dementia)), np.zeros(len(transformed_data_control))), axis=0)

# Get statistics of the dataset
n_ts, ts_sz = X.shape[:2]
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
shp_clf.fit(X, y)

# Make predictions and calculate accuracy score
pred_labels = shp_clf.predict(X)
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