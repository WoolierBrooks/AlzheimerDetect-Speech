import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tslearn.preprocessing import TimeSeriesScalerMinMax
from tslearn.shapelets import LearningShapelets, grabocka_params_to_shapelet_size_dict
from tslearn.utils import ts_size

# Carregar seus dados
folder_dementia = r"C:\Users\Lenovo\Desktop\IC\[99] Database Final (16_287)\cookie_d"
folder_control = r"C:\Users\Lenovo\Desktop\IC\[99] Database Final (16_287)\cookie_c/"

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
y = np.concatenate((np.ones(len(data_dementia)), np.zeros(len(data_control))), axis=0)

# Normalize each of the timeseries
X = TimeSeriesScalerMinMax().fit_transform(X)

# Get statistics of the dataset
n_ts, ts_sz = X.shape[:2]
n_classes = len(set(y))

# Set the number of shapelets per size as done in the original paper
shapelet_sizes = grabocka_params_to_shapelet_size_dict(n_ts=n_ts,
                                                       ts_sz=ts_sz,
                                                       n_classes=n_classes,
                                                       l=0.8,
                                                       r=1)

# Define the model using parameters provided by the authors (except that we
# use fewer iterations here)
shp_clf = LearningShapelets(n_shapelets_per_size=shapelet_sizes,
                            optimizer=tf.optimizers.Adam(.01),
                            batch_size=549,
                            weight_regularizer=.01,
                            max_iter=600,
                            random_state=42,
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