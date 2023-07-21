from sktime.classification.shapelet_based import ShapeletTransformClassifier
from sktime.classification.sklearn import RotationForest

X_train = [2, 1, 4]
y_train = [1, 0, 1]

X_test = [3]
y_test = [0] 

clf = ShapeletTransformClassifier(
    estimator=RotationForest(n_estimators=3),
    n_shapelet_samples=100,
    max_shapelets=10,
    batch_size=20,
) 
clf.fit(X_train, y_train) 
ShapeletTransformClassifier(...)
y_pred = clf.predict(X_test) 