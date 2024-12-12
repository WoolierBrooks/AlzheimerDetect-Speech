from sklearn.feature_selection import SelectKBest, f_classif, RFE, SelectFromModel, SequentialFeatureSelector
from sklearn.svm import SVC

def univariate_feature_selection(X_train, y_train, X_test, k=26):
    selector = SelectKBest(f_classif, k=k)
    X_train_new = selector.fit_transform(X_train, y_train)
    X_test_new = selector.transform(X_test)
    return X_train_new, X_test_new

def recursive_feature_elimination(X_train, y_train, X_test, k=26):
    estimator = SVC(kernel="linear")
    rfe = RFE(estimator, n_features_to_select=k, step=1)
    rfe.fit(X_train, y_train)
    X_train_new = rfe.transform(X_train)
    X_test_new = rfe.transform(X_test)
    return X_train_new, X_test_new

def select_from_model(X_train, y_train, X_test, k=26):
    model = SVC(kernel="linear")
    selector = SelectFromModel(estimator=model, max_features=k)
    selector.fit(X_train, y_train)
    X_train_new = selector.transform(X_train)
    X_test_new = selector.transform(X_test)
    return X_train_new, X_test_new

def sequential_feature_selection(X_train, y_train, X_test, k=26):
    model = SVC(kernel="linear")
    sfs = SequentialFeatureSelector(model, n_features_to_select=k, direction='forward')
    sfs.fit(X_train, y_train)
    X_train_new = sfs.transform(X_train)
    X_test_new = sfs.transform(X_test)
    return X_train_new, X_test_new