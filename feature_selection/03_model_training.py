from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

def train_and_evaluate_model(X_train, X_test, y_train, y_test, model_type='lda'):
    if model_type == 'lda':
        model = LinearDiscriminantAnalysis()
    elif model_type == 'dt':
        model = DecisionTreeClassifier(random_state=42)
    elif model_type == 'nn':
        model = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)
    elif model_type == 'svm':
        model = SVC(kernel='linear')

    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=2)
    mean_accuracy = cv_scores.mean()
    
    return accuracy, mean_accuracy
