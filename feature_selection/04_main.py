from data_preprocessing import load_and_process_data
from feature_selection import univariate_feature_selection, recursive_feature_elimination, select_from_model, sequential_feature_selection
from model_training import train_and_evaluate_model

# Load and preprocess data
X_train, X_test, y_train, y_test = load_and_process_data("adress_20_db_egemaps_train.csv", "adress_20_db_egemaps_test.csv")

# Feature Selection
X_train_new, X_test_new = univariate_feature_selection(X_train, y_train, X_test)

# Train and evaluate model (e.g., using LDA)
accuracy, mean_accuracy = train_and_evaluate_model(X_train_new, X_test_new, y_train, y_test, model_type='lda')

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Mean Accuracy (Cross-validation): {mean_accuracy * 100:.2f}%")