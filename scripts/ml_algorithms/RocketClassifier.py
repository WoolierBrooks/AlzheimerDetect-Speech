import numpy as np
from sktime.classification.kernel_based import RocketClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sktime.datatypes._panel._convert import from_2d_array_to_nested

# Dados de exemplo
X = np.random.rand(100, 10)  # 100 amostras com 10 dimensões
y = np.random.randint(0, 2, size=100)  # Rótulos binários

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_norm = X_train.copy()
norm_data = X_norm.apply(lambda x: (x-x.min())/(x.max()-x.min()), axis=1)
X_norm = norm_data.values
X_nested = from_2d_array_to_nested(X_norm)

# Criar o classificador Rocket
clf = RocketClassifier()

# Treinar o classificador
clf.fit(X_nested, y_train)


# Fazer previsões no conjunto de teste
y_pred = clf.predict(X_test_nested)

# Calcular a acurácia do classificador
accuracy = accuracy_score(y_test, y_pred)
print("Acurácia:", accuracy)