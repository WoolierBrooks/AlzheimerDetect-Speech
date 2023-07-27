import numpy as np
from sklearn.preprocessing import normalize
from sktime.classification.kernel_based import RocketClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sktime.datatypes._panel._convert import from_2d_array_to_nested

# Dados de ejemplo
X = np.random.rand(100, 10)  # 100 amostras com 10 dimensões
y = np.random.randint(0, 2, size=100)  # Rótulos binários

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar os dados de treinamento
X_norm = normalize(X_train, axis=1)

# Converter o array 2D em dados em formato nested (painel)
X_nested = from_2d_array_to_nested(X_norm)

# Criar o classificador Rocket
clf = RocketClassifier()

# Treinar o classificador
clf.fit(X_nested, y_train)

# Normalizar os dados de teste e converter para formato nested
X_test_norm = normalize(X_test, axis=1)
X_test_nested = from_2d_array_to_nested(X_test_norm)

# Fazer previsões no conjunto de teste
y_pred = clf.predict(X_test_nested)

# Calcular a acurácia do classificador
accuracy = accuracy_score(y_test, y_pred)
print("Acurácia:", accuracy)
