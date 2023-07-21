import numpy as np
from sktime.classification.compose import RocketClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Dados de exemplo
X = np.random.rand(100, 10)  # 100 amostras com 10 dimensões
y = np.random.randint(0, 2, size=100)  # Rótulos binários

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar o classificador Rocket
clf = RocketClassifier()

# Treinar o classificador
clf.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = clf.predict(X_test)

# Calcular a acurácia do classificador
accuracy = accuracy_score(y_test, y_pred)
print("Acurácia:", accuracy)