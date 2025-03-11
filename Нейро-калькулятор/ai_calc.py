import numpy as np
from sklearn.neural_network import MLPRegressor

# Генерация данных
X = []
y = []

for a in range(100):
    for b in range(100):
        X.append([a, b, 0])  # 0 для сложения
        y.append(a + b)
        X.append([a, b, 1])  # 1 для вычитания
        y.append(a - b)

X = np.array(X)
y = np.array(y)

# Обучение модели
model = MLPRegressor(hidden_layer_sizes=(10,), max_iter=1000)
model.fit(X, y)

# Предсказание
def predict(a, b, operation):
    op_code = 0 if operation == '+' else 1
    return model.predict([[a, b, op_code]])[0]

print(predict(7, 3, '+'))  # Должно быть около 10
print(predict(7, 3, '-'))  # Должно быть около 4
