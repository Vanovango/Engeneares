from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l1, l2, l1_l2
# from tensorflow.python.keras.layers import BatchNormalization
import matplotlib.pyplot as plt

#Загружаем данные
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

#Нормализация данных

# Среднее значение для обучающего набора данных
mean = x_train.mean(axis=0)
# Стандартное отклонение для обучающего набора данных
std = x_train.std(axis=0)
x_train -= mean
x_train /= std
x_test -= mean
x_test /= std

# Создание модели
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(x_train.shape[1],)))
# model.add(BatchNormalization())
# model.add(Dense(64, activation='relu', input_shape=(x_train.shape[1],), kernel_regularizer=l2(0.01)))
# model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(1)) # если допускаются отрицательные значения
model.add(Dense(1, activation='relu'))


# Компиляция и обучение
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

history = model.fit(x_train, 
                    y_train, 
                    epochs=200, 
                    validation_split=0.1, 
                    verbose=2)

#Оценка модели на тестовом наборе данных
scores = model.evaluate(x_test, y_test, verbose=1)

print('-----------------------------------------------------------------')
print('Критерий оценки')
print("Средняя абсолютная ошибка на тестовых данных:", round(scores[1], 4))
print('-----------------------------------------------------------------')
#Визуализация качества обучения

plt.plot(history.history['mae'], # mean_absolute_error
         label='Средняя абсолютная ошибка на обучающем наборе')
plt.plot(history.history['val_mae'], # val_mean_absolute_error
         label='Средняя абсолютная ошибка на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Средняя абсолютная ошибка')
plt.legend()
plt.show()

#Предсказание для тестовой выборки

pred = model.predict(x_test).flatten()
test_index=25
print("Предсказанная стоимость:", pred[test_index], ", правильная стоимость:", y_test[test_index])

#Визуализация результатов предсказаний
plt.scatter(y_test, pred)
plt.xlabel('Правильные значение, $1K')
plt.ylabel('Предсказания, $1K')
plt.axis('equal')
plt.axis('square')
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
plt.plot([-100, 100], [-100, 100], color='red')
plt.show()