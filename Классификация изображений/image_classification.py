import tensorflow as tf
from keras import datasets, layers, models

# Загрузка данных
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Нормализация данных
train_images, test_images = train_images / 255.0, test_images / 255.0

# Создание модели
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)
])

# Компиляция и обучение
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=2, validation_data=(test_images, test_labels))

# Оценка точности
test_loss, test_acc = model.evaluate(test_images, test_labels)

print(f"Точность на тестовых данных: {test_acc * 100} %")
