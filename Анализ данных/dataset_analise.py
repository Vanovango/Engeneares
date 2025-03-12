import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Загрузка данных
df = pd.read_csv('customer_churn.csv')

# 1. Вывод заголовков таблицы
print("Заголовки таблицы:")
print(df.columns.tolist())  # Вывод в виде списка

# 2. Вывод первых строк для проверки
print("Первые 5 строк данных:")
print(df.head())

# 3. Удаление дубликатов
print("\nКоличество дубликатов до удаления:", df.duplicated().sum())
df.drop_duplicates(inplace=True)
print("Количество дубликатов после удаления:", df.duplicated().sum())

# 4. Проверка пропусков
print("\nКоличество пропусков:")
print(df.isnull().sum())

# 5. Кодирование категориальных признаков
# Label Encoding для 'gender'
le = LabelEncoder()
df['gender_encoded'] = le.fit_transform(df['gender'])

# 6. Вывод результатов после обработки
print("\nДанные после обработки:")
print(df.head())
