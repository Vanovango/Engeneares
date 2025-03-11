import pandas as pd
from sklearn.preprocessing import LabelEncoder

# 1. Загрузка данных
df = pd.read_csv('customer_churn.csv')

# Вывод заголовков таблицы
print("Заголовки таблицы:")
print(df.columns.tolist())  # Вывод в виде списка

# Вывод первых строк для проверки
print("Первые 5 строк данных:")
print(df.head())

# 2. Удаление дубликатов
print("\nКоличество дубликатов до удаления:", df.duplicated().sum())
df.drop_duplicates(inplace=True)
print("Количество дубликатов после удаления:", df.duplicated().sum())

# 3. Проверка пропусков
print("\nКоличество пропусков:")
print(df.isnull().sum())
# Удаляет строки с NaN в любом столбце
df_clean = df.dropna()

# 4. Кодирование категориальных признаков
# Label Encoding для 'gender'
le = LabelEncoder()
df['gender_encoded'] = le.fit_transform(df['gender'])


# 5. Обработка выбросов в 'MonthlyCharges'
Q1 = df['MonthlyCharges'].quantile(0.25)
Q3 = df['MonthlyCharges'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Фильтрация данных
df = df[(df['MonthlyCharges'] >= lower_bound) & (df['MonthlyCharges'] <= upper_bound)]

# Вывод результатов после обработки
print("\nДанные после обработки:")
print(df.head())

print("\nСтатистика после очистки:")
print(df.describe())
