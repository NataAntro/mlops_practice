import pandas as pd

# Импортируем модуль pandas

df = pd.read_csv('datasets/data.csv')

# Загружаем данные из файла 'datasets/data.csv' в объект DataFrame 

mean_age = df['Age'].mean()

# Вычисляем среднее значение столбца 'Age' 

df['Age'].fillna(mean_age, inplace=True)

# Заполняем пропущенные значения в столбце 'Age' средним значением mean_age 

df.to_csv('datasets/data.csv', index=False)

# Сохраняем измененный DataFrame df обратно в файл 'datasets/data.csv', не включая индексы строк в файл

