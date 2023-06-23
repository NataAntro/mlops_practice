import pandas as pd

# Импортируем модуль pandas 

df = pd.read_csv('datasets/data.csv')

# Загружаем данные из файла 'datasets/data.csv' в объект DataFrame 

df = df[['Pclass', 'Sex', 'Age']]

# Из исходного DataFrame df выбираем только столбцы 'Pclass', 'Sex' и 'Age' и перезаписываем df с этими столбцами

df.to_csv('datasets/data.csv', index=False)

# Сохраняем измененный DataFrame df в файл 'datasets/data.csv', не включая индексы строк в файл

