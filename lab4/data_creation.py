import pandas as pd
from catboost.datasets import titanic

# Импортируем модуль pandas 
# Импортируем функцию titanic из модуля catboost.datasets

train_df, _ = titanic()

# Загружаем данные обучающего набора из функции titanic() 

train_df.to_csv('datasets/data.csv', index=False)

# Сохраняем DataFrame train_df в файл 'datasets/data.csv', не включая индексы строк в файл

