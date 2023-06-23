import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Импортируем модуль pandas 
# Импортируем класс OneHotEncoder 

df = pd.read_csv('datasets/data.csv')

# Загружаем данные из файла 'datasets/data.csv' 

one_hot_encoder = OneHotEncoder(sparse_output=False)
# Создаем экземпляр класса OneHotEncoder 

sex_ohe = one_hot_encoder.fit_transform(df[['Sex']])
# Применяем OneHotEncoder к столбцу 'Sex' в DataFrame df и сохраняем результат в переменную sex_ohe

sex_ohe_df = pd.DataFrame(sex_ohe, columns=one_hot_encoder.categories_[0])
# Создаем новый DataFrame sex_ohe_df из массива sex_ohe и используем значения
# из one_hot_encoder.categories_[0] в качестве названий столбцов

df = pd.concat([df, sex_ohe_df], axis=1)
# Объединяем исходный DataFrame df и DataFrame sex_ohe_df по столбцам

df.to_csv('datasets/data.csv', index=False)

# Сохраняем измененный DataFrame df обратно в файл 'datasets/data.csv', не включая индексы строк в файл

