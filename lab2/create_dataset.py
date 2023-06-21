import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

# Загрузка данных
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

# преобразуем данные в DataFrame
df = pd.DataFrame(data)
df['PRICE'] = target

# разделение данных на тренировочный и тестовый наборы
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.2, random_state=42)

#создаём папки для данных
os.makedirs("/media/psf/Home/lab222/train", exist_ok=True)
os.makedirs("/media/psf/Home/lab222/test", exist_ok=True)

# сохранение датасетов во внешние csv файлы
train.to_csv('/media/psf/Home/lab222/train/boston_data_train.csv', index=False)
test.to_csv('/media/psf/Home/lab222/test/boston_data_test.csv', index=False)
