import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import os

# создаем синтетические данные
X, y = np.random.rand(1000, 10), np.random.randint(2, size=1000)
data = np.hstack((X, y.reshape(-1, 1)))
df = pd.DataFrame(data)

# разбиваем данные на обучающую и тестовую выборки
train, test = train_test_split(df, test_size=0.2)

#создаём папки для данных
os.makedirs("train", exist_ok=True)
os.makedirs("test", exist_ok=True)

# сохраняем данные в папки
train.to_csv('train/data.csv', index=False)
test.to_csv('test/data.csv', index=False)
