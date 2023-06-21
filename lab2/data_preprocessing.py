import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

# загружаем данные
train = pd.read_csv('/media/psf/Home/lab222/train/boston_data_train.csv')
test = pd.read_csv('/media/psf/Home/lab222/test/boston_data_test.csv')

# инициализируем объект StandardScaler
scaler = StandardScaler()

# выделяем признаки и целевую переменную
X_train, y_train = train.drop('PRICE', axis=1), train['PRICE'].values.reshape(-1, 1)
X_test, y_test = test.drop('PRICE', axis=1), test['PRICE'].values.reshape(-1, 1)

# обучаем трансформер на тренировочных данных и преобразуем данные
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# сохраняем обработанные данные
pd.DataFrame(np.hstack((X_train, y_train)), columns=train.columns).to_csv('/media/psf/Home/lab222/train/boston_data_train_scaled.csv', index=False)
pd.DataFrame(np.hstack((X_test, y_test)), columns=test.columns).to_csv('/media/psf/Home/lab222/test/boston_data_test_scaled.csv', index=False)
