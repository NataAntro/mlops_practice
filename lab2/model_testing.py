import pandas as pd
from sklearn.metrics import mean_squared_error
import pickle

# загружаем обработанные данные
test = pd.read_csv('/media/psf/Home/lab222/test/boston_data_test_scaled.csv')

# выделяем признаки и целевую переменную
X_test, y_test = test.drop('PRICE', axis=1), test['PRICE']

# загружаем обученную модель из файла
with open('/media/psf/Home/lab222/trained_model.pkl', 'rb') as file:
    model = pickle.load(file)

# используем модель для предсказания на тестовых данных
predictions = model.predict(X_test)

# вычисляем среднеквадратическую ошибку
mse = mean_squared_error(y_test, predictions)

print(f'Mean Squared Error on test data: {mse}')
