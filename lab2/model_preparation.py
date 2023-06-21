import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import pickle

# загружаем обработанные данные
train = pd.read_csv('/media/psf/Home/lab222/train/boston_data_train_scaled.csv')

# выделяем признаки и целевую переменную
X_train, y_train = train.drop('PRICE', axis=1), train['PRICE']

# инициализация и обучение модели
model = GradientBoostingRegressor(random_state=42)
model.fit(X_train, y_train)

# сохраняем обученную модель в файл
with open('/media/psf/Home/lab222/trained_model.pkl', 'wb') as file:
    pickle.dump(model, file)
