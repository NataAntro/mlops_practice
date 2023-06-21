import pandas as pd
import pickle
from sklearn.metrics import accuracy_score

# загружаем данные
test = pd.read_csv('test/data_scaled.csv')

# загружаем модель
with open('model.pkl', 'rb') as f:
    clf = pickle.load(f)

# выделяем признаки и целевую переменную
X_test, y_test = test.iloc[:, :-1], test.iloc[:, -1]

# предсказываем метки классов для тестовой выборки
y_pred = clf.predict(X_test)

# печатаем точность
print(f"Model test accuracy is: {accuracy_score(y_test, y_pred)}")
