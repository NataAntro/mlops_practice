from sklearn.linear_model import LogisticRegression
import pandas as pd
import pickle

# загружаем данные
train = pd.read_csv('train/data_scaled.csv')

# выделяем признаки и целевую переменную
X_train, y_train = train.iloc[:, :-1], train.iloc[:, -1]

# обучаем модель
clf = LogisticRegression().fit(X_train, y_train)

# сохраняем модель
with open('model.pkl', 'wb') as f:
    pickle.dump(clf, f)
