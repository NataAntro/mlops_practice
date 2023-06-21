import pandas as pd
from sklearn.preprocessing import StandardScaler

# загружаем данные
train = pd.read_csv('train/data.csv')
test = pd.read_csv('test/data.csv')

scaler = StandardScaler()

# выделяем признаки и целевую переменную
X_train, y_train = train.iloc[:, :-1], train.iloc[:, -1].values.reshape(-1, 1)
X_test, y_test = test.iloc[:, :-1], test.iloc[:, -1].values.reshape(-1, 1)

# обучаем трансформер и преобразуем данные
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# сохраняем обработанные данные
pd.DataFrame(np.hstack((X_train, y_train))).to_csv('train/data_scaled.csv', index=False)
pd.DataFrame(np.hstack((X_test, y_test))).to_csv('test/data_scaled.csv', index=False)
