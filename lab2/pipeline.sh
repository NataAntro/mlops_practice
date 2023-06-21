#!/bin/bash

# Загружаем нужные библиотеки
echo "----Import library-----"
pip install pandas
pip install numpy
pip install scikit-learn
pip install joblib

echo "----Create Dataset (begin)-----"
python3 /media/psf/Home/lab222/create_dataset.py
echo "----Create Dataset (end)-----"

# Подготовка данных
echo "----Data Preprocessing (begin)-----"
python3 /media/psf/Home/lab222/data_preprocessing.py
echo "----Data Preprocessing (end)-----"

# Обучение модели
echo "----Model Training (begin)-----"
python3 /media/psf/Home/lab222/model_preparation.py
echo "----Model Training (end)-----"

# Тестирование модели и вывод метрики
echo "----Model Testing (begin)-----"
python3 /media/psf/Home/lab222/model_testing.py
echo "----Model Testing (end)-----"
