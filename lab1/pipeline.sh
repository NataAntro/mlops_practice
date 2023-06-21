#!/bin/bash

# Запуск python скрипта для создания исходных данных.
python3 data_creation.py

# Запуск python скрипта для предобработки данных.
python3 data_preprocessing.py

# Запуск python скрипта для создания и обучения модели машинного обучения.
python3 model_preparation.py

# Запуск python скрипта для тестирования обученной модели на тестовых данных.
python3 model_testing.py
