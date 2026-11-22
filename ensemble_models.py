"""Эксперимент с ансамблем или полносвязной нейронной сетью."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

from grader_contracts.ensembles import EnsembleInput, EnsembleResult


def run_ensemble_experiment(data: EnsembleInput) -> EnsembleResult:
    """Обучите выбранную модель и верните вероятности, метки и метрики на test."""
    
    # Преобразуем входные данные в numpy массивы
    X_train = np.array(data.train_features, dtype=float)
    y_train = np.array(data.train_target, dtype=float).flatten()
    X_test = np.array(data.test_features, dtype=float)
    y_test = np.array(data.test_target, dtype=float).flatten()
    
    # Масштабирование признаков (важно для MLP, полезно для градиентного бустинга)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Пробуем несколько моделей и выбираем лучшую по ROC-AUC
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            random_state=data.random_seed,
            n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=data.random_seed
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=data.random_seed
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=data.random_seed,
            early_stopping=True,
            validation_fraction=0.1
        )
    }
    
    best_model_name = None
    best_roc_auc = 0.0
    best_probabilities = None
    best_predictions = None
    
    # Обучаем каждую модель и выбираем лучшую
    for name, model in models.items():
        # Для деревьев не масштабируем, для MLP и градиентного бустинга — да
        if name in ["RandomForest"]:
            X_train_fit = X_train
            X_test_pred = X_test
        else:
            X_train_fit = X_train_scaled
            X_test_pred = X_test_scaled
        
        model.fit(X_train_fit, y_train)
        
        # Получаем вероятности и предсказания
        probabilities = model.predict_proba(X_test_pred)[:, 1]
        predictions = model.predict(X_test_pred)
        
        # Считаем ROC-AUC
        roc_auc = roc_auc_score(y_test, probabilities)
        
        print(f"{name}: ROC-AUC = {roc_auc:.4f}")
        
        # Выбираем лучшую модель
        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            best_model_name = name
            best_probabilities = probabilities
            best_predictions = predictions
    
    print(f"\nВыбрана модель: {best_model_name} с ROC-AUC = {best_roc_auc:.4f}")
    
    # Считаем остальные метрики для лучшей модели
    accuracy = accuracy_score(y_test, best_predictions)
    precision = precision_score(y_test, best_predictions, zero_division=0)
    recall = recall_score(y_test, best_predictions, zero_division=0)
    f1 = f1_score(y_test, best_predictions, zero_division=0)
    
    return EnsembleResult(
        model_name=best_model_name,
        probabilities=best_probabilities,
        predictions=best_predictions,
        roc_auc=float(best_roc_auc),
        accuracy=float(accuracy),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1)
    )

"""Запуск эксперимента с ансамблями."""
from sklearn.model_selection import train_test_split
from grader_contracts.ensembles import EnsembleInput
from ensemble_models import run_ensemble_experiment

# Загружаем данные
df = pd.read_csv('german.csv', sep=';')

# Разделяем на признаки и целевую переменную
X = df.drop('Creditability', axis=1)
y = df['Creditability']

# Разделяем на train и test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Создаём объект входных данных
ensemble_input = EnsembleInput(
    train_features=X_train,
    train_target=y_train,
    test_features=X_test,
    test_target=y_test,
    random_seed=42
)

# Запускаем эксперимент
result = run_ensemble_experiment(ensemble_input)

# Выводим результаты
print("\n=== РЕЗУЛЬТАТЫ ===")
print(f"Модель: {result.model_name}")
print(f"ROC-AUC: {result.roc_auc:.4f}")
print(f"Accuracy: {result.accuracy:.4f}")
print(f"Precision: {result.precision:.4f}")
print(f"Recall: {result.recall:.4f}")
print(f"F1: {result.f1:.4f}")

# Сохраняем в JSON
import json
experiment_results = {
    "model_name": result.model_name,
    "roc_auc": result.roc_auc,
    "accuracy": result.accuracy,
    "precision": result.precision,
    "recall": result.recall,
    "f1": result.f1
}

with open("experiment_results.json", "w", encoding="utf-8") as f:
    json.dump(experiment_results, f, indent=2, ensure_ascii=False)

print("\nРезультаты сохранены в experiment_results.json")
