import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
from mlflow.models import infer_signature
import joblib

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

if __name__ == "__main__":
    # Загружаем очищенные данные
    df = pd.read_csv("./df_clear.csv")
    X = df.drop(columns=['charges'])
    y = df['charges']                    # исходные значения, без преобразований

    # Масштабируем только признаки
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Делим выборку
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42
    )
    
    params = {
        'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1],
        'l1_ratio': [0.001, 0.05, 0.1, 0.2],
        'penalty': ['l1', 'l2', 'elasticnet'],
        'loss': ['squared_error', 'huber', 'epsilon_insensitive'],
        'fit_intercept': [False, True]
    }
    
    # Создаём эксперимент в MLflow
    mlflow.set_experiment("insurance_charges_model")
    
    with mlflow.start_run() as run:
        # Базовый регрессор
        lr = SGDRegressor(random_state=42)
        clf = GridSearchCV(lr, params, cv=3, n_jobs=4)
        
        # Обёртка с автоматическим преобразованием y
        model = TransformedTargetRegressor(
            regressor=clf,
            transformer=PowerTransformer()
        )
        
        # Обучаем на исходных y_train
        model.fit(X_train, y_train)
        
        # Предсказания уже в исходной шкале
        y_pred = model.predict(X_val)
        
        # Вычисляем метрики на исходных данных
        rmse, mae, r2 = eval_metrics(y_val, y_pred)
        
        # Логируем параметры лучшего найдённого регрессора
        best_params = model.regressor_.best_params_
        mlflow.log_params(best_params)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        
        # Сохраняем модель (вся обёртка) с сигнатурой
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(model, "model", signature=signature)
        
        # Сохраняем путь к артефакту в файл для деплоя
        artifact_uri = run.info.artifact_uri.replace("file://", "") + "/model"
        with open("best_model.txt", "w") as f:
            f.write(artifact_uri)
        
        # Выводим путь в лог Jenkins
        print(f"Model saved at: {artifact_uri}")
