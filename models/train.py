import os
from pathlib import Path

import pandas as pd
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT_ENV = Path(__file__).resolve().parent.parent / ".env"
if ROOT_ENV.exists():
    load_dotenv(ROOT_ENV)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "iris-training")
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "IrisModel")
MLFLOW_MODEL_STAGE = os.getenv("MLFLOW_MODEL_STAGE", "Production")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


df = pd.read_csv('./api/clean_iris.csv')

X = df[['sepal_width']]
y = df['sepal_length']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

y_pred_lr = model_lr.predict(X_test)

model_reg= RandomForestRegressor()
model_reg.fit(X_train, y_train)

y_pred_reg = model_reg.predict(X_test)


mse_lr = mean_squared_error(y_test, y_pred_lr)
mse_reg = mean_squared_error(y_test, y_pred_reg)

mae_lr = mean_absolute_error(y_test, y_pred_lr)
mae_reg = mean_absolute_error(y_test, y_pred_reg)

R2_lr = r2_score(y_test, y_pred_lr)
R2_reg = r2_score(y_test, y_pred_reg)

print("Linear regression :")
print("Predicted output: ", y_pred_lr)
print(f"MSE : {mse_lr}")
print(f"MAE : {mae_lr}")
print(f"R² : {R2_lr}")


print("Random forest Regression :")
print("Predicted output: ", y_pred_reg)
print(f"MSE : {mse_reg}")
print(f"MAE : {mae_reg}")
print(f"R² : {R2_reg}")


with mlflow.start_run() as run:
    
    mlflow.sklearn.log_model(model_lr, "linear_regression")
    mlflow.sklearn.log_model(model_reg, "Random_forest")

    mlflow.log_metric('Mean_squared_error_Linear_regression' , mse_lr),

    mlflow.log_metric('Mean_squared_error_Random_Forest_regression' , mse_reg)

    mlflow.log_metric('Mean_absolute_error_Linear_regression' , mae_lr),

    mlflow.log_metric('Mean_absolute_error_Random_Forest_regression' , mae_reg)

    mlflow.log_metric('R_squared_Linear_regression' , R2_lr),

    mlflow.log_metric('R_squared_Random_Forest_regression' , R2_reg)


    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)

    model_result = pd.DataFrame(
        {
            'y_reel' : y_test,
            'y_pred_lr' : y_pred_lr,
            'y_pred_reg' : y_pred_reg
        }
    )

    model_result.to_csv('prediction.csv', index= False)

    mlflow.log_artifact('prediction.csv')

    client = MlflowClient()
    model_source = f"runs:/{run.info.run_id}/Random_forest"
    registration = mlflow.register_model(model_source, MLFLOW_MODEL_NAME)
    client.transition_model_version_stage(
        name=MLFLOW_MODEL_NAME,
        version=registration.version,
        stage=MLFLOW_MODEL_STAGE,
        archive_existing_versions=True,
    )


# Après le chargement
print(f"Dataset loaded succesfuly : {df.shape[0]} lignes, {df.shape[1]} colonnes")
print(df.head())

# Après le split
print(f"Train : {X_train.shape[0]} samples | Test : {X_test.shape[0]} samples")

# Après l'entraînement
print(" LinearRegression model trained ✓")
print("RandomForest model trained ✓")

# Après les prédictions (juste quelques valeurs, pas tout)

print(f"First real value : {y_test.values[:5]}")
print(f"First predictions LR : {y_pred_lr[:5]}")
print(f"First predictions REG : {y_pred_reg[:5]}")