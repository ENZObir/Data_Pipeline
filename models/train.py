from ast import mod
import pandas as pd
import mlflow

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


df = pd.read_csv('./api/clean_iris.csv')

X = df[['sepal_width']]
y = df['sepal_length']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_lr = LinearRegression()
mod.fit(X_train, y_train)

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


with mlflow.start_run():
    
    mlflow.log_param('Mean squared error (Linear regression)' , mse_lr),

    mlflow.log_param('Mean squared error (Random Forest regression)' , mse_reg)

    model_result = pd.DataFrame(
        {
            'y_reel' : y_test,
            'y_pred_lr' : y_pred_lr,
            'y_pred_reg' : y_pred_reg
        }
    )

    model_result.to_csv('prediction.csv', index= False)

    mlflow.log_artifact('prediction.csv')