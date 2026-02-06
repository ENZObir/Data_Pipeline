from os import error
import pandas as pd 
import mlflow

from pydantic.type_adapter import P
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


df = pd.read_csv('./api/setosa.csv')

X = df[['sepal_width']]
y = df['sepal_length']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=20, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


print(" LinearRegression model trained ✓")
print(f"First real value : {y_test.values[:5]}")
print(f"First predictions LR : {y_pred[:5]}")

print("Metrics")
print(f"MSE : {mse}")
print(f"MAE : {mae}")
print(f"R² : {r2}")