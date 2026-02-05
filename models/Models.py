from tkinter import Y
import pandas as pd
import numpy as np
import matplotlib as mlp

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


df = pd.read_csv('')

X = df[['']]
y = df[['']]



lr = LinearRegression()
lr.fit(X, y)

predicted_output_lr = lr.predict(X, y)

reg = RandomForestRegressor()
reg.fit(X, y)

predicted_output_reg = reg.predict(X,y)



mae_lr = mean_absolute_error(lr)
mae_reg = mean_absolute_error(reg)

R2_lr = r2_score(y_pred=predicted_output_lr)
R2_reg = r2_score(y_pred=predicted_output_reg)

print("Linear regression :")
print("Predicted output: ", predicted_output_lr)


print("Random forest Regression :")
print("Predicted output: ", predicted_output_reg)
