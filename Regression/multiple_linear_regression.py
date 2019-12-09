import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

path = "FuelConsumptionCo2.csv"
df = pd.read_csv(path)

x = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']]
y = df[['CO2EMISSIONS']]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

mlr = LinearRegression()
mlr.fit(x_train, y_train)
y_test_hat = mlr.predict(x_test)

print("Coefficients are: ", mlr.coef_)

print("Mean squared error: ", np.mean(np.square(y_test - y_test_hat)))
print("Mean squared error default method: ", mean_squared_error(y_test, y_test_hat))

print("R-score is: ", mlr.score(x_test, y_test))