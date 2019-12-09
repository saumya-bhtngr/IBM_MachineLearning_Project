import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

path = "FuelConsumptionCo2.csv"
df = pd.read_csv(path)

cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
x = df[['ENGINESIZE']]
y = df[['CO2EMISSIONS']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
pf = PolynomialFeatures(degree=2)
x_train_poly = pf.fit_transform(x_train)

#print(x_train_poly)

lr = LinearRegression()
lr.fit(x_train_poly, y_train)

x_test_poly = pf.fit_transform(x_test)
y_test_hat = lr.predict(x_test_poly)

# Get the coefficients
print("Intercept: ", lr.intercept_)
print("Coefficients are: ", lr.coef_)

# Plot engine size versus emissions with these coefficient values
plt.scatter(x_train, y_train, color = 'blue')
XX = np.arange(0, 10, 0.1)
yy = lr.intercept_ + lr.coef_[0][1] * XX + lr.coef_[0][2] * XX * XX
plt.plot(XX, yy, '-r')
plt.xlabel("Engine Size")
plt.ylabel("CO2 Emissions")
plt.savefig("polynomial_regression_scatter_plot")

plt.show()

print("Mean Absolute Error: ", np.mean(np.absolute(y_test - y_test_hat)))
print("Mean Squared Error: ", np.mean(np.square(y_test - y_test_hat)))
print("R-score: ", r2_score(y_test, y_test_hat))

