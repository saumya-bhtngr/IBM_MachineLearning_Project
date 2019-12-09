import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

path = "FuelConsumptionCo2.csv"
df = pd.read_csv(path)

print("First 5 rows \n", df.head())


# descriptive exploration

print("Descriptive summary \n", df.describe())

# plot histograms of features 'ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS'
df_features = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
df_features.hist()
plt.savefig("feature_histogram")
plt.show()
plt.close()

#  plot these feature vs the Emission, to see how linear is their relation
sns.regplot(x="FUELCONSUMPTION_COMB", y="CO2EMISSIONS", data=df)
plt.xlabel("Fuel Consumption")
plt.ylabel("Emmisions")
plt.title("Fuel Consumption Vs Emissions")
plt.savefig("regression_plot_1")
plt.close()

#  plot these feature vs the Emission, to see how linear is their relation
sns.regplot(x="ENGINESIZE", y="CO2EMISSIONS", data=df).set
plt.xlabel("Engine Size")
plt.ylabel("Emmisions")
plt.title("Engine Size Vs Emissions")
plt.savefig("regression_plot_2")
plt.close()

# plot scatter plot of CYLINDER vs the Emission
plt.scatter(df_features.CYLINDERS, df_features.CO2EMISSIONS, color='blue')
plt.xlabel("Cylinder Values")
plt.ylabel("Emission")
plt.title("CYLINDER vs Emission")
plt.savefig("scatter_plot")
plt.show()


# linear regression model
lr = LinearRegression()
x = df[['ENGINESIZE']]
y = df[['CO2EMISSIONS']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
lr.fit(x_train, y_train)

# print the coefficients
print("intercept value is: ", lr.intercept_)
print("coefficient value is: ", lr.coef_)

# use evaluation metrics
y_hat = lr.predict(x_test)
print("Mean absolute error: ", np.mean(np.absolute(y_test - y_hat)))
print("Residual sum of squares (MSE): ", np.mean(np.square(y_test - y_hat)))
print("R2-score: ", lr.score(x_test, y_test), r2_score(y_test, y_hat))












