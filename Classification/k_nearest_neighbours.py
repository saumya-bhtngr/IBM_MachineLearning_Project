import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import numpy as np
from matplotlib.ticker import MaxNLocator
from matplotlib.pyplot import figure

# load the .csv file
path = "teleCust1000t.csv"
df = pd.read_csv(path)

print(df.head())

# check the distinct count for column custcat
print("Distinct count of custcat: \n", df["custcat"].value_counts())

# visualize the custcat values
df["custcat"].hist()
plt.show()
plt.close()

# plot a histogram of income field
df.hist(column="income", bins=50)
plt.savefig("histogram_income")
plt.show()
plt.close()

# define feature set and dependent variable
# convert the Pandas data frame to a Numpy array
x = df[['region', 'tenure', 'age','marital', 'address', 'income', 'ed', 'employ', 'retire', 'gender', 'reside']].values
y = df['custcat'].values

# normalize the data
print(x.dtype)
scaler = StandardScaler()
x = scaler.fit(x).transform(x.astype(float))

print("x values after normalization\n", x[0:5])

# split the data
x_train, x_test, y_train, y_test =  train_test_split(x, y, test_size=0.2, random_state=4)
print("training data shape: ",x_train.shape, y_train.shape)
print("testing data shape: ",x_test.shape, y_test.shape)

kNN = KNeighborsClassifier(n_neighbors=4)
kNN.fit(x_train, y_train)
print(kNN)

y_test_hat = kNN.predict(x_test)
y_train_hat = kNN.predict(x_train)
print("prediction: ", y_test_hat[0:5])

# accuracy
# metrics.accuracy_score calculates jaccard similarity score
print("Training accuracy: ", metrics.accuracy_score(y_train, y_train_hat))
print("Testing accuracy: ", metrics.accuracy_score(y_test, y_test_hat))

# calculate the accuracy of KNN for different Ks
k = 10
accuracy = np.zeros(k-1)
std_accuracy = np.zeros(k-1)

for i in range(1, k):
    kNN_classifier = KNeighborsClassifier(n_neighbors = i)
    kNN_classifier.fit(x_train, y_train)
    y_test_predict = kNN_classifier.predict(x_test)

    accuracy[i-1] = metrics.accuracy_score(y_test, y_test_predict)

print("Mean accuracy: ", accuracy)
print("k with highest accuracy: ", accuracy.argmax()+1)


# plot no of neighbors versus accuracy
plt.plot(np.arange(1, k), accuracy, '-g')
plt.xlabel("Neighbors")
plt.ylabel("Accuracy")
plt.title("Plot of neighbors versus accuracy")

#ax = figure().gca()
#ax.xaxis.set_major_locator(MaxNLocator(integer=True))

plt.savefig("accuracy_plot")
plt.show()
plt.close()







