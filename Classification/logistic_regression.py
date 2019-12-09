import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import jaccard_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix

df = pd.read_csv("ChurnData.csv")

print(df.head(5))
print(df.columns)

x = df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']]
df['churn'] = df['churn'].astype(int)

y = df['churn']


# normalize the dataset
scaler = StandardScaler()
x = scaler.fit(x).transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)

print("training data size ", x_train.shape, y_train.shape)
print("testing data size ", x_test.shape, y_test.shape)

# C parameter implies inverse of regularization constant
# smaller values imply stronger regularization
log_reg = LogisticRegression(C=0.01, solver="liblinear")
log_reg.fit(x_train, y_train)

y_test_hat = log_reg.predict(x_test)

print("Predicted values\n", y_test_hat)

# find the probabilities of class 1 and class 0
y_hat_prob = log_reg.predict_proba(x_test)
print("Outcome probabilities\n", y_hat_prob)

# evaluation

# jaccard index
print("Jaccard index: ", accuracy_score(y_test, y_test_hat))

# log loss
print("Log loss value: ", log_loss(y_test, y_hat_prob))

# confusion matrix
conf_matrix = confusion_matrix(y_test, y_test_hat, labels=[1, 0])

print('Confusion matrix:\n', conf_matrix)

print('Evaluation report:\n', classification_report(y_test, y_test_hat))

# plot the confusion matrix

type = [("without normalization", None), ("with normalization", 'true')]

for title, normalization in type:
    disp = plot_confusion_matrix(log_reg, x_test, y_test, display_labels=[1, 0], cmap=plt.cm.Blues, normalize=normalization)
    disp.ax_.set_title(title)
    print(title)
    print(disp.confusion_matrix)
    plt.savefig("confusion_matrix_{}".format(title))


plt.show()
plt.close()






