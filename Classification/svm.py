import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix

df = pd.read_csv("cell_samples.csv")

print(df.head(10))

print("Columns are: ",df.dtypes)

# drop those rows from the dataframe that have non-numeric value of BareNuc column
print("original shaep:", df.shape)
df = df[pd.to_numeric(df['BareNuc'], errors="coerce").notnull()]
df['BareNuc'] = df['BareNuc'].astype(int)
print("new shape:", df.shape)


# draw a scatter plot of Clump and UnifSize
ax1 = df[df["Class"]==4][0:50].plot(x='Clump', y='UnifSize', kind='scatter', color='Blue', label='Malignant')
df[df['Class']==2][0:50].plot(x='Clump', y='UnifSize', kind='scatter', color="Red", label="Benign", ax=ax1)
plt.savefig("scatter_plot_svm")
plt.show()

# prepare the training and testing data
x = df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]
y = df['Class']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)

svm = SVC(kernel='rbf')

svm.fit(x_train, y_train)
y_test_hat = svm.predict(x_test)

# evaluation
print("Jaccard index: ", accuracy_score(y_test, y_test_hat))
print("F-1 score: ", f1_score(y_test, y_test_hat, average='weighted'))

# compute the confusion matrix
print("Confusion matrix\n", confusion_matrix(y_test, y_test_hat))

print("Classification report\n", classification_report(y_test, y_test_hat, labels=[2, 4]))

# plot the confusion matrix
conf_matrix = plot_confusion_matrix(svm, x_test, y_test, display_labels=[2, 4], normalize=None, cmap=plt.cm.Blues)
conf_matrix.ax_.set_title("Plot of confusion matrix")
plt.savefig("svm_confusion_matrix")

plt.show()
plt.close()





