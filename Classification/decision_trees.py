import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics

path = "drug200.csv"

df = pd.read_csv(path)

print(df.head(5))

# size of the data
print("size of the data is: ",df.shape[0], "rows", df.shape[1], "columns")
print(df.columns)

# feature set and target variable
x = df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
y = df['Drug'].values

# convert the categorical variables to numeric variable using get_dummies()
#df_gender = pd.get_dummies(df['Sex'])
#print(df_gender.head(5))
#df = pd.concat([df, df_gender], axis=1)
#print(df.head(10))

# use label_encoder to convert categorical variables into numeric ones
lable_enc_gender = LabelEncoder()
lable_enc_gender.fit(['F', 'M'])
x[:,1] = lable_enc_gender.transform(x[:,1])

lable_enc_bp = LabelEncoder()
lable_enc_bp.fit(['HIGH', 'NORMAL', 'LOW'])
x[:,2] = lable_enc_bp.transform(x[:,2])

lable_enc_chol = LabelEncoder()
lable_enc_chol.fit(['HIGH', 'NORMAL', 'LOW'])
x[:,3] = lable_enc_chol.transform(x[:,3])

print("After third encoding\n", x)

# build the model
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=3)
decision_tree = DecisionTreeClassifier(criterion="entropy", max_depth=4)

print(decision_tree)
decision_tree.fit(x_train, y_train)
y_test_hat = decision_tree.predict(x_test)


# check the accuracy
print("Accuracy of the decision tree is:", metrics.accuracy_score(y_test, y_test_hat))


