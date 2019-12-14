from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

path = "Cust_Segmentation.csv"

df = pd.read_csv(path)

print("Original dataset\n", df.head(10))

# drop the address feature since it is categorical and
# hence k-Means cannot process it

df = df.drop('Address', axis=1)

print("After dropping address column\n", df.head(10))

# normalize the dataset
# values removes the axis labels and returns a numpys ndarray; start features from age onwards
X = df.values[:, 1:]
print("this is x\n", X)
X = np.nan_to_num(X)
X_copy = X
scaler = StandardScaler()
X = scaler.fit(X).transform(X)

print("After normalizing\n", X)


# apply k-Means and see the labels
k_means_object = KMeans(init='k-means++', n_init=12, n_clusters=3)
k_means_object.fit(X)
labels = k_means_object.labels_

print("Cluster labels: ", labels)
df["Cluster"] = labels
print("Clustered dataframe\n", df)

# check the centroid values
print("Centroid values\n", df.groupby("Cluster").mean())

# scatter plot based on age and income
print(df.columns)
plt.scatter(X_copy[:,0], X_copy[:,3], c=labels.astype(np.float))
plt.xlabel('Age')
plt.ylabel('Income')
plt.title("Customer Segmentation")
plt.savefig("cust_segment_plot")
plt.show()
plt.close()

# draw a 3-D plot
fig = plt.figure(1, figsize=(8, 6))
plt.clf()
axis3D = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
plt.cla()


axis3D.set_xlabel('Education')
axis3D.set_ylabel('Age')
axis3D.set_zlabel('Income')

axis3D.scatter(X_copy[:, 1], X_copy[:, 0], X_copy[:, 3], c=labels.astype(np.float))

plt.savefig('3D_plot_cust_segment')
plt.show()
plt.close()
