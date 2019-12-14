import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# generate a random seed
np.random.seed(0)

# make random clusters of points
# input inlcudes number of data points, their centers and standard deviation of clusters
X, y = make_blobs(n_samples=5000, centers=[[4, 4], [-2, -1], [2, -3],[1,1]] , cluster_std=0.9)

print("Feature vector\n", X)
print("Response vector\n", y)

# draw a scatter plot of X
plt.scatter(X[:,0], X[:,1], marker='.')
plt.savefig("kMeans_scatter_plot")
plt.show()
plt.close()

# use k-means clustering
# n_init denotes the number of times k-means algorithm will run with different centroids
k_means_object = KMeans(init='k-means++', n_clusters=4, n_init=12)
k_means_object.fit(X)
k_means_labels = k_means_object.labels_
k_means_cluster_centers = k_means_object.cluster_centers_

print("Labels: ", k_means_labels)
print("Cluster centers: ", k_means_cluster_centers)

# plot the clustered data along with centroids
fig = plt.figure(figsize=(6, 4))

# create an array of colors equal to length of labels
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels))))

# add an Axis object to the figure
ax = fig.add_subplot()

for k, c in zip(range(4), colors):
    # create an array of true/false values based on cluster labels
    membership = (k_means_labels == k)

    # define the centroid
    centroid = k_means_cluster_centers[k]

    # plot the points
    ax.plot(X[membership, 0], X[membership, 1], 'w', marker='.', markerfacecolor=c)

    # plot the centroids
    ax.plot(centroid[0], centroid[1], 'o', markerfacecolor=c, markersize=6, markeredgecolor='k')

# remove x-axis and y-axis ticks
ax.set_xticks(())
ax.set_yticks(())
ax.set_title("K-Means Clusters")

plt.savefig("data_clusters")
plt.show()
plt.close()

print(membership)













