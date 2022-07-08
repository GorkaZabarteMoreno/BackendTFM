import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import Birch, KMeans


def get_clusters_number(data: np.ndarray):
    sum_squared_distances: list = []
    k_clusters: list = range(1, 10)
    for k in k_clusters:
        k_means: KMeans = KMeans(n_clusters=k)
        k_means.fit(data)
        sum_squared_distances.append(k_means.inertia_)
    plt.plot(k_clusters, sum_squared_distances, 'bx-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Sum of squared distances')
    plt.title('Elbow Method For Optimal k')
    plt.show()


def apply_birch_clustering(data: np.ndarray, n_clusters: int):
    birch_cluster: Birch = Birch(n_clusters=n_clusters)
    birch_cluster.fit(data)
    result: np.ndarray = birch_cluster.predict(data)
    return birch_cluster, result
