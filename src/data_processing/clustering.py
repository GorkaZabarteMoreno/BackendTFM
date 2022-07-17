import matplotlib.pyplot as plt
import numpy as np

from scipy.cluster.hierarchy import dendrogram

from sklearn.cluster import AgglomerativeClustering, KMeans


def hierarchical_clustering(data: np.ndarray, linkage: str):
    data = data[0:30]
    agg_cluster: AgglomerativeClustering = AgglomerativeClustering(affinity='euclidean', linkage=linkage,
                                                                   compute_distances=True)
    agg_cluster.fit(data)
    return agg_cluster


def plot_dendrogram(model, **kwargs):
    counts: np.ndarray = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count: int = 0
        for children_idx in merge:
            if children_idx < n_samples:
                current_count += 1
            else:
                current_count += counts[children_idx - n_samples]
        counts[i] = current_count
    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)
    dendrogram(linkage_matrix, **kwargs)
    plt.show()


def elbow_method(data: np.ndarray):
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


def kmeans(data: np.ndarray, n_clusters: int):
    k_means: KMeans = KMeans(n_clusters=n_clusters, init='k-means++')
    k_means.fit(data)
    return k_means
