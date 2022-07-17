import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def pca(dataframe: pd.DataFrame, principal_components: int = None) -> np.ndarray:
    pca: PCA = PCA(n_components=principal_components)
    pca.fit(dataframe)
    result = pca.transform(dataframe)
    eigenvectors: list[int] = pca.components_
    eigenvalues: list[int] = pca.explained_variance_
    return pca, result, eigenvectors, eigenvalues


def n_components(pca_object: PCA):
    plt.style.use('ggplot')
    plt.plot(pca.explained_variance_, marker='o')
    plt.xlabel('Eigenvalue number')
    plt.ylabel('Eigenvalue size')
    plt.title('Scree plot')
    plt.show()


if __name__ == '__main__':
    pca_info: tuple = pca(dataframe=df.sample(n=500), principal_components=3)
    pca: PCA = pca_info[0]
    pca_results: np.ndarray = pca_info[1]
    pca_components = pca_info[2]
    n_components(pca)
