import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def apply_pca(dataframe: pd.DataFrame, principal_components: int = None) -> np.ndarray:
    pca: PCA = PCA(n_components=principal_components)
    pca.fit(dataframe)
    result = pca.transform(dataframe)
    return pca, result


def get_n_components(pca: PCA):
    plt.style.use('ggplot')
    plt.plot(pca.explained_variance_, marker='o')
    plt.xlabel('Eigenvalue number')
    plt.ylabel('Eigenvalue size')
    plt.title('Scree plot')
    plt.show()
