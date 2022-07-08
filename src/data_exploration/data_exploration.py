import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def get_corr_matrix(dataframe: pd.DataFrame):
    corr_matrix: pd.DataFrame = dataframe.corr()
    sns.heatmap(corr_matrix, annot=True)
    plt.show()
