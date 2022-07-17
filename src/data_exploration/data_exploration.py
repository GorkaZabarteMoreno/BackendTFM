import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def corr_matrix(dataframe: pd.DataFrame):
    correlation_matrix: pd.DataFrame = dataframe.iloc[:, 15:-6].corr(method="pearson")
    sns.heatmap(correlation_matrix.round(decimals=2), annot=True, xticklabels=True, yticklabels=True, square=True)
    plt.show()
