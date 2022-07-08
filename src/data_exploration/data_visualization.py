import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def paint_bar_chart(serie: pd.Series):
    serie_grouped: pd.Series = serie.groupby(serie).sum()
    serie_grouped.plot(kind='bar')
    plt.show()


def paint_biplot(pca: np.ndarray, label: np.ndarray = None):
    figure = plt.figure(figsize=(4, 4))
    ax = figure.add_subplot(111, projection='3d')
    coordinates_x: list[float] = []
    coordinates_y: list[float] = []
    coordinates_z: list[float] = []
    for element in pca:
        coordinates_x.append(element[0])
        coordinates_y.append(element[1])
        coordinates_z.append(element[2])
    ax.scatter(coordinates_x, coordinates_y, coordinates_z, marker='o', c=label)
    ax.set_title("PCA plot and clustering")
    plt.show()


def paint_line_chart(serie: pd.Series):
    serie.value_counts().plot(kind='line')
    plt.show()


def paint_boxplot(serie: pd.Series):
    serie.plot(kind='box')
    plt.show()


def paint_histogram(serie: pd.Series):
    serie.plot(kind='hist')
    plt.show()
