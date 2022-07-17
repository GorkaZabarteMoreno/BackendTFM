import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter
from matplotlib.axes import Axes


def bar_chart2D(ser1: pd.Series, ser2: pd.Series, ser3: pd.Series = None):
    plt.figure()
    plt.bar(ser1, ser2, label=ser2.name)
    if ser3 is not None:
        plt.bar(ser1, ser3, label=ser3.name)
    plt.legend()
    plt.show()


def bar_chart(dataframe: pd.DataFrame, column_name: str):
    figure: Figure = plt.figure()
    axes: Axes = figure.add_subplot(111)
    grouped: pd.Series = (dataframe.value_counts(column_name) / len(dataframe)) * 100
    grouped.plot(kind='bar')
    axes.yaxis.set_major_formatter(PercentFormatter())
    plt.show()


def plot2D(pca: np.ndarray, label: np.ndarray = None):
    figure = plt.figure(figsize=(4, 4))
    ax = figure.add_subplot(111)
    coordinates_x: list[float] = []
    coordinates_y: list[float] = []
    for element in pca:
        coordinates_x.append(element[0])
        coordinates_y.append(element[1])
    ax.scatter(coordinates_x, coordinates_y, marker='o', c=label)
    plt.show()


def plot3D(pca: np.ndarray, label: np.ndarray = None):
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
    plt.show()


def line_chart(serie: pd.Series):
    serie.value_counts().plot(kind='line')
    plt.show()


def boxplot(serie: pd.Series):
    serie.plot(kind='box')
    plt.show()


def histogram(serie: pd.Series):
    serie.plot(kind='hist')
    plt.xticks(serie)
    plt.ylabel('Frecuencias')
    plt.xlabel(serie.name)
    plt.show()


def scatter_plot2D(ser1: pd.Series, ser2: pd.Series):
    plt.scatter(ser1, ser2)
    plt.xlabel(ser1.name)
    plt.ylabel(ser2.name)
    plt.show()


def scatter_plot3D(ser1: pd.Series, ser2: pd.Series, ser3: pd.Series):
    figure: Figure = plt.figure(figsize=(4, 4))
    axes: Axes = figure.add_subplot(111, projection='3d')
    axes.scatter(ser1, ser2, ser3, marker='o')
    plt.xlabel(ser1.name)
    plt.ylabel(ser2.name)
    plt.show()
