import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score


def lin_reg(dataframe: pd.DataFrame, target: pd.Series) -> LinearRegression:
    dataframe = dataframe.to_numpy().reshape(-1, 1)
    target = target.to_numpy().reshape(-1, 1)
    lr: LinearRegression = LinearRegression()
    lr_model: LinearRegression = lr.fit(X=dataframe, y=target)
    return lr_model


def lin_reg_plot(ser1, ser2: pd.Series, intercept: np.ndarray, slope: np.ndarray):
    plt.scatter(ser1, ser2)
    plt.xlabel(ser1.name)
    plt.ylabel(ser2.name)
    x: np.ndarray = np.arange(-3, 5, 0.1)
    y: np.ndarray = intercept + slope * x
    plt.plot(x, y, 'r--')
    plt.show()


def lin_reg_res(lr_model: LinearRegression, new_obs: np.array, dataframe: pd.DataFrame, target: pd.Series):
    lr_prediction = lr_model.predict(new_obs)
    print("Los coeficientes del modelo de Regresión Lineal: ", lr_model.coef_)
    print("La intersección con el ejeY del modelo de Regresión Lineal ", lr_model.intercept_)
    print("Una nueva observación es: ", new_obs)
    print("La predicción es: ", lr_prediction)
    print("El coeficiente de determinación de la predicción es ", lr_model.score(X=dataframe.to_numpy().reshape(-1, 1),
                                                                                 y=target.to_numpy().reshape(-1, 1)))
    lin_reg_plot(ser1=dataframe, ser2=target, intercept=lr_model.intercept_[0], slope=lr_model.coef_[0][0])


def tree_classifier(dataframe: pd.DataFrame, target: pd.Series) -> DecisionTreeClassifier:
    tree: DecisionTreeClassifier = DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=10,
                                                          max_features='log2')
    dataframe = dataframe.to_numpy()
    target = target.to_numpy().reshape(-1, 1)
    tree_model: LinearRegression = tree.fit(X=dataframe, y=target)
    return tree_model


def tree_classifier_res(tree: DecisionTreeClassifier, new_obs: np.array):
    lr_prediction = tree.predict(new_obs.to_numpy().reshape(1, -1))
    print("Las variables más importantes son : ", tree.feature_importances_)
    print("La estructura del árbol :", tree.tree_)
    print("La predicción es: ", lr_prediction)
    plot_tree(decision_tree=tree, max_depth=3, fontsize=12)
    plt.show()


def tree_regressor(dataframe: pd.DataFrame, target: pd.Series) -> DecisionTreeRegressor:
    tree: DecisionTreeRegressor = DecisionTreeRegressor(criterion='squared_error', splitter='best', max_depth=10,
                                                        max_features='log2')
    dataframe = dataframe.to_numpy()
    target = target.to_numpy().reshape(-1, 1)
    tree_model: LinearRegression = tree.fit(X=dataframe, y=target)
    return tree_model


def tree_regressor_res(tree: DecisionTreeRegressor, test_data: pd.DataFrame, test_target: pd.Series):
    tree_prediction = tree.predict(test_data)
    print("Las variables más importantes son : ", tree.feature_importances_)
    print("La estructura del árbol :", tree.tree_)
    print("La predicción es: ", tree_prediction)
    cv = cross_val_score(tree, test_data, test_target, cv=10)
    print("El de accuracy son: ", cv.mean())
    plot_tree(decision_tree=tree, max_depth=3, fontsize=12)
    plt.show()


def value_k(data: pd.DataFrame, target: pd.Series, test_data: pd.DataFrame, test_target: pd.Series):
    k_range = range(1, 10)
    scores: list = []
    for k in k_range:
        knearest_neighbors: KNeighborsClassifier = KNeighborsClassifier(n_neighbors=k)
        knearest_neighbors.fit(data, target)
        knearest_neighbors.predict(test_data)
    plt.figure()
    plt.xlabel('k')
    plt.ylabel('accuracy')
    plt.scatter(k_range, scores)
    plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])


def knn(data: pd.DataFrame, target: pd.Series, k: int) -> KNeighborsClassifier:
    knearest_neighbors: KNeighborsClassifier = KNeighborsClassifier(n_neighbors=k, weights='uniform',
                                                                    metric='euclidean')
    knn_model: KNeighborsClassifier = knearest_neighbors.fit(data, target)
    return knn_model


def knn_res(knn_model: KNeighborsClassifier, test_data: pd.DataFrame):
    knn_prediction = knn_model.predict(test_data[['closeness_centrality', 'degree_centrality']])
    plt.scatter(test_data['closeness_centrality'], test_data['degree_centrality'], c=knn_prediction)
    plt.show()
    print(knn_model)
    print(knn_prediction)


def svm(data: pd.DataFrame, target: pd.Series) -> SVC:
    support_vector_machine: SVC = SVC(C=1.75, kernel='rbf')
    svm_model: SVC = support_vector_machine.fit(data, target)
    return svm_model


def svm_res(model: SVC, dataframe: pd.DataFrame, target: pd.Series):
    prediction = model.predict(dataframe.iloc[0].to_numpy().reshape(1, -1))
    scores = cross_val_score(estimator=model, X=dataframe, y=target)
    print("La precisión del modelo es: ", scores)
    print("Atributos de un jugador ", dataframe.iloc[0])
    print("La predicción de la posición es: ", prediction)
