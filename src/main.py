from data_exploration.data_exploration import get_corr_matrix
from data_exploration.data_visualization import paint_biplot

from data_preprocessing.basic_preprocessing import delete_columns, delete_row_instances, df, \
    generate_columns_from_game, format_string, split_dataset
from data_preprocessing.decomposition import apply_pca, get_n_components
from data_preprocessing.encoders import apply_label_encoder
from data_preprocessing.standardization import apply_standard_scaler

from data_processing.clustering import apply_birch_clustering, get_clusters_number
from data_processing.neural_network import create_neural_network, compile_neural_network, train_neural_network
from data_processing.training_split import split_training_dataset

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import Birch
from sklearn.decomposition import PCA
from keras import Sequential
from keras.callbacks import History

from time import perf_counter

if __name__ == '__main__':

    delete_columns(df, 'date', 'position', 'goals_ag_itb', 'goals_ag_otb', 'saves_itb', 'saves_otb', 'saved_pen')
    df = delete_row_instances(df, 'role', 'GK')

    # Data preparation
    format_string(df['competition'])
    df = generate_columns_from_game(df)
    split_dataset(df, 'competition')

    df = pd.read_csv(filepath_or_buffer='../data/euro_2016.csv')
    delete_columns(df, 'competition', 'game', 'Unnamed: 0')
    format_string(df['role'], df['player'], df['rater'], df['team'], df['home_team'], df['away_team'])

    # Label encoding
    for column in df:
        if df[column].dtype == object:
            df[column] = apply_label_encoder(df[column])

    # Standardization
    for column in df:
        if df[column].dtype == int or df[column].dtype == float:
            df[column] = apply_standard_scaler(df[column])

    # Principal component analysis
    pca_info: tuple = apply_pca(df.sample(n=200), principal_components=3)
    pca: PCA = pca_info[0]
    pca_results: np.ndarray = pca_info[1]
    get_n_components(pca)

    # Correlation matrix
    # get_corr_matrix(df)

    # Clustering
    get_clusters_number(pca_results)
    birch_info: tuple = apply_birch_clustering(data=pca_results, n_clusters=5)
    birch: Birch = birch_info[0]
    birch_results: np.ndarray = birch_info[1]
    paint_biplot(pca_results, birch_results)

    # Train-Test split
    x_train, y_train, x_test, y_test = split_training_dataset(df, 'rate', 0.75)

    # Neural network
    start_time = perf_counter()

    model: Sequential = create_neural_network(x_train.shape[1], 1)
    compile_neural_network(model, 0.001)
    y: History = train_neural_network(model, x_train, y_train, x_test, y_test, 1000, 512)

    results: pd.DataFrame = pd.DataFrame(y.history)
    print(results)

    results.plot(figsize=(8, 5))
    plt.grid(True)
    plt.xlabel("Epochs")
    plt.ylabel("Mean Squared Error")
    plt.gca().set_ylim(0, 2)
    plt.show()

    print("Error (training): ", round((1 - results.mean_squared_error.values[-1:][0]) * 100, 1), "%")
    print("Error (development test): ", round((1 - results.val_mean_squared_error.values[-1:][0]) * 100, 1), "%")
    print("Loss (training): ", round((1 - results.loss.values[-1:][0]) * 100, 1), "%")
    print("Loss (development test): ", round((1 - results.val_loss.values[-1:][0]) * 100, 1), "%")
    print("Time: ", round((perf_counter() - start_time)), "seconds")


