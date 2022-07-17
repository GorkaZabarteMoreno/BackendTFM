import pandas as pd

from data_exploration.data_exploration import corr_matrix
from data_preprocessing.basic_preprocessing import preprocessing
from data_processing.clustering import plot_dendrogram, hierarchical_clustering, elbow_method, kmeans
from data_preprocessing.encoders import encode
from data_preprocessing.standardization import standardize
from data_processing.machine_learning import svm, knn, value_k, lin_reg, lin_reg_res, tree_classifier, \
    tree_classifier_res, svm_res, tree_regressor, tree_regressor_res
from data_processing.neural_network import neural_network
from src.data_exploration.data_visualization import scatter_plot3D, plot3D, plot2D
from src.data_preprocessing.decomposition import pca

if __name__ == '__main__':
    # Basic preprocessing
    # euro: pd.DataFrame = preprocessing(original_dataset="../data/football_ratings.csv", directory_url="../data/",
    #                                   new_dataset="euro_2016.csv")
    # world: pd.DataFrame = preprocessing(original_dataset="../data/football_ratings.csv", directory_url="../data/",
    #                                    new_dataset="world_cup_2018.csv")
    bundesliga: pd.DataFrame = preprocessing(original_dataset="../data/football_ratings.csv", directory_url="../data/",
                                             new_dataset="bundesliga_2017-18.csv")
    premier: pd.DataFrame = preprocessing(original_dataset="../data/football_ratings.csv", directory_url="../data/",
                                          new_dataset="premier_league_2017-18.csv")

    cat_win_target: pd.Series = premier['win']
    rate_train_target: pd.Series = premier['rate']
    rate_test_target: pd.Series = bundesliga['rate']
    cat_dang_mistakes_train = bundesliga['dang_mistakes']
    cat_dang_mistakes_test = premier['dang_mistakes'].head(10000)

    # Encoders
    # euro = encode(euro)
    # world = encode(world)
    bundesliga = encode(bundesliga)
    premier = encode(premier)

    # Standardization
    # euro = standardize(euro)
    # world = standardize(world)
    bundesliga = standardize(bundesliga)
    premier = standardize(premier)

    # Correlation matrix
    # corr_matrix(euro)
    # corr_matrix(world)
    # corr_matrix(bundesliga)
    # corr_matrix(premier)

    # Clustering
    # hierarchical_avg = hierarchical_clustering(data=premier, linkage='average')
    # hierarchical_sim = hierarchical_clustering(data=premier, linkage='single')
    # hierarchical_com = hierarchical_clustering(data=premier, linkage='complete')
    # plot_dendrogram(model=hierarchical_avg)
    # plot_dendrogram(model=hierarchical_sim)
    # plot_dendrogram(model=hierarchical_com)
    # elbow_method(data=premier)
    # clusters = kmeans(data=premier[0:1600], n_clusters=7)
    # pca2D = pca(dataframe=premier[0:1600], principal_components=2)
    # pca3D = pca(dataframe=premier[0:1600], principal_components=3)
    # plot2D(pca=pca2D[1], label=clusters.labels_[0:1600])
    # plot3D(pca=pca3D[1], label=clusters.labels_[0:1600])

    # Linear Regression
    # lr_model: LinearRegression = lin_reg(dataframe=premier['passes_acc'], target=premier['touches'])
    # observations: np.ndarray = np.array([20, 10, 120]).reshape(-1, 1)
    # lin_reg_res(lr_model=lr_model, new_obs=observations, dataframe=premier['passes_acc'], target=premier['touches'])

    # Classification Tree
    # train_tree = premier.drop(columns=['win', 'team', 'home_team', 'goals', 'lost', 'away_team', 'home_team_goals',
    #                                'away_team_goals', 'assists'])
    # observation = premier.drop(columns=['win', 'team', 'home_team', 'goals', 'lost', 'away_team', 'home_team_goals',
    #              'away_team_goals', 'assists']).iloc[10]
    # tree_c = tree_classifier(dataframe=train_tree, target=cat_win_target)
    # tree_classifier_res(tree=tree_c, new_obs=observation)
    # print(train_tree.columns)

    # SVM
    # train = bundesliga.drop(columns='dang_mistakes')
    # test = premier.drop(columns='dang_mistakes').head(10000)
    # svm_model: SVC = svm(data=train, target=cat_dang_mistakes_train)
    # svm_res(model=svm_model, dataframe=test, target=cat_dang_mistakes_test)

    # KNN
    # knn_data = premier[['closeness_centrality', 'degree_centrality', 'yellow_cards']].sample(800)
    # test = premier.sample(200)
    # value_k(data=knn_data[['closeness_centrality', 'degree_centrality']], target=knn_data['yellow_cards'],
    #         test_data=test[['closeness_centrality', 'degree_centrality']], test_target=test['yellow_cards'])
    # knn_model: KNeighborsClassifier = knn(data=knn_data[['closeness_centrality', 'degree_centrality']],
    #                                     target=knn_data['yellow_cards'], k=1)
    # knn_res(knn_model=knn_model, test_data=test)

    # Regression tree
    train_tree = premier.drop(columns='rate')
    test_tree = bundesliga.drop(columns='rate')
    tree_c = tree_regressor(dataframe=train_tree, target=rate_train_target)
    tree_regressor_res(tree=tree_c, test_data=test_tree, test_target=rate_test_target)

    # Neural network
    # neural_network(premier, 'rate')
