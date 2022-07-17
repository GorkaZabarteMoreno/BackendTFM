import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, StandardScaler


def apply_standard_scaler(serie: pd.Series) -> pd.Series:
    scaler: StandardScaler = StandardScaler()
    data: np.ndarray = serie.to_numpy()
    scaler.fit(data.reshape(-1, 1))
    data_scaled: np.ndarray = scaler.transform(data.reshape(-1, 1))
    result: pd.Series = pd.Series(data_scaled.flatten())
    result = result.round(decimals=3)
    return result


def apply_min_max_scaler(serie: pd.Series) -> pd.Series:
    scaler: MinMaxScaler = MinMaxScaler(feature_range=(-1, 1), clip=True)
    data: np.ndarray = serie.to_numpy()
    scaler.fit(data.reshape(-1, 1))
    data_scaled: np.ndarray = scaler.transform(data.reshape(-1, 1))
    result: pd.Series = pd.Series(data_scaled.flatten())
    result = result.round(decimals=3)
    return result


def standardize(dataframe: pd.DataFrame) -> pd.DataFrame:
    for column in dataframe:
        if dataframe[column].dtype == int or dataframe[column].dtype == float:
            dataframe[column] = apply_standard_scaler(dataframe[column])
    return dataframe
