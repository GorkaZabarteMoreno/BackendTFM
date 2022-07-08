import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


def apply_label_encoder(serie: pd.Series) -> pd.Series:
    label_encoder: LabelEncoder = LabelEncoder()
    label_encoder.fit(serie)
    result: pd.Series = label_encoder.transform(serie)
    return result


def apply_one_hot_encoder(serie: pd.Series) -> pd.Series:
    one_hot_encoder: OneHotEncoder = OneHotEncoder()
    one_hot_encoder.fit(serie)
    result: pd.Series = one_hot_encoder.transform(serie)
    return result
