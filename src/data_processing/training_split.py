import pandas as pd


def split(dataframe: pd.DataFrame, label_name: str, training_ratio: float) -> tuple:
    label: pd.Series = dataframe[label_name]
    del dataframe[label_name]

    n_instances: int = dataframe.shape[0]
    n_training_examples: int = int(n_instances * training_ratio)

    attributes_train: pd.DataFrame = dataframe.values[:n_training_examples]
    label_train: pd.Series = label.values[:n_training_examples]

    attributes_test: pd.DataFrame = dataframe.values[n_training_examples:]
    label_test: pd.Series = label.values[n_training_examples:]

    return attributes_train, label_train, attributes_test, label_test
