import pandas as pd


def get_outliers(serie: pd.Series) -> pd.Series:
    lower_quartile: float = serie.quantile(q=0.25, interpolation='linear')
    upper_quartile: float = serie.quantile(q=0.75, interpolation='linear')
    interquartile_range: float = upper_quartile - lower_quartile

    lower_limit: float = lower_quartile - 1.5 * interquartile_range
    upper_limit: float = upper_quartile + 1.5 * interquartile_range

    outliers: pd.Series = serie[(serie < lower_limit) | (serie > upper_limit)]

    return outliers
