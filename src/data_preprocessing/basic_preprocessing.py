import numpy as np
import pandas as pd
import re

col_names: dict = {"match": "game", "pos": "position", "pos_role": "role", "original_rating": "rate",
                   "shotsblocked": "shots_blocked", "chances2score": "chances_to_score", "keypasses": "key_passes",
                   "wasfouled": "was_fouled", "ycards": "yellow_cards", "rcards": "red_cards",
                   "dangmistakes": "dang_mistakes", "owngoals": "own_goals",
                   "betweenness2goals": "betweenness_to_goals", "minutesPlayed": "mins"}

df = pd.read_csv(filepath_or_buffer='../data/data_football_ratings.csv')
df = df.rename(columns=col_names)


def format_string(*ser: pd.Series):
    for s in ser:
        column_values: np.ndarray = s.unique()
        for value in column_values:
            aux_value: str = value.lower()
            aux_value = aux_value.rstrip()
            aux_value = aux_value.replace(' ', '_')
            s[s == value] = aux_value


def delete_columns(dataframe: pd.DataFrame, *column_names: str):
    for column in column_names:
        del dataframe[column]


def delete_row_instances(dataframe: pd.DataFrame, column_name: str, column_value: str) -> pd.DataFrame:
    dataframe: pd.DataFrame = dataframe.loc[dataframe[column_name] != column_value]
    return dataframe


def get_game_attr(dataframe: pd.DataFrame):
    games: pd.Series = dataframe['game']
    home_teams: list[str] = []
    away_teams: list[str] = []
    home_teams_goals: list[int] = []
    away_teams_goals: list[int] = []
    for g in games:
        attr: list = re.split('- |,', g)
        home_teams.append(attr[0])
        away_teams.append(attr[1])
        home_teams_goals.append(attr[2])
        away_teams_goals.append(attr[3])
    return home_teams, away_teams, home_teams_goals, away_teams_goals


def generate_columns_from_game(dataframe: pd.DataFrame) -> pd.DataFrame:
    home_teams, away_teams, home_teams_goals, away_teams_goals = get_game_attr(dataframe)
    attr: list = list(zip(home_teams, away_teams, home_teams_goals, away_teams_goals))
    aux: pd.DataFrame = pd.DataFrame(attr, columns=["home_team", "away_team", "home_team_goals", "away_team_goals"])
    dataframe = pd.concat([aux, dataframe], axis=1)
    return dataframe


def split_dataset(dataframe: pd.DataFrame, column_name: str):
    column: pd.Series = dataframe[column_name]
    column_values: np.ndarray = column.unique()

    for value in column_values:
        aux_dataframe: pd.DataFrame = dataframe[dataframe[column_name] == value]
        url: str = '../data/' + str(value) + '.csv'
        aux_dataframe.to_csv(path_or_buf=url)