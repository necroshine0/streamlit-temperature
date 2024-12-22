import numpy as np
import pandas as pd
from datetime import datetime


def running_mean(city_data, window=30):
    x = city_data["temperature"]
    r_mean = np.convolve(x, np.ones(window) / window, mode='valid')
    city_data = city_data.iloc[window-1:].copy()
    city_data[f"mean_{window}"] = r_mean
    return city_data


def find_anomal(data, mean_std_data, cities):
    for city in cities:
        for season in ["winter", "summer", "autumn", "spring"]:
            loc = (city, season)
            high_q, low_q = mean_std_data.loc[loc, ["high_q", "low_q"]]
            data.loc[[loc], "is_anomal"] = (~data.loc[[loc], "temperature"].between(low_q, high_q)).astype(int)
    return data


def process_data(data):
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    running_mean_data = pd.concat(
        [running_mean(data[data["city"] == city]) for city in data["city"].unique()],
        axis=0
    )

    mean_std_data = running_mean_data.groupby(["city", "season"])["mean_30"].agg(["mean", "std"])
    mean_std_data["high_q"] = mean_std_data["mean"] + 2 * mean_std_data["std"]
    mean_std_data["low_q"] = mean_std_data["mean"] - 2 * mean_std_data["std"]

    data_indexed = data.set_index(["city", "season"]).sort_index()
    labeled_data = find_anomal(data_indexed, mean_std_data, data["city"].unique()).reset_index()
    return labeled_data, mean_std_data


def get_season():
    today = datetime.today()

    season_dict = {
        "winter": [12, 1, 2],
        "spring": [3, 4, 5],
        "summer": [6, 7, 8],
        "autumn": [9, 10, 11]
    }

    month_to_season = {
        month: season
        for season, months in season_dict.items()
        for month in months
    }

    return month_to_season[today.month]
