import asyncio
import aiohttp
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from utils import process_data, get_season


async def get_temp(city, api_key=None):
	async with aiohttp.ClientSession() as session:
	    payload = {"q": city}
	    if api_key is not None:
	        payload["appid"] = api_key

	    async with session.get("http://api.openweathermap.org/geo/1.0/direct", params=payload) as response:
	        response_data = await response.json()
	        lat, lon = response_data[0]["lat"], response_data[0]["lon"]

	    payload = {"lat": lat, "lon": lon, "units": "metric"}
	    if api_key is not None:
	        payload["appid"] = api_key

	    async with session.get("https://api.openweathermap.org/data/2.5/weather", params=payload) as response:
	        weather_data = await response.json()
	        return weather_data["main"]["temp"]


def load_data(file):
    data = pd.read_csv(file)
    if list(data.columns) != ["city", "timestamp", "temperature", "season"]:
        raise ValueError("Passed wrong data format!")
    return data


def display_statistics(data):
    st.write("Описательная статистика температуры с привязкой к сезону")
    st.write(data.groupby("season")["temperature"].describe())


def plot_temperature_series(data, title=None):
    plt.figure(figsize=(12, 5))
    plt.plot(data['timestamp'], data['temperature'], label='Температура', color='royalblue')

    window = 30
    running_mean = np.convolve(data['temperature'], np.ones(window) / window, mode='valid')
    plt.plot(data['timestamp'].values[window-1:], running_mean, label=f'Среднее ({window} дней)', color='lime')

    anomaly_data = data[data["is_anomal"] == 1]
    plt.scatter(anomaly_data["timestamp"], anomaly_data['temperature'], label='Аномалии', color='red')

    if title is not None:
        plt.title(title)

    plt.xlabel('Дата')
    plt.ylabel('Температура')
    plt.xticks(rotation=45)
    plt.legend()
    st.pyplot(plt)


def plot_profiles(data, title=None):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 6))
    for i, season in enumerate(["winter", "spring", "autumn", "summer"]):
        ax = axes[i // 2][i % 2]
        ax.set_title(season)
        ax.set_ylim([-25, 50])
        if i % 2 == 1:
            ax.set_yticks([])
        
        season_data = data[data["season"] == season].copy()
        season_data["date"] = season_data['timestamp'].dt.strftime("%d/%m")
        mean_temps = season_data.groupby("date")["temperature"].mean()
        
        ax.plot(range(len(mean_temps)), mean_temps, color='blue')
        
        mean = season_data['temperature'].mean()
        std = season_data['temperature'].std()
        if season != "winter":
        	ax.annotate(f"mean: {mean:.2f}\nstd: {std:.2f}", xy=(70, -20))
        else:
        	ax.annotate(f"mean: {mean:.2f}\nstd: {std:.2f}", xy=(70, 35))

    fig.supxlabel('№ дня сезона')
    fig.supylabel('Средняя погодоваря температура')
    plt.tight_layout()
    if title is not None:
        fig.suptitle(title, y=1.02)
    st.pyplot(plt)


def main():
    st.title("Приложение для анализа погоды")

    uploaded_file = st.file_uploader("Загрузите файл с историческими данными", type=["csv"])
    api_key = st.text_input("Введите ключ доступа к API", type="password")

    if uploaded_file is not None:
        data = load_data(uploaded_file)
        data, mean_std_data = process_data(data)
        st.write("Данные загружены!")

        city = st.selectbox("Выберите город из списка", list(data["city"].unique()))
        city_data = data[data["city"] == city].sort_values(by="timestamp")
        display_statistics(city_data)
        plot_temperature_series(city_data, title=f"Временной ряд температуры для города {city}")
        plot_profiles(city_data, title=f"Сезонные профили температуры для города {city}")

    if api_key is not None and api_key != "":
        st.write("api_key загружен!")

        try:
            temp = asyncio.run(get_temp(city, api_key))
            season = get_season()

            loc = (city, season)
            high_q, low_q = mean_std_data.loc[loc, ["high_q", "low_q"]]
            is_normal = (low_q <= temp <= high_q)
        except Exception as e:
            print(e)
            st.write("Указан неверный api_key!")

        st.write(f"Температура в городе {city} в данный момент времени: {temp:.2f} C, сезон: {season}")
        st.write(f"Исходя из исторических данных, эта температура {'не ' * is_normal}считается аномальной для указанного сезона")

if __name__ == "__main__":
    main()