import plotly
from plotly import io
from geopy.geocoders import Nominatim
import pickle
import shutil
import os


class TSPPlot:
    layout = {
        "title": "TSP",
        "showlegend": False,
        "geo": {
            "scope": "europe",
            "projection": {"type": "azimuthal equal area"},
            "showland": True
        },
        "margin": {
            "l": 1,
            "r": 1,
            "b": 1,
            "pad": 0
        }
    }

    def __init__(self, cities, username, api_key):
        shutil.rmtree("images/", ignore_errors=True)
        os.mkdir("./images")
        plotly.tools.set_credentials_file(username=username, api_key=api_key)

        try:
            city_data = pickle.load(open("city_data.p", "rb"))
        except FileNotFoundError:
            city_data = {}

        if not city_data or len(city_data) < len(cities):
            geolocator = Nominatim(user_agent="TSP-checker")

            for city in cities:
                g_data = geolocator.geocode(city)
                city_data[city] = {
                    "lat": g_data.latitude,
                    "lon": g_data.longitude,
                }

            pickle.dump(city_data, open("city_data.p", "wb"))
        self.city_data = city_data
        self.cities = [
            {
                "type": "scattergeo",
                "locationmode": "country names",
                "lon": [c["lon"] for n, c in city_data.items()],
                "lat": [c["lat"] for n, c in city_data.items()],
                "mode": "markers",
                "marker": {
                    "size": 6,
                    "color": "black"
                }
            }
        ]

    def plot(self, path, filename, title=None):
        if title is not None:
            self.layout["title"] = title
        paths = [
            {
                "type": "scattergeo",
                "locationmode": "country names",
                "lon": [self.city_data[path[i]]["lon"], self.city_data[path[i + 1]]["lon"]],
                "lat": [self.city_data[path[i]]["lat"], self.city_data[path[i + 1]]["lat"]],
                "mode": "lines",
                "line": {
                    "width": 1,
                    "color": "red"
                }
            }
            for i in range(-1, len(path) - 1)
        ]
        fig = {
            "layout": self.layout,
            "data": self.cities + paths
        }

        io.write_image(fig, 'images/{}'.format(filename))




