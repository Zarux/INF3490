import csv
from itertools import permutations
import time
import random
import numpy as np


class Tsp:

    info = {}

    def __init__(self):
        self.data = self.read_data("european_cities.csv")
        self.city_indices = {city.lower(): idx for idx, city in enumerate(self.data[0])}
        self.data.pop(0)

    def run(self, mode):
        if mode == "exhaustive":
            for n in range(6, 11):
                t0 = time.time()
                trip, distance = self.exhaustive(n_cities=n)
                t1 = time.time() - t0
                print("Path: {} > {} - Distance: {:.2f} - Time: {:.2f}".format(
                    " > ".join([p[0].title() for p in trip]),
                    trip[-1][1].title(),
                    distance,
                    t1
                ))
        elif mode == "hillclimb":
                n = 10
                t0 = time.time()
                trip, distance = self.hill_climb(n_cities=n)
                t1 = time.time() - t0
                print("Path: {} > {}\nDistance: {:.3f}\nTime: {:.3f}\n".format(
                    " > ".join([p[0].title() for p in trip]),
                    trip[-1][1].title(),
                    distance,
                    t1
                ))
                print(self.info["hill"])
        elif mode == "genetic":
            pass

    @staticmethod
    def read_data(f_name: str) -> list:
        with open(f_name, "r") as f:
            return list(csv.reader(f, delimiter=';'))

    def distance(self, a: str, b: str) -> float:
        return float(self.data[self.city_indices[a]][self.city_indices[b]])

    def exhaustive(self, n_cities: int=6) -> tuple:
        cities = list(self.city_indices.keys())[:n_cities]
        best_path = (None, None)
        for perm in permutations(cities[1:]):
            path = [(cities[0], perm[0])] + [perm[i:i + 2] for i in range(len(perm) - 1)] + [(perm[-1], cities[0])]
            total_distance = sum([self.distance(a, b) for a, b in path])
            if best_path[1] is None or total_distance < best_path[1]:
                best_path = (path, total_distance)
        return best_path

    def hill_climb(self, n_cities: int=6, runs: int=20, iterations: int=5000) -> tuple:
        cities = list(self.city_indices.keys())[:n_cities]
        best_path = (None, -1)
        all_dist = []
        worst_path = 0
        for attempt in range(runs):
            perm = np.random.permutation(cities)
            current_path = (None, -1)
            for _ in range(iterations):
                swap = random.sample(range(0, len(perm)), 2)
                perm[swap[0]], perm[swap[1]] = perm[swap[1]], perm[swap[0]]
                path = [(perm[i], perm[i + 1]) for i in range(len(perm) - 1)] + [(perm[-1], perm[0])]
                total_distance = np.sum([self.distance(a, b) for a, b in path])
                if current_path[1] == -1 or total_distance < current_path[1]:
                    current_path = (path, total_distance)

            if best_path[1] == -1 or current_path[1] < best_path[1]:
                best_path = current_path
            if current_path[1] > worst_path:
                worst_path = current_path[1]
            all_dist.append(current_path[1])

        self.info["hill"] = {
            "best": best_path[1],
            "worst": worst_path,
            "mean": np.mean(all_dist),
            "std": np.std(all_dist)
        }
        return best_path


if __name__ == '__main__':
    t = Tsp()
    t.run("hillclimb")
