import csv
from itertools import permutations
import time
import random
import numpy as np
import argparse


class Tsp:

    info = {}

    def __init__(self):
        self.data = self.read_data("european_cities.csv")
        self.city_indices = {city.lower(): idx for idx, city in enumerate(self.data[0])}
        self.data.pop(0)

    def run(self, mode, n: int=10, **kwargs):
        extra_params = {}
        if mode == "exhaustive":
            f = self.exhaustive
        elif mode == "hillclimb":
            f = self.hill_climb
            extra_params = {
                "n_cnt": kwargs.get("n_cnt", 5),
                "max_bad": kwargs.get("max_bad", 500)
            }
        else:
            return

        t0 = time.time()
        trip, distance = f(n_cities=n, **extra_params)
        t1 = time.time() - t0
        print("Path: {} > {}\nDistance: {:.3f}\nTime: {:.3f}\n".format(
            " > ".join([p[0].title() for p in trip]),
            trip[-1][1].title(),
            distance,
            t1
        ))


    @staticmethod
    def read_data(f_name: str) -> list:
        with open(f_name, "r") as f:
            return list(csv.reader(f, delimiter=';'))

    def distance(self, a: str, b: str) -> float:
        return float(self.data[self.city_indices[a]][self.city_indices[b]])

    def exhaustive(self, n_cities: int=6, **kwargs) -> tuple:
        cities = list(self.city_indices.keys())[:n_cities]
        best_path = (None, None)
        for perm in permutations(cities):
            path = [(perm[i], perm[i + 1]) for i in range(-1, len(perm)-1)]
            total_distance = sum([self.distance(a, b) for a, b in path])
            if best_path[1] is None or total_distance < best_path[1]:
                best_path = (path, total_distance)
        return best_path

    def hill_climb(self, n_cities: int=6, max_bad: int=100, n_cnt=10) -> tuple:
        cities = list(self.city_indices.keys())[:n_cities]
        perm = np.random.permutation(cities)
        current_path = (None, -1)
        bad_cnt = 0
        while bad_cnt < max_bad:
            best_neighbour = (None, -1, None)
            # Check n_cnt neighbours for the best one
            for _ in range(n_cnt):
                swap = random.sample(range(0, len(perm)), 2)
                perm[swap[0]], perm[swap[1]] = perm[swap[1]], perm[swap[0]]
                path = [(perm[i], perm[i + 1]) for i in range(-1, len(perm) - 1)]
                total_distance = np.sum([self.distance(a, b) for a, b in path])
                if best_neighbour[1] == -1 or total_distance < best_neighbour[1]:
                    best_neighbour = (path, total_distance, perm.copy())
            # Keep the best neighbour
            perm = best_neighbour[2]
            # Check if that neighbour is better than the current best
            # Keep checking if not
            if current_path[1] == -1 or best_neighbour[1] < current_path[1]:
                current_path = (best_neighbour[0], best_neighbour[1])
                bad_cnt = 0
            else:
                bad_cnt += 1

        return current_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, choices=("exhaustive", "hillclimb", "report"))
    parser.add_argument("-n", "--number-of-cities", type=int, default=6)
    args = parser.parse_args()
    t = Tsp()
    if args.mode != "report":
        t.run(args.mode, n=args.number_of_cities)
    else:
        pass
