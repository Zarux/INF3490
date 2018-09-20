import csv
from itertools import permutations, combinations
import time
import random
import numpy as np
import argparse
from tsp_plot import TSPPlot


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
                "run_cnt": kwargs.get("run_cnt", 20),
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

    def hill_climb(self, n_cities: int=6, run_cnt=20) -> tuple:
        cities = list(self.city_indices.keys())[:n_cities]
        paths = []
        for _ in range(run_cnt):
            i = 0
            pp = TSPPlot(cities, "joakikt", "QjDLDPCNHQBUZypnh9Xb")
            current_path = (None, -1)
            perm = np.random.permutation(cities)
            all_perms = []
            while 1:
                best_neighbour = (None, -1, None)
                home_perm = perm.copy()
                for s1, s2 in combinations(range(len(home_perm)), 2):
                    perm = list(home_perm.copy())
                    perm[s1], perm[s2] = perm[s2], perm[s1]
                    path = [(perm[i], perm[i + 1]) for i in range(-1, len(perm) - 1)]
                    total_distance = np.sum([self.distance(a, b) for a, b in path])
                    all_perms.append((perm.copy(), total_distance))
                    if best_neighbour[1] == -1 or total_distance < best_neighbour[1]:
                        best_neighbour = (path, total_distance, perm.copy())

                perm = best_neighbour[2]
                all_perms.append((perm.copy(), best_neighbour[1]))
                if current_path[1] == -1 or best_neighbour[1] < current_path[1]:
                    current_path = (best_neighbour[0], best_neighbour[1])
                else:
                    break

            paths.append(current_path)
            print(current_path[1], len(all_perms))
            if current_path[1] < 12500:
                last_perm = None
                for p, d in all_perms:
                    pp.plot(list(p.copy()) + [p[0]], "{0:010d}.png".format(i),
                            title="Distance: {:.2f}".format(d))
                    i += 1
                    last_perm = p
                for _ in range(120):
                    pp.plot(list(last_perm) + [last_perm[0]], "{0:010d}.png".format(i),
                            title="Distance: {:.2f}".format(current_path[1]))
                    i += 1
                break
        sorted_paths = sorted(paths, key=lambda x: x[1])
        self.info["hillclimb"] = {
            "distance_worst": sorted_paths[-1][1],
            "distance_best": sorted_paths[0][1],
            "distance_mean": np.mean([p[1] for p in sorted_paths]),
            "distance_std": np.std([p[1] for p in sorted_paths]),
        }
        return sorted_paths[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, choices=("exhaustive", "hillclimb", "report"))
    parser.add_argument("-n", "--number-of-cities", type=int, default=6)
    args = parser.parse_args()
    t = Tsp()
    if args.mode != "report":
        t.run(args.mode, n=args.number_of_cities)
    else:
        for city_count in [24]:
            t.run("hillclimb", n=city_count, run_cnt=1000)
            print(t.info["hillclimb"])
        exit()
        t.run("exhaustive", n=10)
