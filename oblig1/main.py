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
        self.index_cities = {idx: city.lower() for idx, city in enumerate(self.data[0])}
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
        elif mode == "genetic":
            f = self.genetic
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
            current_path = (None, -1)
            perm = np.random.permutation(cities)
            while 1:
                best_neighbour = (None, -1, None)
                home_perm = perm.copy()
                for s1, s2 in combinations(range(len(home_perm)), 2):
                    perm = list(home_perm.copy())
                    perm[s1], perm[s2] = perm[s2], perm[s1]
                    path = [(perm[i], perm[i + 1]) for i in range(-1, len(perm) - 1)]
                    total_distance = np.sum([self.distance(a, b) for a, b in path])
                    if best_neighbour[1] == -1 or total_distance < best_neighbour[1]:
                        best_neighbour = (path, total_distance, perm.copy())

                perm = best_neighbour[2]
                if current_path[1] == -1 or best_neighbour[1] < current_path[1]:
                    current_path = (best_neighbour[0], best_neighbour[1])
                else:
                    break

            paths.append(current_path)

        sorted_paths = sorted(paths, key=lambda x: x[1])
        self.info["hillclimb"] = {
            "distance_worst": sorted_paths[-1][1],
            "distance_best": sorted_paths[0][1],
            "distance_mean": np.mean([p[1] for p in sorted_paths]),
            "distance_std": np.std([p[1] for p in sorted_paths]),
        }
        return sorted_paths[0]

    def genetic(self, n_cities, generations=100, population_cnt=1000, m_rate=0.03):

        def recombine(parents):
            offsprings = []
            for parent_a, parent_b in parents:
                segment_size = np.random.randint(1, len(parent_a) - 1)
                segment_start = np.random.randint(len(parent_a) - segment_size)

                def crossover(p1, p2):
                    offspring = [-1 for _ in range(len(p1))]
                    offspring[segment_start:segment_start + segment_size + 1] = \
                        p1[segment_start:segment_start + segment_size + 1]
                    o_idx = 0
                    for i, nr in enumerate(p2):
                        while o_idx < len(offspring) and offspring[o_idx] > -1:
                            o_idx += 1
                        if nr not in offspring:
                            offspring[o_idx] = nr
                            o_idx += 1
                    return offspring

                offsprings += [crossover(parent_a, parent_b), crossover(parent_b, parent_a)]
            return offsprings

        def mutate(offspring):
            new_offspring = offspring.copy()
            for idx, genotype in enumerate(new_offspring):
                check = random.uniform(0, 1)
                if check < m_rate:
                    new_genotype = list(genotype).copy()
                    s1, s2 = random.sample(range(0, len(new_genotype)), 2)
                    new_genotype[s1], new_genotype[s2] = new_genotype[s2], new_genotype[s1]
                    offspring[idx] = new_genotype
            return new_offspring

        def fitness(genotype):
            path = [(genotype[i], genotype[i + 1]) for i in range(-1, len(genotype) - 1)]
            distance = np.sum([self.distance(self.index_cities[a], self.index_cities[b]) for a, b in path])
            return 1 / distance

        def evolve(gens, pop_cnt):
            if pop_cnt % 2 != 0:
                pop_cnt -= 1
            generation_fitness = []
            cities = list(self.city_indices.keys())[:n_cities]
            population = [
                [self.city_indices[city.lower()] for city in np.random.permutation(cities)]
                for _ in range(pop_cnt)
            ]
            for gen_nr in range(gens):
                parents = zip(population[:pop_cnt // 2], population[pop_cnt // 2:])
                offspring = recombine(parents)
                population += mutate(offspring)
                total_fitness = sum([fitness(x) for x in population])
                generation_fitness.append(total_fitness)
                population = sorted(population, key=fitness, reverse=True)[:pop_cnt]
            path = [
                (self.index_cities[population[0][i]], self.index_cities[population[0][i + 1]])
                for i in range(-1, len(population[0]) - 1)
            ]
            distance = np.sum([self.distance(a, b) for a, b in path])
            return path, distance

        return evolve(gens=generations, pop_cnt=population_cnt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, choices=("exhaustive", "hillclimb", "report"))
    parser.add_argument("-n", "--number-of-cities", type=int, default=6)
    args = parser.parse_args()
    t = Tsp()
    if args.mode != "report":
        t.run(args.mode, n=args.number_of_cities)
    else:
        t.run("genetic", n=24)
        exit()
        for city_count in [10]:
            t.run("hillclimb", n=city_count, run_cnt=1)
            print(t.info["hillclimb"])
        t.run("exhaustive", n=10)
