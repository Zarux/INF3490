import argparse
import csv
import random
import time
from itertools import permutations, combinations
import matplotlib.patches as mpatches
import numpy as np
import matplotlib.pyplot as plt


class Tsp:

    info = {}

    def __init__(self):
        self.data = self.read_data("european_cities.csv")
        self.city_indices = {city.lower(): idx for idx, city in enumerate(self.data[0])}
        self.index_cities = {idx: city.lower() for idx, city in enumerate(self.data[0])}
        self.data.pop(0)

    def run(self, mode, n, **kwargs):
        if mode == "exhaustive":
            f = self.exhaustive
        elif mode == "hillclimb":
            f = self.hill_climb
        elif mode == "genetic":
            f = self.genetic
        else:
            return

        t0 = time.time()
        trip, distance = f(n_cities=n, **kwargs)
        t1 = time.time() - t0
        print("Path: {} > {}\nDistance: {:.3f}\nTime: {:.3f}\n".format(
            " > ".join([p[0].title() for p in trip]),
            trip[-1][1].title(),
            distance,
            t1
        ))
        for k, v in self.info.get(mode, {}).items():
            if not isinstance(v, list):
                print(k, v)

    @staticmethod
    def read_data(f_name):
        with open(f_name, "r") as f:
            return list(csv.reader(f, delimiter=';'))

    def distance(self, a, b):
        return float(self.data[self.city_indices[a]][self.city_indices[b]])

    def exhaustive(self, n_cities=6, **kwargs):
        cities = list(self.city_indices.keys())[:n_cities]
        best_path = (None, None)
        for perm in permutations(cities):
            path = [(perm[i], perm[i + 1]) for i in range(-1, len(perm)-1)]
            total_distance = sum([self.distance(a, b) for a, b in path])
            if best_path[1] is None or total_distance < best_path[1]:
                best_path = (path, total_distance)
        return best_path

    def hill_climb(self, n_cities=6, run_cnt=20, **kwargs):
        cities = list(self.city_indices.keys())[:n_cities]
        paths = []
        for _ in range(run_cnt):
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

    def genetic(self, n_cities, generations=100, population_cnt=100, m_rate=0.1, run_cnt=1, **kwargs):

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
                population = sorted(population, key=fitness, reverse=True)[:pop_cnt]
                generation_fitness.append(fitness(population[0]))

            path = [
                (self.index_cities[population[0][i]], self.index_cities[population[0][i + 1]])
                for i in range(-1, len(population[0]) - 1)
            ]
            distance = np.sum([self.distance(a, b) for a, b in path])
            return path, distance, generation_fitness

        paths = []
        for _ in range(run_cnt):
            paths.append(evolve(gens=generations, pop_cnt=population_cnt))

        sorted_paths = sorted(paths, key=lambda x: x[1])
        self.info["genetic"] = {
            "distance_worst": sorted_paths[-1][1],
            "distance_best": sorted_paths[0][1],
            "distance_mean": np.mean([p[1] for p in sorted_paths]),
            "distance_std": np.std([p[1] for p in sorted_paths]),
            "generation_fitness": sorted_paths[0][2]
        }

        return sorted_paths[0][0:2]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, choices=("exhaustive", "hillclimb", "genetic", "report"))
    parser.add_argument("-n", "--number-of-cities", type=int, default=6)
    parser.add_argument("-m", "--mutation-rate", type=float, default=0.05)
    parser.add_argument("-g", "--generations", type=int, default=100)
    parser.add_argument("-p", "--population-count", type=int, default=200)
    parser.add_argument("-r", "--run-count", type=int, default=1)
    args = parser.parse_args()
    t = Tsp()
    if args.mode != "report":
        t.run(
            args.mode,
            n=args.number_of_cities,
            population_cnt=args.population_count,
            generations=args.generations,
            m_rate=args.mutation_rate,
            run_cnt=args.run_count
        )
    else:
        g_fit = []
        t_p = [50, 100, 200]
        for p in t_p:
            print("POPULATION: ", p)
            t.run(
                "genetic",
                n=24,
                population_cnt=p,
                generations=args.generations,
                m_rate=args.mutation_rate,
                run_cnt=args.run_count
            )
            g_fit.append(t.info["genetic"]["generation_fitness"])

        r_legend = mpatches.Patch(color="red", label="Population: {}".format(t_p[0]))
        b_legend = mpatches.Patch(color="blue", label="Population: {}".format(t_p[1]))
        g_legend = mpatches.Patch(color="green", label="Population: {}".format(t_p[2]))
        plt.plot(
            range(len(g_fit[0])), g_fit[0], "r",
            range(len(g_fit[1])), g_fit[1], "b",
            range(len(g_fit[2])), g_fit[2], "g"
        )
        plt.legend(handles=[r_legend, b_legend, g_legend])
        plt.show()
        for city_count in [10, 24]:
            t.run("hillclimb", n=city_count, run_cnt=20)

        t.run("exhaustive", n=10)
