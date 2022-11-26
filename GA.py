import sys
from copy import deepcopy

import optproblems.cec2005
import optproblems
import numpy as np
import random


class GA:
    def __init__(self, test_function, n_dimension, bounds, pop_size, num_generations=10000, t=2,
                 crossover_percentage=0.95, mutation_rate=0.8, decreasing_mutation_rate=True, elite=0):
        # ToDO:Just for testing purpose
        np.random.seed(42)
        random.seed(42)

        if elite >= pop_size:
            sys.exit(f"Warning! Number of elite [{elite}] must be less than the population size [{pop_size}]\n"
                     f"Program terminated!")

        self.test_function = test_function
        self.n_dimension = n_dimension  # Length of array per sample. The individual's length
        self.bounds = bounds  # The domain space of the test function
        self.num_generations = num_generations  # Number of generations to run
        self.pop_size = pop_size
        self.population = [self.Individual(GA=self) for _ in range(self.pop_size)]
        self.t = t  # Tournament size for selection
        self.crossover_percentage = crossover_percentage
        self.mutation_rate = mutation_rate
        self.decreasing_mutation_rate = decreasing_mutation_rate
        if self.decreasing_mutation_rate:
            self.progress = 0
        self.elite = None if elite <= 0 else elite

    class Individual:
        def __init__(self, GA=None, individual=None):
            self.fitness = 0
            if individual is None:
                self.individual = GA.generate_random_individual()
                return
            self.individual = individual

    def generate_random_individual(self):
        return np.random.uniform(low=self.bounds[0], high=self.bounds[1], size=self.n_dimension)

    def assess_fitness(self, genome):
        individual = optproblems.base.Individual(genome)
        self.test_function.evaluate(individual)
        return individual.objective_values

    def tournament_selection(self):
        # Get random list of indexes
        list_of_i = random.sample(range(self.pop_size), self.t)

        # Initialize the best with the 1st index from the list
        best_per_tournament = self.population[list_of_i[0]]
        for i in range(1, len(list_of_i)):
            next_individual = self.population[list_of_i[i]]
            best_per_tournament = next_individual if next_individual.fitness < best_per_tournament.fitness else best_per_tournament

        return best_per_tournament.individual

    def one_point_crossover(self, parent_a, parent_b):
        cross_over_point = random.randrange(self.n_dimension)
        if cross_over_point == 0:
            return parent_a, parent_b

        crossed_a = np.append(parent_a[:cross_over_point], parent_b[cross_over_point:])
        crossed_b = np.append(parent_b[:cross_over_point], parent_a[cross_over_point:])

        return crossed_a, crossed_b

    def mutate(self, children):
        if self.decreasing_mutation_rate:
            mr = self.mutation_rate * (1 - self.progress)
        else:
            mr = self.mutation_rate

        for gene in range(len(children)):
            if np.random.random() < mr:
                # Random value to added to the gene
                random_value = np.random.uniform(low=self.bounds[0], high=self.bounds[1])
                children[gene] = random_value

        return children

    def run(self):
        for generation in range(self.num_generations):
            if self.decreasing_mutation_rate:
                self.progress = generation / self.num_generations

            for i in range(self.pop_size):
                # AssessFitness(Pi)
                fittness = self.assess_fitness(self.population[i].individual)
                self.population[i].fitness = fittness
                # ToDo: [DOC] add that this line from the pseudo code was tweaked
                # if best_fittness > fittness:
                #     best_fittness = fittness
                #     best_individual = self.population[i]

            new_pop = []
            self.population = sorted(self.population, key=lambda x: x.fitness)
            best_individual = self.population[0]

            if self.elite:
                new_pop = deepcopy(self.population[:self.elite])
                new_pop_to_generate = self.pop_size - self.elite
                i_for_breeding = int(new_pop_to_generate / 2) + new_pop_to_generate % 2
            else:
                i_for_breeding = int(self.pop_size / 2) + self.pop_size % 2

            for i in range(i_for_breeding):
                # Select With Replacement
                parent_a = self.tournament_selection()
                parent_b = self.tournament_selection()

                # Breeding
                if np.random.random() < self.crossover_percentage:
                    children_a, children_b = self.one_point_crossover(parent_a, parent_b)
                else:
                    children_a, children_b = parent_a, parent_b

                # Mutation
                new_pop = new_pop + [self.Individual(individual=self.mutate(children_a)),
                                     self.Individual(individual=self.mutate(children_b))]

            while len(new_pop) > self.pop_size:
                new_pop.pop()

            self.population = new_pop

            # ToDo: Remove print
            # print(best_individual.individual)
            print(best_individual.fitness)
        print("DONE")


if __name__ == '__main__':
    no_dimensions = 2
    bound = (-100, 100)
    pop_size = 30
    benchmark = optproblems.cec2005.F3(no_dimensions)
    opt = benchmark.get_optimal_solutions()

    algorithm = GA(benchmark, no_dimensions, bound, pop_size, elite=1)
    algorithm.run()
