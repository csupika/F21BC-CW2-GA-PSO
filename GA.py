import optproblems.cec2005
import optproblems
import numpy as np
import random


# ToDo: Use HashMap (maybe?) --> If so, then key is the population list as a tuple.

class GA:
    def __init__(self, test_function, n_dimension, bounds, pop_size, limit=100, mutation=0.1, t=2):
        # Just for testing purpose
        np.random.seed(42)

        self.test_function = test_function
        self.n_dimension = n_dimension  # Length of array per sample. The individual's length
        self.bounds = bounds  # The domain space of the test function
        self.limit = limit  # Number of generations to run
        self.mutation = mutation
        self.pop_size = pop_size
        self.population = [self.Individual(self) for _ in range(self.pop_size)]
        self.t = t  # Tournament size for selection

    class Individual:
        def __init__(self, GA):
            self.fitness = 0
            self.individual = GA.generate_random_individual()

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
            next = self.population[list_of_i[i]]
            best_per_tournament = next if next.fitness < best_per_tournament.fitness else best_per_tournament

        return best_per_tournament.individual

    def one_point_crossover(self, parent_a, parent_b):
        cross_over_point = random.randrange(self.n_dimension)
        if cross_over_point == 0:
            return parent_a, parent_b

        crossed_a = np.append(parent_a[:cross_over_point], parent_b[cross_over_point:])
        crossed_b = np.append(parent_b[:cross_over_point], parent_a[cross_over_point:])

        return crossed_a, crossed_b

    def run(self):
        for x in range(self.limit):
            # Initialize the best fitness with the population's first elements fitness
            self.population[0].fitness = self.assess_fitness(self.population[0].individual)
            best_fittness = self.population[0].fitness

            for i in range(1, self.pop_size):
                # AssessFitness(Pi)
                fittness = self.assess_fitness(self.population[i].individual)
                self.population[i].fitness = fittness
                if best_fittness > fittness:
                    best_fittness = fittness
                    best_genome = self.population[i]

            new_pop = []
            for i in range(int(self.pop_size / 2)):
                # SelectWithReplacement
                parent_a = self.tournament_selection()
                parent_b = self.tournament_selection()

                children_a, children_b = self.one_point_crossover(parent_a, parent_b)
                print("done")


if __name__ == '__main__':
    NO_dimensions = 10
    bound = (-5, 5)
    pop_size = 50
    benchmark_f1 = optproblems.cec2005.F6(NO_dimensions)

    algorithm1 = GA(benchmark_f1, NO_dimensions, bound, pop_size)
    algorithm1.run()
