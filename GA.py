import sys
import timeit
from copy import deepcopy
import optproblems.cec2005
import optproblems
import numpy as np
import random


class GA:
    """
    Genetic Algorithm with elitism, decreasing mutation rate and tournament selection using CEC 2005 benchmark set for
    evaluation. The algorithm is based on Sean Luke (2013) Essentials of Metaheuristics book.
    """
    def __init__(self, test_function: optproblems.TestProblem, n_dimension: int, bounds: list[int, int], pop_size: int,
                 num_generations=100, t=2, crossover_percentage=0.95, mutation_rate=0.8, decreasing_mutation_rate=True,
                 elite=0):
        """
        Initializes a Genetic Algorithm.

        :param test_function: Test function from CEC 2005 problem collection, see here:
        https://ls11-www.cs.tu-dortmund.de/people/swessing/optproblems/doc/cec2005.html
        :param n_dimension: Length of array per sample. The chromosome's size.
        :param bounds: The corresponding bounds for the test function. Must respect the test function's defined range,
        see here: https://ls11-www.cs.tu-dortmund.de/people/swessing/optproblems/doc/cec2005.html#test-problems
        :param pop_size: Set of candidate solutions per generation. Cannot be less than the number of elites.
        :param num_generations: Number of generations to run.
        :param t: Tournament size. Cannot be less than 1!
        :param crossover_percentage: The probability of a crossover to happen, which gives approximately the percentage
        when a crossover happens of two parents. Range: [0,1]
        :param mutation_rate: Probability to change a gene in the chromosome's chromosome. Range: [0,1]
        :param decreasing_mutation_rate: If True then the program slowly decreases the mutation rate per generations. In
        the first generation the mutation rate is unchanged (mutation rate * 1), at half-point it's halved
        (mutation rate * 0.5) and near 0 for the last generation.
        :param elite: If 0 then elitism is not used, else it defines the number of elite(s) to keep for the next
        generation. The number of elite must be less than the population size.
        """

        if elite >= pop_size:
            sys.exit(f"Warning! Number of elite [{elite}] must be less than the population size [{pop_size}]\n"
                     f"Program terminated!")
        if t < 1:
            sys.exit(f"Warning! Tournament size [{t}] cannot be less than 1!")

        self.test_function = test_function
        self.n_dimension = n_dimension  # Length of array per sample. The chromosome's length
        self.bounds = bounds  # The domain space of the test function
        self.num_generations = num_generations  # Number of generations to run
        self.pop_size = pop_size
        self.population = [self.Individual(GA=self) for _ in range(self.pop_size)]
        self.t = t  # Tournament size for selection
        self.crossover_percentage = crossover_percentage  # Probability for crossing over two parents.
        self.mutation_rate = mutation_rate  # Probability for mutating a gene in a chromosome.
        self.decreasing_mutation_rate = decreasing_mutation_rate
        if self.decreasing_mutation_rate:
            self.progress = 0  # The progress is the {current number of generation / total generation}. Used for
            # slowly decreasing the value of mutation rate.
        self.elite = None if elite <= 0 else elite  # If Elite is <= 0 then elitism is not used.

    class Individual:
        """
        Object of a candidate solution. A set of individuals are the population.
        """
        def __init__(self, GA=None, existing_chromosome=None):
            """
            Initializes an chromosome either with a random set of genes or with existing genes. If GA is provided,
            the chromosome will be generated with a random set of chromosome. Either GA or existing_chromosome must be
            provided.
            :param GA: Reference of the parent class GA. If provided the individuals chromosomes are randomly
            initialized.
            :param existing_chromosome: The chromosome to use for the chromosome's chromosome. This needs to be provided
            if we want to define the chromosome's chromosome.
            """
            if GA is None and existing_chromosome is None:
                sys.exit("Warning! Individual and GA parameter cannot be null. One of it has to be provided!")
            self.fitness = 0
            if existing_chromosome is None:
                self.chromosome = GA.generate_random_chromosome()
                return
            self.chromosome = existing_chromosome

    def generate_random_chromosome(self) -> np.ndarray:
        """
        Creates a random chromosome from uniform distribution. A chromosome is a set of genes.
        :return: Random chromosome.
        """
        return np.random.uniform(low=self.bounds[0], high=self.bounds[1], size=self.n_dimension)

    def assess_fitness(self, chromosome: np.ndarray) -> float:
        """
        Gives the fitness of the chromosome by passing down the chromosome to the test function.
        :param chromosome: Set of genes.
        :return: Fitness value evaluated by the test function.
        """
        individual = optproblems.base.Individual(chromosome)
        self.test_function.evaluate(individual)
        return individual.objective_values

    def tournament_selection(self) -> np.ndarray:
        """
        Draws random individuals from the population and returns the chromosome with the lowest (best) fitness. The
        tournament size is defined at initialization.
        :return: Chromosome with the best fitness.
        """
        # Get random list of indexes
        list_of_i = random.sample(range(self.pop_size), int(self.t))

        # Initialize the best with the 1st index from the list
        best_per_tournament = self.population[list_of_i[0]]
        for i in range(1, len(list_of_i)):
            next_individual = self.population[list_of_i[i]]
            best_per_tournament = next_individual if next_individual.fitness < best_per_tournament.fitness else best_per_tournament

        return best_per_tournament.chromosome

    def one_point_crossover(self, parent_a: np.ndarray, parent_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Crosses two parents chromosome's at a random point.
        :param parent_a: 1st parent chromosome to use for the crossover.
        :param parent_b: 2nd parent chromosome to use for the crossover.
        :return Two child chromosome with crossed chromosomes from the parents.
        """
        cross_over_point = random.randrange(self.n_dimension)
        if cross_over_point == 0:
            return parent_a, parent_b

        crossed_a = np.append(parent_a[:cross_over_point], parent_b[cross_over_point:])
        crossed_b = np.append(parent_b[:cross_over_point], parent_a[cross_over_point:])

        return crossed_a, crossed_b

    def mutate(self, chromosome: np.ndarray) -> np.ndarray:
        """
        Mutates a chromosome using mutation rate. For each gene in the chromosome draws a number in range [0, 1]. If the
        number is less than the mutation rate then it mutates the gene by replacing it with a new random gene within the
        bounds. If self decreasing mutation is true then
        :param chromosome: Set of genes.
        :return: The mutated chromosome.
        """
        if self.decreasing_mutation_rate:
            mr = self.mutation_rate * (1 - self.progress)
        else:
            mr = self.mutation_rate

        for gene in range(len(chromosome)):
            if np.random.random() < mr:
                # Random value to added to the gene
                random_value = np.random.uniform(low=self.bounds[0], high=self.bounds[1])
                chromosome[gene] = random_value

        return chromosome

    def run(self) -> tuple[Individual, float]:
        """
        Run the genetic algorithm.
        :return: The individual with the lowest (best) fitness and process time of the algorithm.
        """
        start_time = timeit.default_timer()
        for generation in range(self.num_generations):
            if self.decreasing_mutation_rate:  # Get the progress used for slowly decreasing the mutation rate
                self.progress = generation / self.num_generations

            for i in range(self.pop_size):
                # AssessFitness(Pi)
                fittness = self.assess_fitness(self.population[i].chromosome)
                self.population[i].fitness = fittness

            new_pop = []
            # Sort the population by fitness and get the best individual.
            self.population = sorted(self.population, key=lambda x: x.fitness)
            best_individual = self.population[0]

            # If this is the last generation then don't breed, just return the best individual and process time.
            if generation == self.num_generations - 1:
                end_time = timeit.default_timer()
                process_time = end_time - start_time
                return best_individual, process_time

            # If elitism is used then adjust the number of iterations for breeding.
            if self.elite:
                new_pop = deepcopy(self.population[:self.elite])  # Create copy to avoid issues with referencing
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
                new_pop = new_pop + [self.Individual(existing_chromosome=self.mutate(children_a)),
                                     self.Individual(existing_chromosome=self.mutate(children_b))]

            # Pop the last element from the list if any additional child was created due to the breeding or elitism.
            # Breeding can only create an even number of children or using elitism might result in creating an
            # additional child.
            while len(new_pop) > self.pop_size:
                new_pop.pop()

            self.population = new_pop


if __name__ == '__main__':
    no_dimensions = 2
    bound = [-100, 100]
    population = 30
    benchmark = optproblems.cec2005.F3(no_dimensions)
    opt = benchmark.get_optimal_solutions()
    benchmark.evaluate(opt[0])

    algorithm = GA(benchmark, no_dimensions, bound, population)
    result, time = algorithm.run()
    print(result.fitness, " ", time)
