#PSO Particle swarm optimization
import timeit
import random
import optproblems
import optproblems.cec2005


class PSO:
    """
    PSO algorithm with informants implementation, using CEC 2005 benchmark set for evaluation.
    The algorithm is based on Sean Luke (2013) Essentials of Metaheuristics book.
    """
    def __init__(self, problem, problemdimension, bounds, bounded, swarmsize, numberofinformantsperparticle, α, β, γ, δ, e):
        """
        Initializes a PSO algorithm.

        :param problem: Problem from CEC 2005 problem collection, see here:
        https://ls11-www.cs.tu-dortmund.de/people/swessing/optproblems/doc/cec2005.html
        :param problemdimension: Dimension of the problem. Particle position dimension.
        :param bounds: The corresponding bounds for the test function. Must respect the test function's defined range,
        see here: https://ls11-www.cs.tu-dortmund.de/people/swessing/optproblems/doc/cec2005.html#test-problems,
        this is used to initialize particles.
        :param bounded: Is the problem bounded by bounds or are they just used to in itialise our PSO algorithm.
        Must respect the test function's defined range, see here:
        https://ls11-www.cs.tu-dortmund.de/people/swessing/optproblems/doc/cec2005.html#test-problems,
        :param swarmsize: Number of particles in the swarm.
        :param numberofinformantsperparticle: Number of informants per particle.
        :param α: How much of the particles original velocity is retained when updating velocity.
        :param β: how much of the particles personal best is mixed in when updating velocity.
        :param γ: how much of the particles informants’ best is mixed in when updating velocity.
        :param δ: how much of the global best is mixed in when updating velocity.
        :param e: how fast the particles moves.
        """
        self.bounded = bounded
        self.numberofinformant = numberofinformantsperparticle
        self.problemdimension = problemdimension
        self.problem = problem
        self.bias = self.problem.bias
        self.test_function = self.problem(self.problemdimension)
        self.bounds = bounds
        self.swarmsize = swarmsize
        self.swarm = [self.Particle(self) for _ in range(self.swarmsize)]
        self.α = α
        self.β = β
        self.γ = γ
        self.δ = δ
        self.e = e

    class Particle():
        """
        Object of a candidate solution. A set of particles is a swarm.
        """
        def __init__(self, PSO):
            """
            Initializes a particle with a random position and velocity respecting the problem bounds and dimensions.
            """
            self.position = PSO.generaterandomposition()
            self.velocity = PSO.generaterandomvelocity()
            self.bestposition = self.position
            self.fitness = 100000
            self.bestfitness = 100000
            self.bestinformantfitness = 100000
            self.bestinformantposition = self.position
            self.informants = PSO.generaterandomindexinformant()

        def Determinebestinformant(self, swarm):
            """
              Determines which of the informants of the particle is the best according to their fitness, this includes the particle herself.
              :param swarm: Swarm.
            """
            informants = PSO.getinformant(self, self.informants, swarm)
            for informant in informants:
                if (informant.fitness < self.bestinformantfitness):
                    self.bestinformantfitness = informant.fitness
                    self.bestinformantposition = informant.position
            if(self.bestfitness < self.bestinformantfitness):
                self.bestinformantfitness = self.bestfitness
                self.bestinformantposition = self.bestposition



    def AssessFitness(self, position):
        """
        Gives the fitness of the particler by passing down the position of the particle to the test function.
        :param position: List of problemdimension values.
        :return: Fitness value evaluated by the test function.
        """
        individual = optproblems.base.Individual(position)
        self.test_function.evaluate(individual)
        return individual.objective_values


    def generaterandomposition(self):
        """
        Creates a random position from uniform distribution insides the problem bounds.
        :return: Random position.
        """
        Pos = []
        for pos in range(self.problemdimension):
            Pos.append(random.uniform(self.bounds[0], self.bounds[1]))
        return(Pos)

    def generaterandomvelocity(self):
        """
        Creates a random velocity from uniform distribution insides the problem bounds.
        :return: Random velocity.
        """
        Vel = []
        for pos in range(self.problemdimension):
            pos1 = random.uniform(self.bounds[0], self.bounds[1])
            pos2 = random.uniform(self.bounds[0], self.bounds[1])
            Vel.append((pos1+pos2)/10)
        return(Vel)

    def generaterandomindexinformant(self):
        """
        Creates a random index of number of informant size of int value inferior to the swarm size.
        :return: An index allowing to access other particles in the swarm.
        """
        Informant = []
        for i in range(self.numberofinformant):
            Informant.append(random.randint(0,self.swarmsize-1))
        return(Informant)

    def getinformant(self, informantsindex, swarm):
        """
        Return a list of particle accessed using an informant index.
        :return: A list of particle.
        """
        Informant = []
        for i in informantsindex:
            Informant.append(swarm[i])
        return (Informant)

    def run(self, maxnumberofiterations, precisionwanted):
        """
        Run the PSO, stops when the difference between the best fitness found and the optimal best reaches the precision wanted
        or when the maximum number of authorized iterations is reached.
        :param precisionwanted: Precision wanted.
        :param maxnumberofiterations: Max number of iterations.
        :return: The particle with the lowest (best) fitness, if the precisionwanted has been reached and the process time of the algorithm.
        """
        start = timeit.default_timer()
        init = True
        Fitnessb =  self.bias + precisionwanted + 1
        i = 0
        while((self.bias + precisionwanted < Fitnessb) and i<maxnumberofiterations):
            i += 1
            for p in self.swarm:
                p.fitness = self.AssessFitness(p.position)
                if(p.fitness < p.bestfitness):
                    p.bestfitness = p.fitness
                    p.bestposition = p.position
                if(p.fitness < Fitnessb or (init == True)):
                    Fitnessb = p.fitness
                    best = p
                    init = False
                p.Determinebestinformant(self.swarm)
            for p in self.swarm:
                xstar = p.bestposition #previous fittest location of x
                xplus = p.bestinformantposition #xplus = previous fittest location of informants of x informants part not ready yet
                xtot = best.position  #previous fittest location overall
                for dimension in range(self.problemdimension):
                    b = random.uniform(0,self.β)
                    c = random.uniform(0,self.γ)
                    d = random.uniform(0,self.δ)
                    #Velocity update
                    p.velocity[dimension] = p.velocity[dimension]*self.α + b*(xstar[dimension] - p.position[dimension]) + c*(xplus[dimension] - p.position[dimension])+ d*(xtot[dimension] - p.position[dimension])
            for p in self.swarm:
                for dimension in range(self.problemdimension):
                    #Displacement of the particles
                    if(self.bounded):
                        if(p.position[dimension]+self.e*p.velocity[dimension] > self.bounds[1]):
                            p.position[dimension] = self.bounds[1]
                        elif(p.position[dimension]+self.e*p.velocity[dimension] < self.bounds[0]):
                            p.position[dimension] = self.bounds[0]
                        else:
                            p.position[dimension] = p.position[dimension] + self.e * p.velocity[dimension]
                    else:
                        p.position[dimension] = p.position[dimension]+self.e*p.velocity[dimension]
        if(Fitnessb > self.bias + precisionwanted):
            result = False
        else:

            result = True
        time = timeit.default_timer()-start
        return(best, time, result)






