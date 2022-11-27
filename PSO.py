#PSO Particle swarm optimization 3.5 in book:
import math
import random
import optproblems
import optproblems.cec2005
import time

#A particle consists of two parts:
#   Location in space equivalent of genotype in evolutionary algorithms
#   Velocity speed and direction at which the particle is traveling each timestep
#Each particle starts at a random location and with a random velocity vector, often computed by choosing two random points in the space
#and using half the vector from one to the other

#Each timestep we perform the following operations:
#   Assess the fitness of each particle and update the best-discovered locations if necessary.
#   Determine how to Mutate. For each particle ~x, we update its velocity vector ~v by adding in, to some degree, a vector pointing
#towards ~x∗, a vector pointing towards ~x+, and a vector pointing towards ~x!. These are augmented by a bit of random noise
#(different random values for each dimension).
#    Mutate each particle by moving it along its velocity vector.

#Code

class PSO:
    def __init__(self, problem, problemdimension, bounds, bounded, swarmsize, numberofinformantsperparticle, α, β, γ, δ, e):
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
        def __init__(self, PSO):
            self.position = PSO.generaterandomposition()
            self.velocity = PSO.generaterandomvelocity()
            self.bestposition = self.position
            self.fitness = 100000
            self.bestfitness = 100000
            self.bestinformantfitness = 100000
            self.bestinformantposition = self.position
            self.informants = PSO.generaterandomindexinformant()

        def Determinebestinformant(self, swarm):
            informants = PSO.getinformant(self, self.informants, swarm)
            for informant in informants:
                if (informant.fitness < self.bestinformantfitness):
                    self.bestinformantfitness = informant.fitness
                    self.bestinformantposition = informant.position
            if(self.bestfitness < self.bestinformantfitness):
                self.bestinformantfitness = self.bestfitness
                self.bestinformantposition = self.bestposition



    def AssessFitness(self, position):
        individual = optproblems.base.Individual(position)
        self.test_function.evaluate(individual)
        return individual.objective_values


    def generaterandomposition(self):
        Pos = []
        for pos in range(self.problemdimension):
            Pos.append(random.uniform(self.bounds[0], self.bounds[1]))
        return(Pos)

    def generaterandomvelocity(self):
        Vel = []
        for pos in range(self.problemdimension):
            pos1 = random.uniform(self.bounds[0], self.bounds[1])
            pos2 = random.uniform(self.bounds[0], self.bounds[1])
            Vel.append((pos1+pos2)/10)
        return(Vel)

    def generaterandomindexinformant(self):
        Informant = []
        for i in range(self.numberofinformant):
            Informant.append(random.randint(0,self.swarmsize-1))
        return(Informant)

    def getinformant(self, informantsindex, swarm):
        Informant = []
        for i in informantsindex:
            Informant.append(swarm[i])
        return (Informant)

    def run(self, numberofiterations, precisionwanted):
        start = time.process_time()
        init = True
        Fitnessb =  self.bias + precisionwanted + 1
        i = 0
        while((self.bias + precisionwanted < Fitnessb) and i<numberofiterations):
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
        StringResult = ""
        StringResult += "Test function: "+str(self.test_function)+", "
        StringResult += "Dimension: "+ str(self.problemdimension)+", "
        StringResult += "Number of iteration: "+ str(i)+", "
        StringResult += "Solution reached: "+ str(Fitnessb)+", "
        StringResult += "Global optimum: " + str(self.bias) + ", "
        StringResult += "Process time: "+ str(time.process_time()-start)+", "
        StringResult += "Parameter for best solution: "+ str(best.position)+", "
        if(Fitnessb > self.bias + precisionwanted):
            StringResult += "Failed try augmenting the number of iteration or change the hyperparameters"
            result = False
        else:
            StringResult += "Success"
            result = True
        return(StringResult, result)





    #Experiment with F12 in 10 dimensions

    #SwarmsizeList = [100,500,2000]
    #NumberofinformantsperparticleList = [2,6,10]
    #NumberofiterationsList = [20,50,100]
    #αList = [0.1,0.4]
    #βList = [0.4,0.8]
    #γList = [0.4,0.8]
    #δList = [0.4,0.8]
    #eList = [0.8,1.2]
    #for swarmsize in SwarmsizeList:
    #    for numberofinformantsperparticle in NumberofinformantsperparticleList:
    #        for numberofiterations in NumberofiterationsList:
    #            for α in αList:
    #                for β in βList:
    #                    for δ in δList:
    #                        for e in eList:
    #                            PSO = PSO(optproblems.cec2005.F12, 10, [-3,1], swarmsize, numberofinformantsperparticle, α, β, γ, δ, e)
    #                            result = PSO.run(numberofiterations, 0.1)
    #                            print(result[0])

    #To experiment more on hyperparameters, I recommend not using every problem and every dimension but look specifically at the evolution of one problem results to reduce computation time
    #Studying precise example also allows to analyse why a certain set of hyperparameters are adapted to this problem according to the landscape of this problem

