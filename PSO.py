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
    def __init__(self, test_function, problemdimension, bounds, swarmsize, α, β, γ, δ, e):
        self.numberofinformant = 5
        self.test_function = test_function
        self.problemdimension = problemdimension
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
                if (abs(informant.fitness) < abs(self.bestinformantfitness)):
                    self.bestinformantfitness = informant.fitness
                    self.bestinformantposition = informant.position
            if(abs(self.bestfitness) < abs(self.bestinformantfitness)):
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
            Informant.append(random.randint(0,swarmsize-1))
        return(Informant)

    def getinformant(self, informantsindex, swarm):
        Informant = []
        for i in informantsindex:
            Informant.append(swarm[i])
        return (Informant)

    def run(self, numberofiterations, precisionwanted):
        start = time.process_time()
        init = True
        Fitnessb = precisionwanted + 1
        i = 0
        while(abs(Fitnessb) > precisionwanted and i<numberofiterations):
            i += 1
            for p in self.swarm:
                p.fitness = self.AssessFitness(p.position)
                if(abs(p.fitness) < abs(p.bestfitness)):
                    p.bestfitness = p.fitness
                    p.bestposition = p.position
                if((abs(p.fitness) < abs(Fitnessb)) or (init == True)):
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
                    if(p.position[dimension]+self.e*p.velocity[dimension] > self.bounds[1]):
                        p.position[dimension] = self.bounds[1]
                    elif(p.position[dimension]+self.e*p.velocity[dimension] < self.bounds[0]):
                        p.position[dimension] = self.bounds[0]
                    else:
                        p.position[dimension] = p.position[dimension]+self.e*p.velocity[dimension]
        StringResult = ""
        StringResult += "Test function: "+str(self.test_function)+", "
        StringResult += "Dimension: "+ str(self.problemdimension)+", "
        StringResult += "Number of iteration: "+ str(i)+", "
        StringResult += "Fitness reached: "+ str(Fitnessb)+", "
        StringResult += "Process time: "+ str(time.process_time()-start)+", "
        if(abs(Fitnessb) > precisionwanted):
            StringResult += "Failed try with other parameters"
            result = False
        else:
            StringResult += "Success"
            result = True
        return(StringResult, result)


#Testing
if __name__ == '__main__':
    DimensionList = [2,10,30,50]

    BenchmarkList = []
    BoundsList = []
    #Problem with no bounds are not included
    TestProblemList = [optproblems.cec2005.F1,optproblems.cec2005.F2,optproblems.cec2005.F3,optproblems.cec2005.F4,optproblems.cec2005.F5,optproblems.cec2005.F6,optproblems.cec2005.F8,optproblems.cec2005.F9,optproblems.cec2005.F10,optproblems.cec2005.F11,optproblems.cec2005.F12,optproblems.cec2005.F13,optproblems.cec2005.F14,optproblems.cec2005.F15,optproblems.cec2005.F16,optproblems.cec2005.F17,optproblems.cec2005.F18,optproblems.cec2005.F19,optproblems.cec2005.F20,optproblems.cec2005.F21,optproblems.cec2005.F22,optproblems.cec2005.F23,optproblems.cec2005.F24]
    for dimension in DimensionList:
        i = 0
        for testproblem in TestProblemList:
            i += 1
            BenchmarkList.append(testproblem(dimension))
            if(i < 7 or i == 13):
                BoundsList.append([-100,100])
            if(i == 7):
                BoundsList.append([-32, 32])
            if(i > 13 or i == 8 or i == 9):
                BoundsList.append([-5, 5])
            if (i == 10):
                BoundsList.append([-0.5, 0.5])
            if (i == 11):
                BoundsList.append([-math.pi,math.pi])
            if (i == 12):
                BoundsList.append([-3,1])

    #HyperParameters
    swarmsize = 2000
    α = 0.5 #how much of the original velocity is retained
    β = 0.4 #how much of the personal best is mixed in. If β is large, particles tend to move more towards their own personal bests rather than towards global bests. This breaks the swarm into a lot of separate hill-climbers rather than a joint searcher.
    γ = 0.6 #how much of the informants’ best is mixed in. The effect here may be a mid-ground between β and δ. The number of informants is also a factor (assuming they’re picked at random): more informants is more like the global best and less like the particle’s local best.
    δ = 0 #how much of the global best is mixed in. If δ is large, particles tend to move more towards the best known region. This converts the algorithm into one large hill-climber rather than separate hill-climbers. Perhaps because this threatens to make the system highly exploitative, δ is often set to 0 in modern implementations
    e = 1 #e how fast the particle moves. If e is large, the particles make big jumps towards the better areas— and can jump over them by accident. Thus a big e allows the system to move quickly to best-known regions, but makes it hard to do fine-grained optimization. Just like in hill-climbing. Most commonly, e is set to 1.
    numberofiterations = 100
    precisionwanted = 0.1

    #Create algorithms
    PSOList = []
    for i in range(len(BenchmarkList)):
        if i < 24:
            PSOList.append(PSO(BenchmarkList[i], DimensionList[0], BoundsList[i], swarmsize, α, β, γ, δ, e))
        elif i < 47:
            PSOList.append(PSO(BenchmarkList[i], DimensionList[1], BoundsList[i], swarmsize, α, β, γ, δ, e))
        elif i < 70:
            PSOList.append(PSO(BenchmarkList[i], DimensionList[2], BoundsList[i], swarmsize, α, β, γ, δ, e))
        else:
            PSOList.append(PSO(BenchmarkList[i], DimensionList[3], BoundsList[i], swarmsize, α, β, γ, δ, e))


    #Try each test function of optproblems.cec2005 with dimension 2 10 and 50 and appropriate bounds
    Results = []
    Resutlsbool = []
    for i in PSOList:
        result = i.run(numberofiterations, precisionwanted)
        print(result[0])
        Results.append(result[0])
        Resutlsbool.append(result[1])

    Percentagesuccess = (sum(Resutlsbool)/len(PSOList))*100
    print(Percentagesuccess)

    #Lets see with an increased number of iterations and an increased particle speed

    # HyperParameters
    swarmsize = 2000
    α = 0.5  # how much of the original velocity is retained
    β = 0.4  # how much of the personal best is mixed in. If β is large, particles tend to move more towards their own personal bests rather than towards global bests. This breaks the swarm into a lot of separate hill-climbers rather than a joint searcher.
    γ = 0.6  # how much of the informants’ best is mixed in. The effect here may be a mid-ground between β and δ. The number of informants is also a factor (assuming they’re picked at random): more informants is more like the global best and less like the particle’s local best.
    δ = 0  # how much of the global best is mixed in. If δ is large, particles tend to move more towards the best known region. This converts the algorithm into one large hill-climber rather than separate hill-climbers. Perhaps because this threatens to make the system highly exploitative, δ is often set to 0 in modern implementations
    e = 2  # e how fast the particle moves. If e is large, the particles make big jumps towards the better areas— and can jump over them by accident. Thus a big e allows the system to move quickly to best-known regions, but makes it hard to do fine-grained optimization. Just like in hill-climbing. Most commonly, e is set to 1.
    numberofiterations = 300
    precisionwanted = 0.1

    # Create algorithms
    PSOList = []
    for i in range(len(BenchmarkList)):
        if i < 24:
            PSOList.append(PSO(BenchmarkList[i], DimensionList[0], BoundsList[i], swarmsize, α, β, γ, δ, e))
        elif i < 47:
            PSOList.append(PSO(BenchmarkList[i], DimensionList[1], BoundsList[i], swarmsize, α, β, γ, δ, e))
        elif i < 70:
            PSOList.append(PSO(BenchmarkList[i], DimensionList[2], BoundsList[i], swarmsize, α, β, γ, δ, e))
        else:
            PSOList.append(PSO(BenchmarkList[i], DimensionList[3], BoundsList[i], swarmsize, α, β, γ, δ, e))



    #Try each test function of optproblems.cec2005 with dimension 2 10 30 and 50 and appropriate bounds
    Results = []
    Resutlsbool = []
    for i in PSOList:
        result = i.run(numberofiterations, precisionwanted)
        print(result[0])
        Results.append(result[0])
        Resutlsbool.append(result[1])

    Percentagesuccess2 = (sum(Resutlsbool)/len(PSOList))*100
    print(Percentagesuccess2)

    print("Success percentage for first experiment ",Percentagesuccess," and for second one ",Percentagesuccess2)

    #To experiment more on hyperparameters, I recommend not using every problem and every dimension but look specifically at the evolution of one problem results to reduce computation time
    #Studying precise example also allows to analyse why a certain set of hyperparameters are adapted to this problem according to the landscape of this problem

