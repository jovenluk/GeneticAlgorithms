# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 12:25:52 2019

@author: Fernando García Varela

"""
# source: https://towardsdatascience.com/evolution-of-a-salesman-a-complete-genetic-algorithm-tutorial-for-python-6fe5d2b3ca35
import numpy as np, random, operator, pandas as pd, matplotlib.pyplot as plt

# =============================================================================
# ---- fINITIALVALUES
# =============================================================================
GENERATIONS = 200

# datos de una ciudad
# fGENOME
class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance(self, city):
        xDis = abs(self.x - city.x)
        yDis = abs(self.y - city.y)
        distance = np.sqrt((xDis ** 2) + (yDis ** 2))
        return distance

    def __repr__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"

# =============================================================================
# ---- fFITNESS
# =============================================================================
# calcula el fitness
class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness= 0.0

    def routeDistance(self):
        if self.distance ==0:
            pathDistance = 0
            for i in range(0, len(self.route)):
                fromCity = self.route[i]
                toCity = None
                if i + 1 < len(self.route):
                    toCity = self.route[i + 1]
                else:
                    toCity = self.route[0]
                    # la ciudad sabe cómo calcular la distancia
                    # incrementamos
                pathDistance += fromCity.distance(toCity)
            self.distance = pathDistance
        return self.distance

    # fFITNESS
    def routeFitness(self):
        if self.fitness == 0:
            # calculamos el INVERSO de la distancia completa
            self.fitness = 1 / float(self.routeDistance())
        return self.fitness

# crea rutas aleatorias (crea un CROMOSOMA)
# =============================================================================
# ---- fCROMOSOME
# =============================================================================
def createRoute(cityList):
    # creamos una secuencia aleatoria de ciudades a visitar
    route = random.sample(cityList, len(cityList))
    return route

# fPOPULATION
# crea un conjunto de rutas aleatorias
def initialPopulation(popSize, cityList):
    population = []

    for i in range(0, popSize):
        population.append(createRoute(cityList))

    return population

# hace el ranking de rutas
def rankRoutes(population):
    fitnessResults = {}
    # pone como clave el idx original de ruta
    # y como valor el fitness de su ruta
    for i in range(0,len(population)):
        # calculamos el fitness de cada ruta
        fitnessResults[i] = Fitness(population[i]).routeFitness()
    return sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = True)

"""
    Fitness proportionate selection (the version implemented below):
    The fitness of each individual relative to the population is used to assign a
    probability of selection. Think of this as the fitness-weighted probability of being selected.
"""
# =============================================================================
# ---- fSELECTION
# =============================================================================
def selection(popRanked, eliteSize):
    selectionResults = []
    df = pd.DataFrame(np.array(popRanked), columns=["Index","Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100*df.cum_sum/df.Fitness.sum()

    # usamos ELITISMO (cogiendo los mejores y asegurando que entren en la siguiente generación)
    for i in range(0, eliteSize):
        selectionResults.append(popRanked[i][0])

    for i in range(0, len(popRanked) - eliteSize):
        pick = 100*random.random()
        # cogemos cualquier ruta que tenga menos de "pick" Km
        for i in range(0, len(popRanked)):
            # recupera un valor en base a fila, columna
            if pick <= df.iat[i,3]:
                selectionResults.append(popRanked[i][0])
                break
    return selectionResults

# =============================================================================
# ---- fMATINGPOOL
# =============================================================================
# mating pool de padres
def matingPool(population, selectionResults):
    matingpool = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])
    return matingpool

# =============================================================================
# ---- fCROSSOVER
# =============================================================================
# ordered crossover (aka breeding)
def breed(parent1, parent2):
    child = []
    childP1 = []
    childP2 = []

    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))

    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene):
        childP1.append(parent1[i])

    # nos aseguramosque no hay ciudades repetidas (ingenioso!!!!)
    childP2 = [item for item in parent2 if item not in childP1] # iterator!!!!

    child = childP1 + childP2
    return child

# cruce
def breedPopulation(matingpool, eliteSize):
    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))

    # fELITISM
    for i in range(0,eliteSize): # garantizo que los mejores pasan a la siguiente generación sin cruce ni mutación
        children.append(matingpool[i])

    for i in range(0, length):
        child = breed(pool[i], pool[len(matingpool)-i-1])
        children.append(child)
    return children

# =============================================================================
# ---- fMUTATION
# =============================================================================
# mutación de un elemento único
def mutate(individual, mutationRate):
    for swapped in range(len(individual)):
        if(random.random() < mutationRate):
            swapWith = int(random.random() * len(individual))

            city1 = individual[swapped]
            city2 = individual[swapWith]

            individual[swapped] = city2
            individual[swapWith] = city1
    return individual

# mutación de la población
def mutatePopulation(population, mutationRate):
    mutatedPop = []

    for ind in range(0, len(population)):
        mutatedInd = mutate(population[ind], mutationRate)
        mutatedPop.append(mutatedInd)
    return mutatedPop

# =============================================================================
# ---- fGA
# =============================================================================
def nextGeneration(currentGen, eliteSize, mutationRate):
    popRanked = rankRoutes(currentGen)
    selectionResults = selection(popRanked, eliteSize)
    matingpool = matingPool(currentGen, selectionResults)
    children = breedPopulation(matingpool, eliteSize)
    nextGeneration = mutatePopulation(children, mutationRate)
    return nextGeneration

# fGA - esta es la versión que sólo ejecuta y no pinta nada...
# VOILÁ - Genetic Algorithm 100%!!!!
def geneticAlgorithm(population, popSize, eliteSize, mutationRate, generations):
    pop = initialPopulation(popSize, population)
    print("Initial distance: " + str(1 / rankRoutes(pop)[0][1]))

    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)

    print("Final distance: " + str(1 / rankRoutes(pop)[0][1]))
    bestRouteIndex = rankRoutes(pop)[0][0]
    bestRoute = pop[bestRouteIndex]
    return bestRoute


# *****************************************************************************
#   TEST ALGORITHM
#   25 random cities - For Brute Force it would be 300 Sixtillion Routes
# *****************************************************************************
"""
cityList = []

# creamos 25 ciudades
for i in range(0,25):
    cityList.append(City(x=int(random.random() * 200), y=int(random.random() * 200)))

geneticAlgorithm(population=cityList, popSize=100, eliteSize=20, mutationRate=0.01, generations=500)
"""


# =============================================================================
# ---- Program
# =============================================================================
# algo más complejo porque queremos mostrar resultados
def geneticAlgorithmPlot(population, popSize, eliteSize, mutationRate, generations):

    cityList=[]
    cityList = population
    N=len(cityList)-1
    colors = np.random.rand(N)
    x_ord = []
    for i in range(1, len(cityList)):
        x_ord.append(cityList[i].x)
    y_ord = []
    for i in range(1, len(cityList)):
        y_ord.append(cityList[i].y)

    plt.ion()
    #fig = plt.figure(1)

    fig = plt.figure(num=None, figsize=(16, 12), dpi=80, facecolor='w', edgecolor='k')
    fig.canvas.set_window_title('SALESMAN TRAVELER PROBLEM')


    #fig = plt.figure(1)
    #ax1 = fig.add_subplot(111)
    #ax1.set_title("SalesMan - Ruta Aleatoria", fontsize=20)

    # lauouts of subplots in source: https://matplotlib.org/users/gridspec.html
    ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=1)
    ax2 = plt.subplot2grid((3, 2), (0, 1), colspan=1)
    ax3 = plt.subplot2grid((3, 2), (1, 0), colspan=2, rowspan=2)

    #ax1 = fig.add_subplot(221)
    ax1.set_title("SalesMan - Ruta Inicial Aleatoria", fontsize=20)

    #ax2 = fig.add_subplot(222)
    ax2.set_title("SalesMan - Ruta Final", fontsize=20)

    #ax3 = fig.add_subplot(223)
    ax3.set_title("SalesMan - Evolución Fitness & Epochs", fontsize=20)

    # pintamos la primera ruta
    ax1.scatter(x_ord, y_ord, c='b', alpha=0.5)

    # pintamos evolución del algoritmo
    pop = initialPopulation(popSize, population)
    print("Initial distance: " + str(1 / rankRoutes(pop)[0][1]))

    costeruta = str(1 / rankRoutes(pop)[0][1])
    ax1.plot(x_ord, y_ord, label= "Route cost " + costeruta)
    ax1.set_ylabel('Coord Y')
    ax1.set_xlabel('Coord X')
    ax1.legend()
    #ax1.ylabel('y coord')
    #ax1.xlabel('x_coord')
    fig.canvas.draw()

    progress = []
    progress.append(1 / rankRoutes(pop)[0][1])
    #progress, = ax.plot(x, y, 'b-')
    #plt.show()
    ax3.plot(progress, scalex=GENERATIONS, scaley=2000, label="Route Cost", c='r')
    ax3.legend()

    # ·····················································
    # ALGORITMO
    # ·····················································
    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)
        progress.append(1 / rankRoutes(pop)[0][1])
        if not i % 100:
            ax3.plot(progress, scalex=GENERATIONS, scaley=2000, label="Route Cost")
            ax3.set_ylabel('Fitness')
            ax3.set_xlabel('Epoch')
            fig.canvas.draw()

        # condición de salida?

        if (1 / rankRoutes(pop)[0][1]) < 1000:
            break


    #plt.show()

    # pintamos la ruta final
    bestRouteIndex = rankRoutes(pop)[0][0]
    bestRoute = pop[bestRouteIndex]

    cityList = bestRoute
    x_ord = []
    for i in range(1, len(cityList)):
        x_ord.append(cityList[i].x)
    y_ord = []
    for i in range(1, len(cityList)):
        y_ord.append(cityList[i].y)

    ax2.scatter(x_ord, y_ord, c='g', alpha=0.5)
    costeruta = str(1 / rankRoutes(pop)[0][1])
    ax2.plot(x_ord, y_ord, label = "Route Cost " + costeruta)
    ax2.set_ylabel('Coord Y')
    ax2.set_xlabel('Coord X')
    ax2.legend()
    #plt.ylabel('y coord')
    #plt.xlabel('x_coord')
    fig.canvas.draw()


    print("Final distance: " + str(1 / rankRoutes(pop)[0][1]))


cityList = []

# ---- creamos 25 ciudades aleatorias
for i in range(0,25):
    cityList.append(City(x=int(random.random() * 200), y=int(random.random() * 200)))



geneticAlgorithmPlot(population=cityList, popSize=100, eliteSize=20, mutationRate=0.01, generations=GENERATIONS)








