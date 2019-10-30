# -*- coding: utf-8 -*-
"""
Created on Thu Sept 20 23:07:21 2019

@author: Fernando
"""


import datetime
import random
import unittest
import statistics
import sys
import time
import numpy as np

import matplotlib.pyplot as plt

# =============================================================================
# ---- fCHROMOSOME
# =============================================================================
# ·······················································································
# Definción de un cromosoma
# ·······················································································
class _cromosoma:
    def __init__(self, genes, fitness = 0):
        self.Genes = genes # el cromosoma contiene sus genes
        self.Fitness = self.fnGetFitness(genes) # el cromosoma contiene su fitness
        global idx
        self.Idx = idx
        idx += 1

    def __repr__(self):
        return "<Cromosoma :%5s :%s :%s>" % (self.Idx, self.Genes, round(self.Fitness,5))

    def __str__(self):
        return "<Cromosoma :%5s :%s :%s>" % (self.Idx, self.Genes, round(self.Fitness,5))

    def fnGetFitness(self, genes):
       result = 0
       for i in range(0, len(genes)):
              result += genes[i]*X[i]
       #return 1/(abs(y-result)+1)
       return abs(y-result)

    def fnUpdate(self):
        self.Fitness = self.fnGetFitness(self.Genes) # el cromosoma contiene su fitness

# =============================================================================
# ---- fPOPULATION
# =============================================================================
# ·······················································································
# INICIALIZAMOS LA POBLACION
# ·······················································································
def fnFeedPopulation(initial_population = False):

       if initial_population == True:
              for n in range(0, n_cromosomas_per_population):
                     genes = [] # creamos los genes

                     # fGENOME
                     #☺ iniciliazmos el cromosoma con números aleatorios entre -100 y 100
                     for i in range(0,len(geneset)):
                            w = random.randint(-100,100)
                            genes.append(w)
                     cromosoma = _cromosoma(genes)
                     population.append(cromosoma)
              return population.copy()

# =============================================================================
# ---- fELITISM
# =============================================================================
# ·······················································································
# ELITISM
# ·······················································································
def fnElitism(rankedPopulation):
       eliteSize = int(len(rankedPopulation)*elitismPercentage)
       sizeOriginalRankedPopulation = len(rankedPopulation)
       selectionResults = rankedPopulation.copy() # estos son los elitism por si no me llegase nada
       resultPopulation = []
       if eliteSize > 0:
              selectionResults = []
              for i in range(0, eliteSize): # cogemos los elementos por elitism
                     selectionResults.append( rankedPopulation[i] )

       # quitamos de population los elitism que acabo de coger si procede
       for n in range(0, eliteSize):
              rankedPopulation.pop(0)
       # del resto de population hacemos una RouletteWheel
       # obtengo tantos elementos como necesito
       newPopulation = fnRouletteWheel(rankedPopulation, sizeOriginalRankedPopulation-eliteSize) # aquí tenemos la población seleccionada por RouletteWheel

       # newPopulation contiene el elitism y el resto de population elegido por roulettewheel
       for n in selectionResults:
              resultPopulation.append(n)
       for n in newPopulation:
              resultPopulation.append(n)

       return selectionResults, resultPopulation

# =============================================================================
# ---- fCROSSOVER
# =============================================================================
# ·······················································································
# MATING
# obtiene un número de hijos de descendencia, selecciona a los padres y los cruza repetidamente
# ·······················································································
def fnMating(parent1, parent2, max_n_children = 10):
       # tengo que retornar tantos hijos como padres me han llegado
       children = []

       # los dos padres elegidos también sobreviven
       children.append(parent1)
       children.append(parent2)

       # cuánta descendencia queremos?
       n_children = random.randint(1, max_n_children)
       for n in range(0, n_children): # cruzamos los padres repetidas veces
              child = fnCrossOver(parent1, parent2)
              child = fnMutate(child)

              children.append(child)

       return children.copy()

# =============================================================================
# ---- fMATINGPOOL
# =============================================================================
# ·······················································································
# MATING POOL - devuelve dos padres
# ·······················································································
def fnMatingPool(rankedPopulation):
       parent1 = fnRouletteWheelGetParent(population)
       parent2 = fnRouletteWheelGetParent(population)
       return parent1, parent2


# =============================================================================
# ---- fSELECTION
# =============================================================================
# ·······················································································
# ROULETTE WHEEL para encontrar los dos padres a cruzar
# ·······················································································
# el elitism lo mantengo para la siguiente generación
def fnRouletteWheelGetParent(population):
       probabilities = []
       prob = 0.0
       sumaFitness = sum(n.Fitness for n in population)
       population = fnRankPopulation(population) # lo ordeno al reves para que los primeros tengan menos probabilidad
       for i in population:
              prob += (i.Fitness / sumaFitness)
              probabilities.append(prob)
       # tiramos la bola
       bola = random.random()
       # encontramos qué posición nos devuelve
       posicion = 0
       for x in probabilities:
              if bola < x:
                     posicion +=1
       posicion -= 1
       return population[posicion]

# fSELECTION
# ·······················································································
# SELECTION - ROULETTE WHEEL selecciona tantos elementos como se le pide
# ·······················································································
def fnRouletteWheel(population, size):
       probabilities = []
       prob = 0.0
       sumaFitness = sum(n.Fitness for n in population)
       population = fnRankPopulation(population)

       newPopulation = []

       for i in population:
              prob += (i.Fitness / sumaFitness)
              probabilities.append(prob)

       for tiradas in range(0, size):
              # tiramos la bola
              bola = random.random()
              # encontramos qué posición nos devuelve
              posicion = 0
              for x in probabilities:
                     if bola < x:
                            posicion +=1
              posicion -= 1
              newPopulation.append(population[posicion])

       return newPopulation

# =============================================================================
# ---- fFITNESS
# =============================================================================
# ·······················································································
# Calculamos el FITNESS como el inverso del absoluto del error
# ·······················································································
def fnGetFitnessCromosoma(cromosoma):
       cromosoma.Fitness = fnGetFitnessGenes(cromosoma.Genes)
       return cromosoma.Fitness


def fnGetFitnessGenes(genes):
       result = 0
       for i in range(0, len(genes)):
              result += genes[i]*X[i]
       #return 1/(abs(y-result)+1)
       return abs(y-result)

# =============================================================================
# ---- fMUTATION
# =============================================================================
# ·······················································································
# MUTACION
# Aplicamos, si 0 division y si 1 multiplicacion
# ·······················································································
def fnMutate(cromosoma):
       for i in range(0, len(cromosoma.Genes)):
              if random.random() < pm: # cambiamos el W?
                     if random.random() < 0.5:
                            cromosoma.Genes[i] /= geneset[i]
                     else:
                            cromosoma.Genes[i] *= geneset[i]
              if random.random() < pm: # cambiamos el signo?
                     if random.random() < 0.5:
                            cromosoma.Genes[i] *= -1
       cromosoma.fnUpdate()
       #cromosoma.Fitness = fnGetFitnessCromosoma(cromosoma)
       return cromosoma

# =============================================================================
# ---- fCROSSOVER
# =============================================================================
# ·······················································································
# CROSSOVER
# Cruzamos dos cromosomas y devolvemos un hijo
# ·······················································································
def fnCrossOver(cromosoma1, cromosoma2):
         # slicer = int(len(self.gene(loser))/2)
         # slicer = 3
         order = random.randint(0,1) # el orden de cruce es aleatorio
         slicer = random.randint(1,len(cromosoma1.Genes)-1)
         genes = []
         if order == 0:
                for i in range(0, slicer):
                       genes.append(cromosoma1.Genes[i])
                for i in range(slicer, len(cromosoma2.Genes)):
                       genes.append(cromosoma2.Genes[i])
         else:
                for i in range(0, slicer):
                       genes.append(cromosoma2.Genes[i])
                for i in range(slicer, len(cromosoma2.Genes)):
                       genes.append(cromosoma1.Genes[i])

         cromosoma = _cromosoma(genes)
         child = cromosoma
         return child

# ·······················································································
# RANK
# Ordenamos la population por orden de importancia
# ·······················································································
def fnRankPopulation(population, reverse = False):
       population = sorted(population, key = lambda item: item.Fitness, reverse = reverse) # ordenamos los cromosomas en función de su fitness
       return population.copy()


# ·······················································································
# NEW POPULATION
# Insertamos el elitism y los mejores children de la generación actual
# ·······················································································
def fnNewPopulation(elitism, children):
       population = []
       for i in elitism: # cargamos el elitism (garantizamos que los mejores de la generación anterior perduren en esta generación)
              population.append(i)
       children = fnRankPopulation(children) # ordenamos la prole para quedarnos con los mejores
       pendingCromosomas = n_cromosomas_per_population-len(elitism) # cromosomas que nos quedan por añadir
       for n in children: # cogemos, de los hijos, todos los que podamos hasta completar la población
              # esta generación de hijos también contiene padres (y puede que alguno de elitism) - para evitar que se bloquee y no aumente la diversidad, garantizamos que no haya duplicados
              existe = fnIsAlreadyInPopulation( n, population)
              if existe == False:
                     population.append(n)
                     pendingCromosomas -= 1
                     if pendingCromosomas == 0:
                            break

       return population.copy()


# ·······················································································
# Existe este cromosoma en Population?
# ·······················································································
def fnIsAlreadyInPopulation(children, population):
       for i in population:
              if children.Idx == i.Idx:
                     return True
       return False

# =============================================================================
# ---- fGA
# =============================================================================
# ·······················································································
# ALGORITMO GENETICO COMPLETO
#
# ·······················································································
def fnGeneticAlgorithm():
       population = [] # guardará una generación
       population = fnFeedPopulation(initial_population = True)

       for n in range(0, tournaments):
              population = fnRankPopulation(population) # ordenamos la population para que queden los mejores arriba
              elitism, population = fnElitism(population) # nos quedamos con los n mejores y hacemos una selección por RouletteWheel del resto
              newChildren = []
              for i in range(0, n_cruces):

                     parent1, parent2 = fnMatingPool(population) # generamos el mating pool (usa RouletteWheel para la selección de los dos padres)
                     children = fnMating(parent1, parent2) # cruzamos a los padres y obtenemos una ristra de hijos
                     for n in children:
                            newChildren.append(n) # añadimos a la prole

              population = fnRankPopulation(fnNewPopulation(elitism, newChildren))
              resultadoPropuesto = population[0]
              if resultadoPropuesto.Fitness == 0:
                     break

       return resultadoPropuesto

# =============================================================================
# ---- fINITIALVALUES
# =============================================================================
# ·······················································································
# AJUSTANDO LOS PARAMETROS DE UNA REGRESION
# ·······················································································

X = [4,-2,7,5,11,1] # estos son los datos que tenemos que ajustar
y = 44.1

idx = 0 # idx de cada cromosoma
geneset = [2,2,2,2,2,2] # realmente opera sobre las posibles mutaciones, ya que el valor puede ser cualquiera
population = [] # número de elementos
n_cromosomas_per_population = 50
tournaments = 1000 # número de generaciones
pm = 0.5 # posibilidad de mutacion
matingPoolPercentage = 0.2 # cogemos el 20% de los mejores elementos para cruzarlos
elitismPercentage = 0.2 # porcentaje de elementos que mantenemos como elitism
n_cruces = 10 # cuántos cruces de padres queremos
# el elitismo en esta función está asegurado porque nos quedamos con los mejores


# =============================================================================
# ---- fMAIN
# =============================================================================
fnGeneticAlgorithm()




