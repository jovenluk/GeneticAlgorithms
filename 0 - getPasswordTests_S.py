# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 23:07:21 2019

@author: Fernando
"""

## Sheppard, Clinton. Genetic Algorithms with Python (Posición en Kindle272-279). Edición de Kindle.

# File: guessPasswordTests.py
#    from chapter 1 of _Genetic Algorithms with Python_
#
# Author: Clinton Sheppard <fluentcoder@gmail.com>
# Copyright (c) 2016 Clinton Sheppard
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied.  See the License for the specific language governing
# permissions and limitations under the License.

import datetime
import random
import unittest
import statistics
import sys
import time

import matplotlib.pyplot as plt


# ························································································
# ---- fCHROMOSOME
# Contiene toda la información importante de un cromosoma (realmente es una solucion propuesta)
# ························································································
class Chromosome:
    def __init__(self, genes, fitness, age = 0):
        self.Genes = genes # el cromosoma contiene sus genes
        self.Fitness = fitness # el cromosoma contiene su fitness
        self.Age = age # recuerda cuántas veces ha sido utilizado

# ························································································
# ---- fCHROMOSOME
# Esta función genera un nuevo cromosoma
# ························································································
def _generate_parent(length, geneSet, get_fitness):
    genes = []

    # va generando a "trozos" una secuencia aleatoria para el parent
    while len(genes) < length:
        sampleSize = min(length - len(genes), len(geneSet))
        genes.extend(random.sample(geneSet, sampleSize))
    #genes = ''.join(genes)
    fitness = get_fitness(genes) # calculamos el fitness
    return Chromosome(genes, fitness) # retornamos un cromosoma con su fitness ya calculado

# ························································································
# ---- fMUTATION
# esta función es muy simple y siempre hace una mutación
# ························································································
def _mutate(parent, geneSet, get_fitness):
    childGenes = parent.Genes[:] # el hijo contiene una copia exacta del padre (no hay crossover)
    index = random.randrange(0, len(parent.Genes)) # selecciona un gen únicamente
    newGene, alternate = random.sample(geneSet, 2) # coge dos genes al azar
    # interesante construcción
    # si el if devuelve TRUE entonces se ejecuta lo que está antes, tal y como está
    # pero si devuelve FALSE entonces childGenes tomará el valor de newGene
    childGenes[index] = alternate if newGene == childGenes[index] else newGene # garantiza que cambia un gen al menos
    #genes = ''.join(childGenes)
    fitness = get_fitness(childGenes) # calculamos el fitness
    return Chromosome(childGenes, fitness) # retornamos un cromosoma con su fitness ya calculado

# ························································································
# ---- fFITNESS
# la función de fitness devuelve un 1 por cada letra encontrada en su sitio
# el mejor fitness es el LEN del guess
# ························································································
def get_fitness(guess, target):
    return sum(1 for expected, actual in zip(target, guess)
               if expected == actual)

# ························································································
# ---- fDISPLAY
# función que se encarga de enseñar algo por pantalla
# ························································································
def display(candidate, startTime, epoch):
    timeDiff = datetime.datetime.now() - startTime
    print("{}\t{}\t{}\t{}".format(
        ''.join(candidate.Genes), candidate.Fitness, timeDiff, epoch))


"""
# probamos con otra cadena más larga
def test_For_I_am_fearfully_and_wonderfully_made(self):
       target = "For I am fearfully and wonderfully made."
       self.guess_password(target)
"""

# ························································································
# ---- fGA
# EVOLUCIONA LAS GENERACIONES
# Pseudo
# Genera un nuevo cromosoma (get_improvement)
# Si el fitness es el óptimo, termina el proceso
# ························································································
def get_best(get_fitness, targetLen, optimalFitness, geneSet, display,
             custom_mutate=None, custom_create=None, maxAge=None):

     progress = []
     tournaments = []

     plt.ion()
     fig = plt.figure(num=1, figsize=(12, 12), dpi=80, facecolor='w', edgecolor='k')
     fig.canvas.set_window_title('GUESS THE PASSWORD PROBLEM')
     ax1 = fig.add_subplot(111)
     ax1.set_title("Guess the password - Fitness curve and epochs", fontsize=20)

     list_of_errors = []
     for improvement, epoch in _get_improvement(_mutate, _generate_parent, targetLen, geneSet, get_fitness, maxAge): # devuelve un conjunt de cromosomas
         display(improvement, epoch)

         list_of_errors.append( improvement.Fitness )
         med = statistics.mean(list_of_errors)
         progress.append( med )
         tournaments.append(epoch)
         ax1.plot(tournaments, list_of_errors, label="Median Fitness Cost", scaley=12)
         ax1.set_ylabel('Fitness')
         ax1.set_xlabel('Epoch')
         #plt.draw()

         fig.canvas.draw()

         if not optimalFitness > improvement.Fitness: # si el optimal fitness se alcanza lo devuelve
             return improvement, epoch

# ························································································
# EVOLUCION
# Función simplificada: crea un nuevo cromosoma, lo muta, lo compara con el anterior y devuelve el mejor de los dos
# NOTA TECNICA: Va creando un ITERATOR
# ························································································
def _get_improvement(new_child, generate_parent, targetLen, geneSet, get_fitness, maxAge=0):
    epoch = 1 # inicializamos epoch a 1, como hay yields, la función continuará y epoch se seguirá incrementando en cada iteracción
    bestParent = generate_parent(targetLen, geneSet, get_fitness)
    #bestParent = generate_parent() # genero un parent
    yield bestParent, epoch
    while True:
        epoch+=1
        child = new_child(bestParent, geneSet, get_fitness) # genero un child (realmente está llamando a la función MUTATE)
        if bestParent.Fitness > child.Fitness: # si el fitness del hijo no es mejor que el del padre, continuo
            continue
        if not child.Fitness > bestParent.Fitness: # si el fitness del hijo es mejor que el del padre
            bestParent = child # ahora el mejor padre pasa a ser el hijo
            continue
        yield child, epoch  # retorno en el iterator el cromosoma que ha evolucionado gracias a la mutación y su epoch
        bestParent = child



"""
# definimos un target aleatorio de 150 letras!!!
def test_Random(self):
       length = 150
       target = ''.join(random.choice(self.geneset)
                  for _ in range(length))

guess_password(target)

# le pasamos la función de la que queremos hacer benchmark
def test_benchmark(self):
       genetic.Benchmark.run(self.test_Random)
"""


# ························································································
# probamos con "Hello World"
# ························································································
# ························································································
# definimos los genes
# ························································································


# ························································································
# ---- fMAIN
# (1) inicio de todo el proceso
# ························································································
def guess_password(target):
       startTime = datetime.datetime.now()
       # fGENOTYPE
       geneset = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!.,"



       # ························································································
       # (2) notificamos qué funciones vamos a usar
       # ························································································
       def fnGetFitness(genes):
            return get_fitness(genes, target)

       def fnDisplay(candidate, epoch):
            display(candidate, startTime, epoch)

       # ························································································
       # decidimos que el objetivo a lograr es que todas las letras estén en su lugar (asociado a la función fitness que hemos codificado)
       # ························································································
       optimalFitness = len(target)

       # ························································································
       # obtenemos el best pasándole la función que calcula el fitness, dos veces el len del target (!!!), el genoma y la función para hacer display de resultados
       # ························································································
       best, epoch = get_best(fnGetFitness, # función custom que calcula el fitness del cromosoma
                                len(target),  # longitud del cromosoma
                                optimalFitness, # cuál es el fitness óptimo
                                geneset, # el genoma que vamos a usar
                                fnDisplay) # función custom de display (muy sencillita)

# ························································································
# ---- Programa
# ························································································
target = "Hello World!"
guess_password(target)

# ························································································
# yield construye iteradores para ser usados por otras funciones
# permite el uso en PARALELO de las dos funciones
# ························································································
#def contador(max):
#    n=0
#    while n < max:
#          print("f(contador) : " + str(n))
#          yield n
#          n=n+1
#
#contad = contador(10)
#for i in contad:
#    print("valor: "+str(i))



