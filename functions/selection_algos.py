import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from copy import deepcopy
from functions.solutions import *
import math 


     
def rank_selection_optimized(population: list, function, maximization: bool, l=0.2, fitness_list=None):
    """
    - function: 'linear' or 'exponential'
    - maximization: True for maximization problems
    - l: lambda for exponential rank-based selection
    - fitness_list: list of fitness values in the same order as population
    """

    n = len(population)
    ranking = list(range(1, n + 1))

    # Use provided fitness_list or compute it
    if fitness_list is None:
        fitness_list = [ind.fitness() for ind in population]

    # Sort population and fitness values accordingly
    if maximization:
        paired = sorted(zip(population, fitness_list), key=lambda x: x[1])
    else:
        paired = sorted(zip(population, fitness_list), key=lambda x: x[1], reverse=True)

    sorted_population, sorted_fitness = zip(*paired)

    # Compute rank-based selection probabilities
    probabilities = []

    if function == 'linear':
        denominator = sum(ranking)
        probabilities = [rank / denominator for rank in ranking]

    elif function == 'exponential':
        denominator = sum(math.exp(-l * (n - rank)) for rank in ranking)
        probabilities = [math.exp(-l * (n - rank)) / denominator for rank in ranking]

    # Roulette wheel selection
    random_nr = random.uniform(0, 1)
    box_boundary = 0
    for ind_idx, ind in enumerate(sorted_population):
        box_boundary += probabilities[ind_idx]
        if random_nr <= box_boundary:
            return deepcopy(ind)




def tournament_selection_optimized(population: list[Solution], k, maximization: bool, fitness_list: list):
    if k >= len(population):
        raise ValueError("Tournament size k must be smaller than the population size")

    # Sample k indices with replacement
    indices = random.choices(range(len(population)), k=k)

    # Pair individuals directly with fitness
    tournament = [(population[i], fitness_list[i]) for i in indices]

    # Select the best based on fitness
    if maximization:
        best = max(tournament, key=lambda x: x[1])
    else:
        best = min(tournament, key=lambda x: x[1])
    return deepcopy(best[0])