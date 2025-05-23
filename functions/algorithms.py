import random
from copy import deepcopy
from typing import Callable
from functions.solutions import *
from functions.selection_algos import * 
import time




def genetic_algorithm_optimized(
    initial_population: list[Solution],
    max_gen: int,
    selection_algorithm: Callable,
    maximization: bool = False,
    xo_prob: float = 0.9,
    mut_prob: float = 0.2,
    elitism: bool = True,
    verbose: bool = False,
) -> tuple[Solution, list, int, float]:
    """
    Executes a genetic algorithm to optimize a population of solutions.

    Args:
        initial_population (list[Solution]): The starting population of solutions.
        max_gen (int): The maximum number of generations to evolve.
        selection_algorithm (Callable): Function used for selecting individuals.
        maximization (bool, optional): If True, maximizes the fitness function; otherwise, minimizes. Defaults to False.
        xo_prob (float, optional): Probability of applying crossover. Defaults to 0.9.
        mut_prob (float, optional): Probability of applying mutation. Defaults to 0.2.
        elitism (bool, optional): If True, carries the best individual to the next generation. Defaults to True.
        verbose (bool, optional): If True, prints detailed logs for debugging. Defaults to False.

    Returns:
        tuple(Solution, list, int, float): Best solution, fitness history, generation of convergence, and convergence time.
        
    """
    # 1. Initialize a population with N individuals
    population = initial_population
    
    best_fitness_history = []
    best_solution = None

    # For convergence tracking
    best_fitness = None
    convergence_gen = 0
    convergence_time = 0
    start_time = time.time()
    
    # 2. Repeat until termination condition
    for gen in range(1, max_gen + 1):

        # 2.1. Create an empty population P'
        new_population = []
        
        def get_best_ind_optimized(population: list[Solution], maximization: bool):
            fitness_list = [ind.fitness() for ind in population]
            if maximization:
                return population[fitness_list.index(max(fitness_list))], fitness_list
            else:
                return population[fitness_list.index(min(fitness_list))], fitness_list 
            
        # Evaluate fitness once per population
        current_best, fitness_list = get_best_ind_optimized(population, maximization)
        current_fitness = current_best.fitness()
        best_fitness_history.append(current_fitness)

        # Update convergence
        if (best_fitness is None) or \
           (maximization and current_fitness > best_fitness) or \
           (not maximization and current_fitness < best_fitness):
            best_fitness = current_fitness
            best_solution = deepcopy(current_best)
            convergence_gen = gen
            convergence_time = time.time() - start_time

        if elitism:
            new_population.append(deepcopy(current_best))

        # 2.3. Repeat until P' contains N individuals
        while len(new_population) < len(population):
            # 2.3.1. Choose 2 individuals from P using a selection algorithm
            first_ind = selection_algorithm(population=population, maximization=maximization, fitness_list=fitness_list)
            second_ind = selection_algorithm(population=population, maximization=maximization, fitness_list=fitness_list)

            # 2.3.2. Choose an operator between crossover and replication
            # 2.3.3. Apply the operator to generate the offspring
            if random.random() < xo_prob:
                offspring1, offspring2 = first_ind.crossover(second_ind)
            else:
                offspring1, offspring2 = deepcopy(first_ind), deepcopy(second_ind)
            
            # 2.3.4. Apply mutation to the offspring
            first_new_ind = offspring1.mutation(mut_prob)
            # 2.3.5. Insert the mutated individuals into P'
            new_population.append(first_new_ind)

            if len(new_population) < len(population):
                second_new_ind = offspring2.mutation(mut_prob)
                new_population.append(second_new_ind)
        
        
        # 2.4. Replace P with P'
        population = new_population
    
        if verbose:
            print(f'Best individual in gen {gen} with fitness: {best_fitness}')
            
    # Final update after last generation
    final_best, _ = get_best_ind_optimized(population, maximization)
    final_fitness = final_best.fitness()
    best_fitness_history.append(final_fitness)

    if (maximization and final_fitness > best_fitness) or \
       (not maximization and final_fitness < best_fitness):
        best_solution = deepcopy(final_best)
        convergence_gen = max_gen
        convergence_time = time.time() - start_time
        
    # 3. Return the best individual in P
    return best_solution, best_fitness_history, convergence_gen, convergence_time





def simulated_annealing(
    initial_solution: Solution,
    C: float,
    L: int,
    H: float,
    maximization: bool = True,
    max_iter: int = 10,
    verbose: bool = False,
    neighbor_operator: Callable = None
) -> tuple[Solution, list, int, float]:
    """
    Executes Simulated Annealing with convergence time tracking.

    Args:
        initial_solution (Solution): Initial solution object to optimize.
        C (float): Initial temperature parameter for the annealing schedule.
        L (int): Number of iterations per temperature level.
        H (float): Cooling rate for reducing the temperature (C = C / H).
        maximization (bool, optional): If True, maximizes the fitness function; else minimizes, defaults to True.
        max_iter (int, optional): Maximum number of temperature iterations, defaults to 10.
        verbose (bool, optional): If True, prints detailed logs for debugging, defaults to False.
        neighbor_operator (Callable, optional): Function to generate a random neighbor, defaults to None.

    Returns:
        tuple[Solution, list[float], int, float]: Best solution, fitness history, iteration of convergence, and convergence time.
    """
    current_solution = initial_solution
    best_solution = deepcopy(current_solution)
    current_fitness = current_solution.fitness() 
    best_fitness = current_fitness

    iter = 1
    start_time = time.time()
    convergence_iter = 1
    convergence_time = 0
    fitness_history = []
    
    evaluations = 0

    fitness_history.append(best_fitness)
    
    if verbose:
        print(f'Initial solution: {current_solution.repr} with fitness {best_fitness}')

    while iter <= max_iter:
        for _ in range(L):
            # Get random neighbor using optional operator
            random_neighbor = current_solution.get_random_neighbor(neighbor_operator)

            neighbor_fitness = random_neighbor.fitness()
            evaluations += 1

            if (
                (maximization and neighbor_fitness > current_fitness) or
                (not maximization and neighbor_fitness < current_fitness)
            ):
                current_solution = deepcopy(random_neighbor)
                current_fitness = neighbor_fitness # cache the result 

                if (
                    (maximization and neighbor_fitness > best_fitness) or
                    (not maximization and neighbor_fitness < best_fitness)
                ):
                    best_solution = deepcopy(random_neighbor)
                    best_fitness = neighbor_fitness
                    convergence_iter = iter
                    convergence_time = time.time() - start_time

            else:
                # Accept worse solution with probability
                delta = abs(current_fitness - neighbor_fitness)
                acceptance_prob = np.exp(-delta / C)
                if random.random() < acceptance_prob:
                    current_solution = deepcopy(random_neighbor)
                    current_fitness = neighbor_fitness # cache the result
            
            if evaluations % 50 == 0:
                fitness_history.append(current_fitness)

        C = C / H
        iter += 1
    
    if evaluations % 50 != 0:
        fitness_history.append(current_fitness)

    return best_solution, fitness_history, convergence_iter, round(convergence_time, 2)






def hill_climbing(
    initial_solution: Solution,
    maximization: bool = True,
    max_iter: int = 100,
    verbose: bool = False,
) -> tuple[Solution, list, int, float]:
    """
    Executes Hill Climbing algorithm with convergence time tracking.

    Args:
        initial_solution (Solution): Initial solution object to optimize.
        max_iter (int, optional): Maximum number of iterations, defaults to 100.
        verbose (bool, optional): If True, prints detailed logs for debugging, defaults to False.
        neighbor_operator (Callable, optional): Function to generate a random neighbor, defaults to None.
        maximization (bool, optional): If True, maximizes the fitness function; else minimizes, defaults to True.

    Returns:
        tuple[Solution, list[float], int, float]: Best solution, fitness history, iteration of convergence, and convergence time.
    """
    current = deepcopy(initial_solution)
    best_solution = deepcopy(current)
    best_fitness = current.fitness()
    convergence_iter = 1
    fitness_history = []
    start_time = time.time()
    fitness_history.append(best_fitness)

    for i in range(1, max_iter + 1):
        
        if verbose:
            print(f"Iteration {i}: Current fitness = {current.fitness()}")

        neighbors = current.get_neighbors()
        evaluated_neighbors = [(n, n.fitness()) for n in neighbors]
        
        improved = False


        if maximization:
            best_neighbor, best_neighbor_fitness = max(evaluated_neighbors, key=lambda x: x[1])
        else:
            best_neighbor, best_neighbor_fitness = min(evaluated_neighbors, key=lambda x: x[1])


        if (maximization and best_neighbor_fitness > current.fitness()) \
            or (not maximization and best_neighbor_fitness < current.fitness()):
                
            current = deepcopy(best_neighbor)
            improved = True

            best_solution = deepcopy(best_neighbor)
            best_fitness = best_neighbor_fitness
            convergence_iter = i

        fitness_history.append(best_fitness)

        if not improved:
            break

    convergence_time = round(time.time() - start_time, 2)
    return best_solution, fitness_history, convergence_iter, convergence_time
