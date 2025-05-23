from abc import ABC, abstractmethod
import random
from copy import deepcopy
from typing import Callable
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from functions.crossover import *
from functions.mutations import *



class Solution(ABC):
    def __init__(self, repr=None):
        # To initialize a solution we need to know it's representation. If no representation is given, a solution is randomly initialized.
        if repr is None:
            repr = self.random_initial_representation()
        # Attributes
        self.repr = repr

    # Method that is called when we run print(object of the class)
    def __repr__(self):
        return str(self.repr)

    # Other methods that must be implemented in subclasses
    @abstractmethod
    def fitness(self):
        pass

    @abstractmethod
    def random_initial_representation(self):
        pass



class Wedding_Solution(Solution):
    def __init__(
        self,
        scores: pd.DataFrame | np.ndarray,
        repr: list = None,
    ):
        
        if repr is not None:
            repr = self._validate_repr(repr)
                
        self.scores = self._convert_scores(scores)
        
        super().__init__(repr=repr)
    
    
    def random_initial_representation(self):
        
        representation = []
        all_people = list(range(1, 65))
        
        for i in range(8):
            table = random.sample(all_people, 8)
            all_people = [person for person in all_people if person not in table]
            representation.append(table)
        return representation
     
     
    def fitness(self):
        
        total_score = 0
        for table in self.repr: 
            for i in range(len(table)):
                for j in range(i+1, len(table)):
                    # only considers the upper triangle of the scores
                    total_score += self.scores[table[i]-1][table[j]-1]
        return total_score
    
  
    def __repr__(self):
        
        repr_str = ""
        for idx, table in enumerate(self.repr, start=1):
            repr_str += f"\nTable {idx}: {table}"
        return repr_str
    
    
    def _validate_repr(self, repr):
        
        if isinstance(repr, np.ndarray) and len(repr) != 8:
            raise ValueError("Representation must be a list of 8 tables")

        if not all(len(table) == 8 for table in repr):
            raise ValueError("Each table must have 8 people")
        
        return repr
    
    
    def _convert_scores(self, scores):
        
        if isinstance(scores, pd.DataFrame):
            return scores.to_numpy()
        elif isinstance(scores, np.ndarray):
            return scores
        else:
            raise TypeError("Scores must be a DataFrame or a numpy array")
    


class Wedding_GA_Solution(Wedding_Solution):
    def __init__(
        self, 
        mutation_function,
        crossover_function,
        scores: pd.DataFrame | np.ndarray,
        repr=None, 
    ):
        super().__init__(
            repr=repr,
            scores=scores,
        )
        
        self.mutation_function = mutation_function
        self.crossover_function = crossover_function
        
        
    def mutation(self, mut_prob):
        
        new_repr = self.mutation_function(self.repr, mut_prob)
        
        return Wedding_GA_Solution(
            repr=new_repr,
            mutation_function=self.mutation_function,
            crossover_function=self.crossover_function,
            scores=self.scores
        )
        
        
    def crossover(self, other_solution):
        
        offspring1_repr, offspring2_repr = self.crossover_function(self.repr, other_solution.repr)
        
        return (
            Wedding_GA_Solution(
                repr=offspring1_repr,
                mutation_function=self.mutation_function,
                crossover_function=self.crossover_function,
                scores=self.scores
            ),
            Wedding_GA_Solution(
                repr=offspring2_repr,
                mutation_function=self.mutation_function,
                crossover_function=self.crossover_function,
                scores=self.scores
            )
        )




class Wedding_SimulatedAnnealing_Solution(Wedding_Solution):
    def get_random_neighbor(self, neighbor_operator: Callable):
        
        neighbor = deepcopy(self.repr)
        
        # Apply the neighbor operator
        neighbor_repr = neighbor_operator(representation=neighbor, mut_prob=1)

        # Return the neighbor as a new solution
        return Wedding_SimulatedAnnealing_Solution(
            repr=neighbor_repr,
            scores=self.scores,
        )
        





class Wedding_HC_Solution(Wedding_Solution):
    def get_neighbors(self):
        neighbors = []
        for i in range(8):
            for j in range(i + 1, 8):  # only i < j to avoid duplicate swaps
                for p1 in range(8):
                    for p2 in range(8):
                        new_repr = deepcopy(self.repr)
                        # Swap guests at p1 in table i with p2 in table j
                        new_repr[i][p1], new_repr[j][p2] = new_repr[j][p2], new_repr[i][p1]
                        neighbors.append(
                            Wedding_HC_Solution(repr=new_repr, scores=self.scores)
                        )
        return neighbors
    

