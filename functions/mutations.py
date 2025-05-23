import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from copy import deepcopy
from itertools import combinations



def swap_mutation(
    representation: list[list[int]], 
    mut_prob: float
    ) -> list[list[int]]:
    """
    Performs a swap mutation between two different tables in the seating arrangement.

    Args:
        representation (list[list[int]]): A 2D list representing 8 tables, each containing 8 guest IDs (integers from 0 to 63).
        mut_prob (float): The probability of performing the mutation, between 0.0 and 1.0.

    Returns:
        list[list[int]]: The mutated seating arrangement, or the original if mutation does not occur.
    """
    if random.random() <= mut_prob:
        new_repr = deepcopy(representation)
        # This ensures that table1_idx != table2_idx
        table1_idx, table2_idx = random.sample(range(0,7), 2)
        
        person1_idx = random.randint(0, 7) #These indexes can be equal
        person2_idx = random.randint(0, 7)
        
        new_repr[table1_idx][person1_idx], new_repr[table2_idx][person2_idx] = (
            new_repr[table2_idx][person2_idx],
            new_repr[table1_idx][person1_idx],
        )
        return new_repr
    return representation





def inversion_mutation(
    representation: list[list[int]], 
    mut_prob: float
    ) -> list[list[int]]:
    """
    Performs an inversion mutation by reversing a random segment of the flattened guest list.

    Args:
        representation (list[list[int]]): A 2D list representing 8 tables, each containing 8 guest IDs (integers from 0 to 63).
        mut_prob (float): The probability of performing the mutation, between 0.0 and 1.0.

    Returns:
        list[list[int]]: The mutated seating arrangement, or the original if mutation does not occur.
    """
    if random.random() < mut_prob:
        # Flatten 2D seating into 1D list
        flat = [guest for table in representation for guest in table]

        # Pick two random positions to define the inversion segment
        i, j = sorted(random.sample(range(64), 2))

        # Reverse the segment
        flat[i:j+1] = flat[i:j+1][::-1]

        # Reconstruct 8 tables of 8 guests each
        new_repr = [flat[k:k + 8] for k in range(0, 64, 8)]

        return new_repr
    return representation




def scramble_mutation_optimized(
    representation: list[list[int]], 
    mut_prob: float, 
    k: int = 1
    ) -> list[list[int]]:
    """
    Optimized scramble mutation with adjustable weight decay for scramble size selection.

    Args:
        representation (list[list[int]]): A 2D list representing 8 tables, each containing 8 guest IDs (integers from 0 to 63).
        mut_prob (float): The probability of performing the mutation, between 0.0 and 1.0.
        k (int, optional): Exponent controlling the weight decay for scramble size selection, defaults to 2.

    Returns:
        list[list[int]]: The mutated seating arrangement, or the original if mutation does not occur.
    """
    if random.random() < mut_prob:
        # Flatten 8x8 representation
        flat = [guest for table in representation for guest in table]
        
        # Define the weights for the scramble size (higher weights for smaller sizes)
        weights = [1 / (i**k) for i in range(2, 64)] 
        scramble_size = random.choices(range(2, 64), weights=weights, k=1)[0]
        
        # Select random subset of indices to scramble
        indices = random.sample(range(len(flat)), scramble_size)

        # Extract and shuffle the values
        values = [flat[i] for i in indices]
        random.shuffle(values)

        # Reassign shuffled values back into their positions
        for i, idx in enumerate(indices):
            flat[idx] = values[i]

        # Rebuild into 8 tables of 8 guests each
        new_repr = [flat[i:i + 8] for i in range(0, 64, 8)]


        return new_repr

    return representation 

