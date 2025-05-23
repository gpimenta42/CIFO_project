
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from copy import deepcopy
from itertools import combinations



def classic_order_crossover(
    p1: list[list[int]], 
    p2: list[list[int]], 
    verbose: bool = False
    )-> tuple[list[list[int]], list[list[int]]]:
    """
    Performs classic Order Crossover (OX) with shuffled guest indices.

    Args:
        p1 (list[list[int]]): First parent seating arrangement, a 2D list with 8 tables of 8 guest IDs (integers from 1 to 64).
        p2 (list[list[int]]): Second parent seating arrangement, a 2D list with 8 tables of 8 guest IDs (integers from 1 to 64).
        verbose (bool, optional): If True, prints the proportion of guest pairs preserved, defaults to False.

    Returns:
        tuple[list[list[int]], list[list[int]]]: Two child seating arrangements in 8x8 table format.
    """
    # Step 1: Flatten parents
    p1_flat = [g for table in p1 for g in table]
    p2_flat = [g for table in p2 for g in table]

    # Step 2: Shuffle guest indices
    shuffled_indices = list(range(64))
    random.shuffle(shuffled_indices)

    p1_shuffled = [p1_flat[i] for i in shuffled_indices]
    p2_shuffled = [p2_flat[i] for i in shuffled_indices]

    # Step 3: Choose crossover segment
    start, end = sorted(random.sample(range(64), 2))

    child1_shuffled = [None] * 64
    child2_shuffled = [None] * 64

    # Copy the crossover segment
    child1_shuffled[start:end+1] = p1_shuffled[start:end+1]
    child2_shuffled[start:end+1] = p2_shuffled[start:end+1]

    # Fill in the remaining spots from the other parent
    def fill_ox(target, donor):
        donor_idx = 0
        for i in range(64):
            if target[i] is None:
                while donor[donor_idx] in target:
                    donor_idx += 1
                target[i] = donor[donor_idx]

    fill_ox(child1_shuffled, p2_shuffled)
    fill_ox(child2_shuffled, p1_shuffled)

    # Step 4: Unshuffle — map back to original guest positions
    child1 = [None] * 64
    child2 = [None] * 64
    for i, idx in enumerate(shuffled_indices):
        child1[idx] = child1_shuffled[i]
        child2[idx] = child2_shuffled[i]

    # Step 5: Rebuild 8x8 table format
    child1_repr = [child1[i:i + 8] for i in range(0, 64, 8)]
    child2_repr = [child2[i:i + 8] for i in range(0, 64, 8)]
    
    if verbose:
        def get_pairs(rep):
            pairs = set()
            for table in rep:
                for a, b in combinations(sorted(table), 2):
                    pairs.add((a, b))
            return pairs

        parent_pairs = get_pairs(p1)
        child1_pairs = get_pairs(child1_repr)
        child2_pairs = get_pairs(child2_repr)

        same1 = len(parent_pairs & child1_pairs)
        same2 = len(parent_pairs & child2_pairs)
        total = len(parent_pairs)

        print(f"Child 1: {same1}/{total} guest pairs ({same1/total:.1%}) stayed together")
        print(f"Child 2: {same2}/{total} guest pairs ({same2/total:.1%}) stayed together")
        
    return child1_repr, child2_repr




def group_preserving_order_crossover(
    p1: list[list[int]], 
    p2: list[list[int]]
    ) -> tuple[list[list[int]], list[list[int]]]:
    """
    Performs group-preserving order crossover, maintaining some entire tables from one parent.

    Args:
        p1 (list[list[int]]): First parent seating arrangement, a 2D list with 8 tables of 8 guest IDs (integers from 1 to 64).
        p2 (list[list[int]]): Second parent seating arrangement, a 2D list with 8 tables of 8 guest IDs (integers from 1 to 64).

    Returns:
        tuple[list[list[int]], list[list[int]]]: Two child seating arrangements in 8x8 table format.
    """
    def build_child(preserve_from, fill_from):
        # Shuffle both parent table orders
        preserve_tables = deepcopy(preserve_from)
        fill_tables = deepcopy(fill_from)
        random.shuffle(preserve_tables)
        random.shuffle(fill_tables)

        # Randomly choose how many tables to preserve
        num_preserve = random.randint(1, 7)
        preserve_indices = sorted(random.sample(range(8), num_preserve))
        preserved = [preserve_tables[i] for i in preserve_indices]
        preserved_guests = set(g for table in preserved for g in table)

        # Flatten the fill-from parent and remove preserved guests
        fill_order = [g for table in fill_tables for g in table if g not in preserved_guests]

        # Fill remaining tables
        remaining_tables = [fill_order[i*8:(i+1)*8] for i in range(8 - num_preserve)]

        # Combine preserved and new tables
        child = preserved + remaining_tables
        random.shuffle(child)

        # Validation
        flat = [g for t in child for g in t]
        assert len(flat) == 64 and len(set(flat)) == 64, "Invalid child: duplicates or missing guests"
        return child


    child1 = build_child(p1, p2)
    child2 = build_child(p2, p1)
    return child1, child2




def partially_mapped_crossover(
    rep1: list[list[int]], 
    rep2: list[list[int]]
    ) -> tuple[list[list[int]], list[list[int]]]:
    """
    Performs partially mapped crossover (PMX) with shuffled indices.

    Args:
        rep1 (list[list[int]]): First parent seating arrangement, a 2D list with 8 tables of 8 guest IDs (integers from 1 to 64).
        rep2 (list[list[int]]): Second parent seating arrangement, a 2D list with 8 tables of 8 guest IDs (integers from 1 to 64).

    Returns:
        tuple[list[list[int]], list[list[int]]]: Two child seating arrangements in 8x8 table format.
    """
    parent1_flat = [guest for table in rep1 for guest in table]
    parent2_flat = [guest for table in rep2 for guest in table]

    # Step 1: Shuffle indices
    shuffled_indices = list(range(64))
    random.shuffle(shuffled_indices)

    p1_shuffled = [parent1_flat[i] for i in shuffled_indices]
    p2_shuffled = [parent2_flat[i] for i in shuffled_indices]

    # Step 2: Choose crossover segment
    start, end = sorted(random.sample(range(64), 2))
    child1_shuffled = [None] * 64
    child2_shuffled = [None] * 64

    # Step 3: Copy mapped section from other parent
    child1_shuffled[start:end+1] = p2_shuffled[start:end+1]
    child2_shuffled[start:end+1] = p1_shuffled[start:end+1]

    # Step 4: Create mappings
    mapping1 = {p2_shuffled[i]: p1_shuffled[i] for i in range(start, end+1)}
    mapping2 = {p1_shuffled[i]: p2_shuffled[i] for i in range(start, end+1)}

    def resolve_conflict(val, mapped_section, mapping):
        while val in mapped_section:
            val = mapping[val]
        return val

    # Step 5: Fill remaining positions
    for i in range(64):
        if i < start or i > end:
            # For child1
            val1 = p1_shuffled[i]
            if val1 not in child1_shuffled:
                child1_shuffled[i] = val1
            else:
                child1_shuffled[i] = resolve_conflict(val1, child1_shuffled[start:end+1], mapping1)

            # For child2
            val2 = p2_shuffled[i]
            if val2 not in child2_shuffled:
                child2_shuffled[i] = val2
            else:
                child2_shuffled[i] = resolve_conflict(val2, child2_shuffled[start:end+1], mapping2)

    # Step 6: Unshuffle to original guest order
    child1 = [None] * 64
    child2 = [None] * 64
    for i, idx in enumerate(shuffled_indices):
        child1[idx] = child1_shuffled[i]
        child2[idx] = child2_shuffled[i]

    # Step 7: Rebuild into 8x8 tables
    offspring1 = [child1[i:i+8] for i in range(0, 64, 8)]
    offspring2 = [child2[i:i+8] for i in range(0, 64, 8)]

    return offspring1, offspring2





def pmx_table_block_crossover(
    p1: list[list[int]], 
    p2: list[list[int]]
    ) -> tuple[list[list[int]], list[list[int]]]:
    """
    Performs PMX crossover on contiguous table blocks.

    Args:
        p1 (list[list[int]]): First parent seating arrangement, a 2D list with 8 tables of 8 guest IDs (integers from 1 to 64).
        p2 (list[list[int]]): Second parent seating arrangement, a 2D list with 8 tables of 8 guest IDs (integers from 1 to 64).

    Returns:
        tuple[list[list[int]], list[list[int]]]: Two child seating arrangements in 8x8 table format.
    """
    # Step 0: Shuffle table order to break position bias
    p1_ = deepcopy(p1)
    p2_ = deepcopy(p2)
    random.shuffle(p1_)
    random.shuffle(p2_)
    
    # Flatten both parents
    p1_flat = [g for table in p1_ for g in table]
    p2_flat = [g for table in p2_ for g in table]
    
    # Randomly choose number of contiguous tables for PMX mapping
    num_tables = random.randint(1, 7)  # can't be 8 or you'll map all tables
    
    # Step 1: Randomly choose the starting table index
    start_table = random.randint(0, 8 - num_tables)  # ensures room for n tables
    
    # Step 2: Calculate flat index range for contiguous tables
    start = start_table * 8
    end = (start_table + num_tables) * 8 - 1  # inclusive

    # Step 2: Initialize children
    child1 = [None] * 64
    child2 = [None] * 64

    # Step 3: Copy PMX segment from opposite parent
    child1[start:end+1] = p2_flat[start:end+1]
    child2[start:end+1] = p1_flat[start:end+1]

    # Step 4: Build mapping dictionaries
    mapping1 = {p2_flat[i]: p1_flat[i] for i in range(start, end+1)}
    mapping2 = {p1_flat[i]: p2_flat[i] for i in range(start, end+1)}

    # If a value in the mapping already exists in the child 
    def resolve_conflict(val, mapping, segment_vals):
        while val in segment_vals:
            val = mapping[val]
        return val

    # Step 5: Fill the rest of child1
    for i in range(64):
        if i < start or i > end:
            val1 = p1_flat[i]
            val2 = p2_flat[i]

            # For child1
            if val1 not in child1:
                child1[i] = val1
            else:
                child1[i] = resolve_conflict(val1, mapping1, child1[start:end+1])

            # For child2
            if val2 not in child2:
                child2[i] = val2
            else:
                child2[i] = resolve_conflict(val2, mapping2, child2[start:end+1])

    # Step 6: Rebuild into 8x8 tables
    child1_repr = [child1[i:i + 8] for i in range(0, 64, 8)]
    child2_repr = [child2[i:i + 8] for i in range(0, 64, 8)]

    return child1_repr, child2_repr




