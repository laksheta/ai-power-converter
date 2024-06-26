import numpy

def cal_pop_fitness(equation_inputs, pop):
    # Calculating the fitness value of each solution in the current population.
    # The fitness function calulates the sum of products between each input and its corresponding weight.
    fitness = numpy.sum(pop*equation_inputs, axis=1)
    return fitness

def select_mating_pool(pop, fitness, num_parents):
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
    parents = numpy.empty((num_parents, pop.shape[1]))
    for parent_num in range(num_parents):
        min_fitness_idx = numpy.where(fitness == numpy.min(fitness))
        min_fitness_idx = min_fitness_idx[0][0]
        parents[parent_num, :] = pop[min_fitness_idx, :]
        fitness[min_fitness_idx] = numpy.inf  # Set the minimum fitness to infinity to avoid selecting it again
    return parents

import numpy as np

# def adapt_crossover_point_for_stagnation(crossover_point):
#     # Modify crossover point for adaptive crossover during stagnation
#     # Example: Increment the crossover point by a fixed amount
#     return crossover_point + 1


# def detect_stagnation(best_fitnesses, stagnation_threshold=3):
#     if len(best_fitnesses) < stagnation_threshold:
#         return False
    
#     # Check if best fitness values have remained unchanged for stagnation_threshold generations
#     for i in range(len(best_fitnesses) - stagnation_threshold, len(best_fitnesses) - 1):
#         if best_fitnesses[i] != best_fitnesses[i+1]:
#             return False
    
#     return True

# def calculate_crossover_point(parents):
#     # Calculate crossover point based on some heuristic or default value
#     return int(parents.shape[1] / 2)

# def adaptive_crossover(parents, offspring_size,best_fitnesses):
#     offspring = np.empty(offspring_size)
#     crossover_point = calculate_crossover_point(parents)
    
#     # Check if stagnation is detected and adapt crossover accordingly
#     if detect_stagnation(best_fitnesses):
#         # Apply adaptive crossover strategy
#         crossover_point = adapt_crossover_point_for_stagnation(crossover_point)
    
#     for k in range(offspring_size[0]):
#         parent1_idx = k % parents.shape[0]
#         parent2_idx = (k + 1) % parents.shape[0]
#         offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
#         offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    
#     return offspring



def crossover(parents, offspring_size):
    offspring = numpy.empty(offspring_size)
    # The point at which crossover takes place between two parents. Usually, it is at the center.
    crossover_point = numpy.uint8(offspring_size[1]/2)

    for k in range(offspring_size[0]):
        # Index of the first parent to mate.
        parent1_idx = k%parents.shape[0]
        # Index of the second parent to mate.
        parent2_idx = (k+1)%parents.shape[0]
        # The new offspring will have its first half of its genes taken from the first parent.
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        # The new offspring will have its second half of its genes taken from the second parent.
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring


def crossover_inner(parents, offspring_size, crossover_point):
    offspring = np.empty(offspring_size)
    
    for k in range(offspring_size[0]):
        # Print the crossover point for debugging purposes (optional)
        print(f"Offspring {k}: Crossover point is {crossover_point}")

        # Index of the first parent to mate.
        parent1_idx = k % parents.shape[0]
        # Index of the second parent to mate.
        parent2_idx = (k + 1) % parents.shape[0]
        # The new offspring will have its first half of its genes taken from the first parent.
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        # The new offspring will have its second half of its genes taken from the second parent.
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
        
    return offspring



def detect_stagnation(best_fitnesses, stagnation_threshold=3):
    if len(best_fitnesses) < stagnation_threshold:
        return False
    
    # Check if best fitness values have remained unchanged for stagnation_threshold generations
    return all(best_fitnesses[i] == best_fitnesses[i+1] for i in range(len(best_fitnesses) - stagnation_threshold, len(best_fitnesses) - 1))


def crossover_adaptive(parents, offspring_size, best_fitnesses, stagnation_threshold=3):
    offspring = np.empty(offspring_size)
    last_crossover_point = None  # Initialize the last crossover point

    stagnation = detect_stagnation(best_fitnesses, stagnation_threshold)
    
    if stagnation:
        while True:
            # Randomly select crossover point between 1 and offspring_size[1] - 1
            crossover_point = np.random.randint(1, offspring_size[1])
            # Ensure the new crossover point is different from the last one
            if crossover_point != last_crossover_point:
                break
    else:
        # Use default crossover point (halfway)
        crossover_point = offspring_size[1] // 2

    last_crossover_point = crossover_point

    print("Crossover point:", crossover_point)
    
    for k in range(offspring_size[0]):
        parent1_idx = k % parents.shape[0]
        parent2_idx = (k + 1) % parents.shape[0]
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    
    return offspring



def mutation(offspring_crossover, mutation_ranges, L_range, C_range, fsw_range, t_dt_range, num_mutations=1):
    mutated_offspring = np.copy(offspring_crossover)
    num_genes = mutated_offspring.shape[1]
    
    # Mutation changes a number of genes as defined by the num_mutations argument.
    # The changes are random.
    for idx in range(mutated_offspring.shape[0]):  # Iterate over each individual in the population
        for _ in range(num_mutations):  # Perform the specified number of mutations for each individual
            for gene_idx in range(num_genes):  # Iterate over each gene in the individual
                # Determine whether to add or subtract the mutation value randomly
                add_or_subtract = np.random.choice([-1, 1])
                
                # Obtain the mutation range for the current gene
                mutation_range = mutation_ranges[gene_idx]
                
                if gene_idx == 0:  # Check if the gene is for L
                    # For L, scale the mutation range to match the integer range
                    upper_bound = int(mutation_range[1] * 1e6)  # Scale to match the integer range
                    mutation_value = add_or_subtract * np.random.randint(1, upper_bound + 1)
                    mutation_value /= 1e6  # Scale back to the original range if necessary
                elif gene_idx == 1:  # Check if the gene is for C
                    # For C, scale the mutation range to match the integer range
                    upper_bound = int(mutation_range[1] * 1e6)  # Scale to match the integer range
                    mutation_value = add_or_subtract * np.random.randint(1, upper_bound + 1)
                    mutation_value /= 1e6  # Scale back to the original range if necessary
                elif gene_idx == 2:  # Check if the gene is for fsw
                    # For fsw, directly generate a random integer within the specified range
                    mutation_value = add_or_subtract * np.random.randint(mutation_range[0], mutation_range[1] + 1)
                else:  # Check if the gene is for t_dt
                    # For t_dt, directly generate a random integer within the specified range
                    mutation_value = add_or_subtract * np.random.randint(int(t_dt_range[0] * 1e6), int(t_dt_range[1] * 1e6) + 1)
                    mutation_value /= 1e6  # Scale back to the original range if necessary
                
                # Apply mutation to the gene
                mutated_offspring[idx, gene_idx] += mutation_value
                
                # Ensure the mutated value remains within the specified range
                if gene_idx == 0:  # L gene
                    mutated_offspring[idx, gene_idx] = np.clip(mutated_offspring[idx, gene_idx], L_range[0], L_range[1])
                elif gene_idx == 1:  # C gene
                    mutated_offspring[idx, gene_idx] = np.clip(mutated_offspring[idx, gene_idx], C_range[0], C_range[1])
                elif gene_idx == 2:  # fsw gene
                    mutated_offspring[idx, gene_idx] = np.clip(mutated_offspring[idx, gene_idx], fsw_range[0], fsw_range[1])
                else:  # t_dt gene
                    mutated_offspring[idx, gene_idx] = np.clip(mutated_offspring[idx, gene_idx], t_dt_range[0], t_dt_range[1])
    
    return mutated_offspring

def mutation_inner(offspring_crossover, mutation_ranges, L_range, C_range, fsw_range, t_dt_range, num_mutations=1):
    mutated_offspring = np.copy(offspring_crossover)
    num_genes = mutated_offspring.shape[1]
    
    for idx in range(mutated_offspring.shape[0]):  # Iterate over each individual in the population
        mutation_indices = np.random.choice(num_genes, num_mutations, replace=False)  # Ensure unique genes are mutated
        
        # print(f"\nIndividual {idx}:")
        # print(f"  Original: {mutated_offspring[idx]}")
        # print(f"  Mutation Indices: {mutation_indices}")
        
        for gene_idx in mutation_indices:  # Perform the specified number of mutations for each individual
            # Determine whether to add or subtract the mutation value randomly
            add_or_subtract = np.random.choice([-1, 1])
            
            # Obtain the mutation range for the current gene
            mutation_range = mutation_ranges[gene_idx]
            
            if gene_idx == 0:  # If the gene is for L
                upper_bound = int((L_range[1] - L_range[0]) * 1e6)
                mutation_value = add_or_subtract * np.random.randint(1, upper_bound + 1)
                mutation_value /= 1e6  # Scale back to the original range
            elif gene_idx == 1:  # If the gene is for C
                upper_bound = int((C_range[1] - C_range[0]) * 1e6)
                mutation_value = add_or_subtract * np.random.randint(1, upper_bound + 1)
                mutation_value /= 1e6  # Scale back to the original range
            elif gene_idx == 2:  # If the gene is for fsw
                mutation_value = add_or_subtract * np.random.randint(int(fsw_range[0] / 1e3), int(fsw_range[1] / 1e3) + 1) * 1e3
            else:  # If the gene is for t_dt
                upper_bound = int((t_dt_range[1] - t_dt_range[0]) * 1e6)
                if upper_bound > 0:
                    mutation_value = add_or_subtract * np.random.randint(1, upper_bound + 1)
                    mutation_value /= 1e6  # Scale back to the original range
                else:
                    mutation_value = add_or_subtract * (t_dt_range[1] - t_dt_range[0]) / 10  # A small fixed value for mutation
            
            # print(f"    Gene {gene_idx} Mutation: {add_or_subtract} * {mutation_value} = {add_or_subtract * mutation_value}")
            
            # Apply mutation to the gene
            mutated_offspring[idx, gene_idx] += mutation_value
            
            # Ensure the mutated value remains within the specified range
            if gene_idx == 0:  # L gene
                mutated_offspring[idx, gene_idx] = np.clip(mutated_offspring[idx, gene_idx], L_range[0], L_range[1])
            elif gene_idx == 1:  # C gene
                mutated_offspring[idx, gene_idx] = np.clip(mutated_offspring[idx, gene_idx], C_range[0], C_range[1])
            elif gene_idx == 2:  # fsw gene
                mutated_offspring[idx, gene_idx] = np.clip(mutated_offspring[idx, gene_idx], fsw_range[0], fsw_range[1])
            else:  # t_dt gene
                mutated_offspring[idx, gene_idx] = np.clip(mutated_offspring[idx, gene_idx], t_dt_range[0], t_dt_range[1])
        
        # print(f"  Mutated: {mutated_offspring[idx]}")
    
    return mutated_offspring



