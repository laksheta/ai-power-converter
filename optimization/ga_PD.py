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

import numpy as np

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



