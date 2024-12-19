import random
import numpy as np


def fitness_function(x):
    return np.sum(x**2 - 10 * np.cos(2 * np.pi * x) + 10)  # Negative to maximize


def create_population(population_size, gene_length):
    population = []
    for _ in range(population_size):
        chromosome = [random.uniform(-5, 5) for _ in range(gene_length)]
        population.append(chromosome)
    return population


def selection(population, fitness_scores):
    selected_parents = []
    for _ in range(2):
        max_fitness = max(fitness_scores)
        max_index = fitness_scores.index(max_fitness)
        selected_parents.append(population[max_index])
        fitness_scores[max_index] = -float("inf")  # Prevent repeated selection
    return selected_parents


def crossover(parent1, parent2, crossover_rate):
    child1, child2 = parent1.copy(), parent2.copy()
    if random.random() < crossover_rate:
        crossover_point = random.randint(0, len(parent1) - 1)
        child1[crossover_point:], child2[crossover_point:] = (
            parent2[crossover_point:],
            parent1[crossover_point:],
        )
    return child1, child2


def mutation(individual, mutation_rate):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] += random.uniform(-0.5, 0.5)
    return individual


def genetic_algorithm(
    population_size, gene_length, num_generations, crossover_rate, mutation_rate
):
    population = create_population(population_size, gene_length)
    for generation in range(num_generations):
        fitness_scores = [fitness_function(individual[0]) for individual in population]
        best_fitness = max(fitness_scores)
        best_x = population[fitness_scores.index(best_fitness)][0]
        print(
            f"Generation {generation+1}: Best fitness = {best_fitness:.4f}, Best x = {best_x:.4f}"
        )
        new_population = []
        for _ in range(population_size // 2):
            parent1, parent2 = selection(population, fitness_scores.copy())
            child1, child2 = crossover(parent1, parent2, crossover_rate)
            child1 = mutation(child1, mutation_rate)
            child2 = mutation(child2, mutation_rate)
            new_population.extend([child1, child2])
        population = new_population
    best_individual = max(population, key=lambda x: fitness_function(x[0]))
    return best_individual[0]


# Example usage
best_x = genetic_algorithm(
    population_size=100,
    gene_length=1,
    num_generations=100,
    crossover_rate=0.8,
    mutation_rate=0.1,
)
print("Pranav Y - 1BM22CS204")
print("Final Solution: Best x =", best_x)
print("Final Solution: Best fitness =", fitness_function(best_x))
