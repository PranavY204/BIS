import random
import numpy as np

# Define terminal set (input variables and constants) and function set
TERMINALS = ["x"]
FUNCTIONS = ["+", "-", "*", "/"]

# Define the GEP parameters
POPULATION_SIZE = 50
GENE_LENGTH = 10
NUM_GENERATIONS = 50
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.7
ELITISM_COUNT = 2

# Target function for symbolic regression (example: y = x^2 + 2x + 1)
def target_function(x):
    return x**2 + 2*x + 1

# Generate random genes
def random_gene():
    """Create a random gene (linear string representation)."""
    gene = []
    for _ in range(GENE_LENGTH):
        if random.random() < 0.5:
            gene.append(random.choice(TERMINALS))
        else:
            gene.append(random.choice(FUNCTIONS))
    return gene

# Convert a gene to an expression tree and evaluate it
def evaluate_gene(gene, x):
    """Convert a gene into an expression and evaluate it for a given x."""
    stack = []
    for symbol in gene:
        if symbol in TERMINALS:
            stack.append(x)
        elif symbol in FUNCTIONS:
            if len(stack) < 2:
                return None  # Invalid expression
            b = stack.pop()
            a = stack.pop()
            try:
                if symbol == "+":
                    stack.append(a + b)
                elif symbol == "-":
                    stack.append(a - b)
                elif symbol == "*":
                    stack.append(a * b)
                elif symbol == "/":
                    stack.append(a / b if b != 0 else 1)
            except ZeroDivisionError:
                return None
        else:
            stack.append(float(symbol))
    return stack[0] if len(stack) == 1 else None

# Fitness function
def fitness(gene, x_values, y_values):
    """Compute fitness as the inverse of error."""
    error = 0
    for x, y in zip(x_values, y_values):
        result = evaluate_gene(gene, x)
        if result is None:
            return float('inf')
        error += abs(result - y)
    return 1 / (1 + error)

# Perform mutation
def mutate(gene):
    """Mutate a gene with a given mutation rate."""
    new_gene = gene[:]
    for i in range(len(gene)):
        if random.random() < MUTATION_RATE:
            if random.random() < 0.5:
                new_gene[i] = random.choice(TERMINALS)
            else:
                new_gene[i] = random.choice(FUNCTIONS)
    return new_gene

# Perform crossover
def crossover(parent1, parent2):
    """Perform single-point crossover between two parents."""
    if random.random() < CROSSOVER_RATE:
        point = random.randint(1, len(parent1) - 1)
        return parent1[:point] + parent2[point:]
    return parent1

# Main GEP algorithm
def gep():
    # Generate initial population
    population = [random_gene() for _ in range(POPULATION_SIZE)]
    
    # Training data
    x_values = np.linspace(-10, 10, 100)
    y_values = target_function(x_values)

    best_gene = None
    best_fitness = float('-inf')

    for generation in range(NUM_GENERATIONS):
        # Evaluate fitness of each individual
        fitness_values = [fitness(gene, x_values, y_values) for gene in population]

        # Find the best gene
        max_fitness = max(fitness_values)
        if max_fitness > best_fitness:
            best_fitness = max_fitness
            best_gene = population[fitness_values.index(max_fitness)]

        print(f"Generation {generation + 1}: Best Fitness = {best_fitness}")

        # Selection (elitism)
        sorted_indices = np.argsort(fitness_values)[::-1]
        new_population = [population[i] for i in sorted_indices[:ELITISM_COUNT]]

        # Create next generation
        while len(new_population) < POPULATION_SIZE:
            parent1 = population[random.choice(sorted_indices)]
            parent2 = population[random.choice(sorted_indices)]
            offspring = crossover(parent1, parent2)
            offspring = mutate(offspring)
            new_population.append(offspring)

        population = new_population

    print("Best Gene:", best_gene)
    return best_gene

# Run the GEP algorithm
if __name__ == "__main__":
    best_solution = gep()
    print("Best Solution Representation:", best_solution)
