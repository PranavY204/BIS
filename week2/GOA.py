import random


def objective_function(x):
    # return max(0, x) 
    return abs(x)


def generate_individual(lower_bound, upper_bound):
    return random.uniform(lower_bound, upper_bound)


def create_population(pop_size, lower_bound, upper_bound):
    return [generate_individual(lower_bound, upper_bound) for _ in range(pop_size)]


def fitness(individual):
    return objective_function(individual)


def select(population, fitnesses):
    # total_fitness = sum(fitnesses)
    # print("Current Fitness scores: ", fitnesses)
    # selected = random.choices(population, weights=fitnesses, k=2)
    return [x for _, x in sorted(zip(fitnesses, population), reverse=True)][:2]
    # return selected


def crossover(parent1, parent2):
    alpha = random.random()  
    child1 = alpha * parent1 + (1 - alpha) * parent2
    child2 = alpha * parent2 + (1 - alpha) * parent1
    return child1, child2


def mutate(individual, mutation_rate, lower_bound, upper_bound):
    if random.random() < mutation_rate:
        return generate_individual(lower_bound, upper_bound)
    return individual


def genetic_algorithm(pop_size, lower_bound, upper_bound, generations, mutation_rate):
    population = create_population(pop_size, lower_bound, upper_bound)
    
    for generation in range(generations):
        fitnesses = [fitness(idx) for idx in population]
     
        new_population = []
        for _ in range(pop_size // 2):
         
            parent1, parent2 = select(population, fitnesses)
            
          
            child1, child2 = crossover(parent1, parent2)
            
           
            child1 = mutate(child1, mutation_rate, lower_bound, upper_bound)
            child2 = mutate(child2, mutation_rate, lower_bound, upper_bound)
            
            new_population.extend([child1, child2])
            population = population[2:]
            # print(population)
        
        population = new_population

       
        best_individual = max(population, key=fitness)
        print(f"Generation {generation + 1}: Best = {best_individual}, Fitness = {fitness(best_individual)}")
    

    return max(population, key=fitness)


pop_size = 20 
lower_bound = -20  
upper_bound = 10  
generations = 10 
mutation_rate = 0.1 


best_solution = genetic_algorithm(pop_size, lower_bound, upper_bound, generations, mutation_rate)
print(f"Best solution: {best_solution}, Fitness: {fitness(best_solution)}")

