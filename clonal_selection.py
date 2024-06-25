##################   import   ###########################

import numpy as np
from sklearn.metrics import  accuracy_score
import VGG
from tensorflow.python.keras.models import load_model


##################   Functions   #########################


def eval_solution(solution, data, labels, model):
    # Use the model to predict on the provided data
    predictions = np.argmax(model.predict(data), axis=1)
    true_labels = np.argmax(labels, axis=1)
    accuracy = accuracy_score(true_labels, predictions)
    return accuracy

def clone_and_mutate(solution, mutation_rate):
    # Create a clone of the solution and introduce mutations
    cloned_solution = solution.copy()
    for i in range(len(solution)):
        if np.random.rand() < mutation_rate:
            cloned_solution[i] += np.random.normal(0, 0.1)
    return cloned_solution



def clonal_selection(data, labels, initial_solutions, num_iterations, mutation_rate, clone_size, model):
    # Initialize the population
    population = initial_solutions
    # Iterate over generations
    for _ in range(num_iterations):
        # Evaluate current solutions
        fitness = [eval_solution(sol, data, labels, model) for sol in population]
        # Select elite solutions
        elite_indices = np.argsort(fitness)[-clone_size:]
        elite_solutions = [population[i] for i in elite_indices]
        # Clone and mutate elite solutions
        clones = []
        for sol in elite_solutions:
            for _ in range(clone_size):
                clones.append(clone_and_mutate(sol, mutation_rate))
        # Evaluate clones
        clone_fitness = [eval_solution(sol, data, labels, model) for sol in clones]
        # Select the best solutions (elites + clones)
        all_solutions = population + clones
        all_fitness = fitness + clone_fitness
        best_indices = np.argsort(all_fitness)[-clone_size:]
        best_solutions = [all_solutions[i] for i in best_indices]
        # Update population
        population = best_solutions
    # Select the best overall solution
    fitness = [eval_solution(sol, data, labels, model) for sol in population]  # Re-evaluate final fitness
    best_solution = population[np.argmax(fitness)]
    best_fitness = np.max(fitness)
    return best_solution, best_fitness



################   test   ###################################

x_train, x_test, y_train, y_test = VGG.data_MNIST()
model = load_model('mnist_vgg16_model.h5')

initial_solutions = [np.random.rand(28, 28, 3) for _ in range(5)]
num_iterations = 5 
mutation_rate = 0.1
clone_size = 3 

best_solution, best_fitness = clonal_selection(x_train, y_train, initial_solutions, num_iterations, mutation_rate, clone_size, model)

print("Best Solution:", best_solution)
print("Best Fitness (Accuracy):", best_fitness)
