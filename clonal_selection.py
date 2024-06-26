# ##################   import   ###########################

# import numpy as np
# from sklearn.metrics import accuracy_score
# import VGG
# from tensorflow.python.keras.models import load_model
# import tensorflow as tf


# ##################   Functions   #########################

# def generate_batches(data, labels, batch_size):
#     for start_idx in range(0, len(data), batch_size):
#         end_idx = min(start_idx + batch_size, len(data))
#         yield data[start_idx:end_idx], labels[start_idx:end_idx]

# def eval_solution(model, data, labels, batch_size=32):
#     print("evaluation starting")
#     true_labels = np.argmax(labels, axis=1)
#     all_predictions = []

#     for batch_data, batch_labels in generate_batches(data, labels, batch_size):
#         # Get predicted class label
#         predictions = model(batch_data, training=False)
#         predicted_class = tf.argmax(predictions, axis=1)
#         all_predictions.extend(predicted_class.numpy())

#     # Calculate accuracy
#     accuracy = accuracy_score(true_labels, all_predictions)
#     return accuracy

# def clone_and_mutate(solution, mutation_rate):
#     cloned_solution = solution.copy()
#     for i in range(len(solution)):
#         if np.random.rand() < mutation_rate:
#             cloned_solution[i] += np.random.normal(0, 0.1)
#     return cloned_solution

# def clonal_selection(data, labels, initial_solutions, num_iterations, mutation_rate, clone_size, model, batch_size=32):
#     print("clonal selection init")
#     population = initial_solutions
#     for _ in range(num_iterations):
#         fitness = [eval_solution(model, data, labels, batch_size) for sol in population]
#         elite_indices = np.argsort(fitness)[-clone_size:]
#         elite_solutions = [population[i] for i in elite_indices]
#         clones = []
#         for sol in elite_solutions:
#             for _ in range(clone_size):
#                 clones.append(clone_and_mutate(sol, mutation_rate))
#         clone_fitness = [eval_solution(model, data, labels, batch_size) for sol in clones]
#         all_solutions = population + clones
#         all_fitness = fitness + clone_fitness
#         best_indices = np.argsort(all_fitness)[-clone_size:]
#         best_solutions = [all_solutions[i] for i in best_indices]
#         population = best_solutions
#     fitness = [eval_solution(model, data, labels, batch_size) for sol in population]
#     best_solution = population[np.argmax(fitness)]
#     best_fitness = np.max(fitness)
#     return best_solution, best_fitness


# ################   test   ###################################

# x_train, x_test, y_train, y_test = VGG.data_MNIST()
# model = load_model('mnist_vgg16_model.h5')

# initial_solutions = [np.random.rand(32, 32, 3) for _ in range(5)]
# print(np.shape(x_test), np.shape(x_train))
# num_iterations = 5 
# mutation_rate = 0.1
# clone_size = 3 
# batch_size = 32

# best_solution, best_fitness = clonal_selection(x_train, y_train, initial_solutions, num_iterations, mutation_rate, clone_size, model, batch_size)

# print("Best Solution:", best_solution)
# print("Best Solution:", type(best_solution))
# print("Best Fitness (Accuracy):", best_fitness)


import numpy as np
from sklearn.metrics import accuracy_score
import VGG
from tensorflow.python.keras.models import load_model
import tensorflow as tf
from White_box_attack import fgsm_attack, pgd_attack, deepfool_attack, differential_evolution_attack

def generate_batches(data, labels, batch_size):
    for start_idx in range(0, len(data), batch_size):
        end_idx = min(start_idx + batch_size, len(data))
        yield data[start_idx:end_idx], labels[start_idx:end_idx]

def eval_solution(model, data, labels, batch_size=32, epsilon=0.01):
    true_labels = np.argmax(labels, axis=1)
    all_predictions = []
    all_adv_predictions_fgsm = []
    all_adv_predictions_pgd = []
    all_adv_predictions_deepfool = []
    all_adv_predictions_de = []

    for batch_data, batch_labels in generate_batches(data, labels, batch_size):
        batch_data = tf.convert_to_tensor(batch_data, dtype=tf.float32)
        batch_labels = tf.convert_to_tensor(batch_labels, dtype=tf.float32)

        # Normal predictions
        predictions = model(batch_data, training=False)
        predicted_class = tf.argmax(predictions, axis=1)
        all_predictions.extend(predicted_class.numpy())
        print("normal predictions done")

        # Adversarial predictions using FGSM
        adv_data_fgsm = fgsm_attack(model, batch_data, batch_labels, epsilon)
        adv_predictions_fgsm = model(adv_data_fgsm, training=False)
        adv_predicted_class_fgsm = tf.argmax(adv_predictions_fgsm, axis=1)
        all_adv_predictions_fgsm.extend(adv_predicted_class_fgsm.numpy())
        print("fgsm predictions done")


        # Adversarial predictions using PGD
        adv_data_pgd = pgd_attack(model, batch_data, batch_labels, epsilon, epsilon_iter=0.01, iterations=10)
        adv_predictions_pgd = model(adv_data_pgd, training=False)
        adv_predicted_class_pgd = tf.argmax(adv_predictions_pgd, axis=1)
        all_adv_predictions_pgd.extend(adv_predicted_class_pgd.numpy())
        print("PGD predictions done")


        # Adversarial predictions using DeepFool
        for i in range(batch_data.shape[0]):
            adv_data_deepfool = deepfool_attack(model, batch_data[i:i+1], batch_labels[i:i+1], max_iter=50, epsilon=epsilon)
            adv_predictions_deepfool = model(adv_data_deepfool, training=False)
            adv_predicted_class_deepfool = tf.argmax(adv_predictions_deepfool, axis=1)
            all_adv_predictions_deepfool.append(adv_predicted_class_deepfool.numpy()[0])
        print("deepfool predictions done")


        # Adversarial predictions using Differential Evolution
        for i in range(batch_data.shape[0]):
            adv_data_de = differential_evolution_attack(model, batch_data[i], batch_labels[i], epsilon)
            adv_predictions_de = model(tf.expand_dims(adv_data_de, axis=0), training=False)
            adv_predicted_class_de = tf.argmax(adv_predictions_de, axis=1)
            all_adv_predictions_de.append(adv_predicted_class_de.numpy()[0])
        print("Differential evolution predictions done")
        

    # Calculate accuracy and adversarial accuracy
    accuracy = accuracy_score(true_labels, all_predictions)
    adv_accuracy_fgsm = accuracy_score(true_labels, all_adv_predictions_fgsm)
    adv_accuracy_pgd = accuracy_score(true_labels, all_adv_predictions_pgd)
    adv_accuracy_deepfool = accuracy_score(true_labels, all_adv_predictions_deepfool)
    adv_accuracy_de = accuracy_score(true_labels, all_adv_predictions_de)

    # Return a combined score (you can adjust the weight as needed)
    combined_score = (accuracy + adv_accuracy_fgsm + adv_accuracy_pgd + adv_accuracy_deepfool + adv_accuracy_de) / 5.0
    # combined_score = (accuracy + adv_accuracy_fgsm + adv_accuracy_pgd +  adv_accuracy_de) / 4.0
    print("combined score calculated")
    return combined_score

def clone_and_mutate(model, mutation_rate):
    cloned_model = tf.keras.models.clone_model(model)
    cloned_model.set_weights(model.get_weights())
    new_weights = []
    for layer_weights in cloned_model.get_weights():
        if np.random.rand() < mutation_rate:
            layer_weights += np.random.normal(0, 0.1, layer_weights.shape)
        new_weights.append(layer_weights)
    cloned_model.set_weights(new_weights)
    return cloned_model

def clonal_selection(data, labels, initial_model, num_iterations, mutation_rate, clone_size, batch_size=32, epsilon=0.01):
    print("Clonal selection initialization")
    population = [initial_model]
    for _ in range(num_iterations):
        fitness = [eval_solution(model, data, labels, batch_size, epsilon) for model in population]
        elite_indices = np.argsort(fitness)[-clone_size:]
        elite_solutions = [population[idx] for idx in elite_indices]
        new_population = []
        for model in elite_solutions:
            new_population.append(model)
            for _ in range(clone_size - 1):
                cloned_model = clone_and_mutate(model, mutation_rate)
                new_population.append(cloned_model)
        population = new_population
    best_idx = np.argmax(fitness)
    best_model = population[best_idx]
    best_fitness = fitness[best_idx]
    return best_model, best_fitness




###############   Test   ############
x_train, x_test, y_train, y_test = VGG.data_MNIST()
model = load_model('mnist_vgg16_model.h5')
num_iterations = 5
mutation_rate = 0.2
clone_size = 3
batch_size = 32
epsilon = 0.01

best_model, best_fitness = clonal_selection(x_train, y_train, model, num_iterations, mutation_rate, clone_size, batch_size, epsilon)
print(f"Best fitness: {best_fitness}")
model.save('best_model.h5')

