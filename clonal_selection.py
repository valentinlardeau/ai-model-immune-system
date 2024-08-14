##########################   Import   ##########################

import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.python.keras.models import load_model
import tensorflow as tf
from White_box_attack import fgsm_attack, pgd_attack, differential_evolution_attack
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler("process.log"), logging.StreamHandler()])



##########################   Function   ##########################

"""
brief  : cutting data into batches to reduce the calculation time
input  : data        : data we want to split
         labels      : labels of the data
         batch_size  : size of the batches we want to create
"""
def generate_batches(data, labels, batch_size):
    # loop to separate all the data
    for start_idx in range(0, len(data), batch_size):
        end_idx = min(start_idx + batch_size, len(data))
        yield data[start_idx:end_idx], labels[start_idx:end_idx]



"""
brief  : evaluating the model on all attacks to determine how accurate they are
input  : model : model we want to test
         data   : data to test the model on
         batch_size  : size of the batches we want to create (32 by default)
         epsilon : value of the epsilon on the training
         return_all : boolean to indicate if we need all the return or not
return : return either the combined score or all metrics
"""
def eval_solution(model, data, labels, batch_size=32, epsilon=0.01,return_all=False):
    logging.info("Evaluating solution...")
    true_labels = np.argmax(labels, axis=1)
    all_predictions = []
    all_adv_predictions_fgsm = []
    all_adv_predictions_pgd = []
    # all_adv_predictions_deepfool = []
    all_adv_predictions_de = []
    # generating the batches and convertising them into tensor
    for batch_data, batch_labels in generate_batches(data, labels, batch_size):
        batch_data = tf.convert_to_tensor(batch_data, dtype=tf.float32)
        batch_labels = tf.convert_to_tensor(batch_labels, dtype=tf.float32)
        # Predicting results under normal circumstances and under attack situations

        # Normal predictions
        predictions = model(batch_data, training=False)
        predicted_class = tf.argmax(predictions, axis=1)
        all_predictions.extend(predicted_class.numpy())
        logging.info("Normal predictions done for batch")

        # Adversarial predictions using FGSM
        adv_data_fgsm = fgsm_attack(model, batch_data, batch_labels, epsilon)
        adv_predictions_fgsm = model(adv_data_fgsm, training=False)
        adv_predicted_class_fgsm = tf.argmax(adv_predictions_fgsm, axis=1)
        all_adv_predictions_fgsm.extend(adv_predicted_class_fgsm.numpy())
        logging.info("FGSM predictions done for batch")

        # Adversarial predictions using PGD
        adv_data_pgd = pgd_attack(model, batch_data, batch_labels, epsilon, epsilon_iter=0.01, iterations=10)
        adv_predictions_pgd = model(adv_data_pgd, training=False)
        adv_predicted_class_pgd = tf.argmax(adv_predictions_pgd, axis=1)
        all_adv_predictions_pgd.extend(adv_predicted_class_pgd.numpy())
        logging.info("PGD predictions done for batch")

        # not kept because too long to train 
        # Adversarial predictions using DeepFool
        # for i in range(batch_data.shape[0]):
        #     adv_data_deepfool = deepfool_attack(model, batch_data[i:i+1], batch_labels[i:i+1], max_iter=50, epsilon=epsilon)
        #     adv_predictions_deepfool = model(adv_data_deepfool, training=False)
        #     adv_predicted_class_deepfool = tf.argmax(adv_predictions_deepfool, axis=1)
        #     all_adv_predictions_deepfool.append(adv_predicted_class_deepfool.numpy()[0])
        # print("deepfool predictions done")

        # Adversarial predictions using Differential Evolution
        for i in range(batch_data.shape[0]):
            adv_data_de = differential_evolution_attack(model, batch_data[i], batch_labels[i], epsilon)
            adv_predictions_de = model(tf.expand_dims(adv_data_de, axis=0), training=False)
            adv_predicted_class_de = tf.argmax(adv_predictions_de, axis=1)
            all_adv_predictions_de.append(adv_predicted_class_de.numpy()[0])
        logging.info("Differential Evolution predictions done for batch")

    # Calculate accuracy and adversarial accuracy
    accuracy = accuracy_score(true_labels, all_predictions)
    adv_accuracy_fgsm = accuracy_score(true_labels, all_adv_predictions_fgsm)
    adv_accuracy_pgd = accuracy_score(true_labels, all_adv_predictions_pgd)
    adv_accuracy_de = accuracy_score(true_labels, all_adv_predictions_de)
    combined_score = (accuracy + adv_accuracy_fgsm + adv_accuracy_pgd +  adv_accuracy_de) / 4.0
    logging.info("Combined score calculated")
    # returning the right answer depening of the use case
    if return_all:
        return accuracy, adv_accuracy_fgsm, adv_accuracy_pgd, adv_accuracy_de, combined_score
    else:
        return combined_score


"""
brief  : cloning and applying mutation on the model
input  : model : model we want to work on
         mutation_rate   : data not attacked to compare
         num_classes  : proper to MNIST (hence 10)
return : data attacked as x and y
"""
def clone_and_mutate(model, mutation_rate):
    # cloning the model
    cloned_model = tf.keras.models.clone_model(model)
    cloned_model.set_weights(model.get_weights())
    new_weights = []
    # making mutation on the model
    for layer_weights in cloned_model.get_weights():
        if np.random.rand() < mutation_rate:
            layer_weights += np.random.normal(0, 0.2, layer_weights.shape)
        new_weights.append(layer_weights)
    cloned_model.set_weights(new_weights)
    return cloned_model



"""
brief  : realising the clonal selction algorythm
input  : data          : dataset we work on
         labels        : labels of the dataset we work on
         results       : dictionnary on which we stock the results
         initial_model : model we start with
         num_iteration : number of iteration we are going to do during the training
         mutation_rate : value of the mutation rate we want to use during the training
         clone_size    : size of the cloned model
         batch_size    : size of the batches we want to work on (32 by default)
         epsilon       : value of the epsilon we want to work with
return : return the final best model and its fitness
"""
def clonal_selection(data, labels,results,  initial_model, num_iterations, mutation_rate, clone_size, batch_size=32, epsilon=0.01):
    logging.info("Clonal selection initialization")
    # liste of the population initialised with the initial model
    population = [initial_model]
    # training on the  number of iteration
    for iteration in range(num_iterations):
        logging.info(f"Iteration {iteration+1}/{num_iterations}")
        # calculation of the evaluation of the cloned model
        fitness = [eval_solution(model, data, labels, batch_size, epsilon) for model in population]
        elite_indices = np.argsort(fitness)[-clone_size:]
        elite_solutions = [population[idx] for idx in elite_indices]
        new_population = []
        # working on models
        for model in elite_solutions:
            new_population.append(model)
            # creating clone and mutation 
            for _ in range(clone_size - 1):
                cloned_model = clone_and_mutate(model, mutation_rate)
                new_population.append(cloned_model)
        population = new_population
        # evaluation of the best model to have the best metrics and results
        accuracy, adv_accuracy_fgsm, adv_accuracy_pgd, adv_accuracy_de, combined_score = eval_solution(population[-1], data, labels, batch_size, epsilon, return_all=True)
        results['accuracy'].append(accuracy)
        results['adv_accuracy_fgsm'].append(adv_accuracy_fgsm)
        results['adv_accuracy_pgd'].append(adv_accuracy_pgd)
        results['adv_accuracy_de'].append(adv_accuracy_de)
        results['combined_score'].append(combined_score)
    # keeping best data
    best_idx = np.argmax(fitness)
    best_model = population[best_idx]
    best_fitness = fitness[best_idx]
    return best_model, best_fitness






