###############   Import   #####################

import matplotlib.pyplot as plt
import clonal_selection as cs
import VGG
import logging
import pickle
from keras.src.utils import to_categorical
from tensorflow.python.keras.models import load_model
import numpy as np
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler("process.log"), logging.StreamHandler()])


##############   Function   ####################


"""
brief  : loading the dataset that has already been attacked
input  : dataset_path : path to the dataset i want to load
         original_y   : data not attacked to compare
         num_classes  : proper to MNIST (hence 10)
return : data attacked as x and y
"""
def load_combined_attacked_dataset(dataset_path, original_y, num_classes=10):
    with open(dataset_path, 'rb') as f:
        x = pickle.load(f)
    # Check if original_y is in categorical format
    if original_y.ndim == 1 or original_y.shape[1] == 1:
        original_y = to_categorical(original_y, num_classes)
    # Ensure y is duplicated correctly
    if x.shape[0] > original_y.shape[0]:
        num_repeats = x.shape[0] // original_y.shape[0] + 1
        y = np.tile(original_y, (num_repeats, 1))[:x.shape[0]]
    else:
        y = original_y[:x.shape[0]]
    return x, y


"""
brief  : saving the final model
input  : model        : model we want to save
         model_name   : name under which we want to save it
"""
def save_model(model, model_name):
    # Due to compatibility issues, sometimes i need the h5 version...
    try:
        model.save(f'{model_name}.h5')
        logging.info(f"Model saved successfully in HDF5 format as {model_name}.h5")
    except Exception as e:
        logging.error(f"Error saving the model in HDF5 format: {e}")
    # ...and sometimes i need the .keras version
    try:
        model.save(f'{model_name}.keras')
        logging.info(f"Model saved successfully in Keras format as {model_name}.keras")
    except Exception as e:
        logging.error(f"Error saving the model in Keras format: {e}")


"""
brief  : creating table and graph to see the results
input  : results : dictionnary of the results to see create the graph
"""
def visualize_results(results):
    iterations = len(results['combined_score'])
    # check to see if there iare results
    if iterations == 0:
        logging.error("No data available to plot.")
        return
    # creation of the graphs
    plt.figure(figsize=(14, 8))
    plt.subplot(2, 1, 1)
    # regular case (multiple iterations)
    if iterations > 1:
        plt.plot(range(iterations), results['accuracy'], label='Standard Accuracy', color='blue')
        plt.plot(range(iterations), results['adv_accuracy_fgsm'], label='FGSM Accuracy', color='red')
        plt.plot(range(iterations), results['adv_accuracy_pgd'], label='PGD Accuracy', color='green')
        plt.plot(range(iterations), results['adv_accuracy_de'], label='Differential Evolution Accuracy', color='orange')
    # preventing error cases
    else:
        plt.scatter([0], results['accuracy'], label='Standard Accuracy', color='blue')
        plt.scatter([0], results['adv_accuracy_fgsm'], label='FGSM Accuracy', color='red')
        plt.scatter([0], results['adv_accuracy_pgd'], label='PGD Accuracy', color='green')
        plt.scatter([0], results['adv_accuracy_de'], label='Differential Evolution Accuracy', color='orange')
    # labeling the first graph
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy over Iterations')
    plt.legend()
    plt.subplot(2, 1, 2)
    if iterations > 1:
        plt.plot(range(iterations), results['combined_score'], label='Combined Score', color='black')
    else:
        plt.scatter([0], results['combined_score'], label='Combined Score', color='black')
    # labeling the second graph
    plt.xlabel('Iteration')
    plt.ylabel('Combined Score')
    plt.title('Combined Score over Iterations')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results_plot.png')
    plt.show()


######## Main ###########

# Getting data and model
x_train, x_test, y_train, y_test = VGG.data_MNIST()
x_attacked, y_attacked = load_combined_attacked_dataset("attacked_datasets/combined_attacked_dataset.pkl", y_train)
model = cs.load_model('mnist_vgg16_model.h5')

# setting hyperparameters
num_iterations = 3
mutation_rate = 0.35
clone_size = 2
batch_size = 128
epsilon = 0.01
results = {
    'accuracy': [],
    'adv_accuracy_fgsm': [],
    'adv_accuracy_pgd': [],
    # 'adv_accuracy_deepfool': [],
    'adv_accuracy_de': [],
    'combined_score': []
    }


# Evaluation before clonal selection on both attacked and regular datasets
logging.info("Starting initial evaluation")
lst_result_before_clonal_selection = VGG.result(x_test,y_test, model)
lst_result_before_clonal_selection_attacked = VGG.result(x_attacked,y_attacked, model)

# begining of the process
logging.info("Initial evaluation done")
logging.info("Starting clonal selection process")
best_model, best_fitness = cs.clonal_selection(x_train, y_train,results, model, num_iterations, mutation_rate, clone_size, batch_size, epsilon)
logging.info(f"Best fitness: {best_fitness}")


# Save the model
save_model(model, 'best_model')

# visualization of results
logging.info("visualising results")
visualize_results(results)


# evaluation of the new model on both attacked and regular datasets
logging.info("Starting final evaluation")
lst_result_after_clonal_selection = VGG.result(x_test,y_test,best_model)
lst_result_after_clonal_selection_attacked = VGG.result(x_attacked,y_attacked,best_model)
logging.info("Final evaluation done")


# Print of the results
print("Results before clonal selection on regular dataset:", lst_result_before_clonal_selection)
print("Results after clonal selection on regular dataset:", lst_result_after_clonal_selection)

print("Results before clonal selection on attacked dataset:", lst_result_before_clonal_selection_attacked)
print("Results after clonal selection on attacked dataset:", lst_result_after_clonal_selection)
