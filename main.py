import matplotlib.pyplot as plt
import clonal_selection as cs
import VGG
import json
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler("process.log"), logging.StreamHandler()])




def visualize_results(results):
    iterations = len(results['combined_score'])
    
    # Plot the accuracy for normal and adversarial examples
    plt.figure(figsize=(14, 8))

    plt.subplot(2, 1, 1)
    plt.plot(range(iterations), results['accuracy'], label='Standard Accuracy', color='blue')
    plt.plot(range(iterations), results['adv_accuracy_fgsm'], label='FGSM Accuracy', color='red')
    plt.plot(range(iterations), results['adv_accuracy_pgd'], label='PGD Accuracy', color='green')
    # plt.plot(range(iterations), results['adv_accuracy_deepfool'], label='DeepFool Accuracy', color='purple')
    plt.plot(range(iterations), results['adv_accuracy_de'], label='Differential Evolution Accuracy', color='orange')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy over Iterations')
    plt.legend()
    
    # Plot the combined score
    plt.subplot(2, 1, 2)
    plt.plot(range(iterations), results['combined_score'], label='Combined Score', color='black')
    plt.xlabel('Iteration')
    plt.ylabel('Combined Score')
    plt.title('Combined Score over Iterations')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('results_plot.png')
    plt.show()


    ######## Main ###########

x_train, x_test, y_train, y_test = VGG.data_MNIST()
model = cs.load_model('mnist_vgg16_model.h5')
num_iterations = 6
mutation_rate = 0.35
clone_size = 7
batch_size = 64
epsilon = 0.01
results = {
    'accuracy': [],
    'adv_accuracy_fgsm': [],
    'adv_accuracy_pgd': [],
    # 'adv_accuracy_deepfool': [],
    'adv_accuracy_de': [],
    'combined_score': []
    }


logging.info("Starting initial evaluation")
lst_result_before_clonal_selection = VGG.result(x_test,y_test, model)
logging.info("Initial evaluation done")

logging.info("Starting clonal selection process")
best_model, best_fitness = cs.clonal_selection(x_train, y_train,results, model, num_iterations, mutation_rate, clone_size, batch_size, epsilon)
logging.info(f"Best fitness: {best_fitness}")

try:
    # Try saving the model in HDF5 format
    best_model.save('best_model.h5')
    logging.info("Model saved successfully in HDF5 format.")
except Exception as e:
    logging.error(f"Error saving the model in HDF5 format: {e}")
    try:
        # Try saving the model in SavedModel format
        best_model.save('best_model_saved', save_format='tf')
        logging.info("Model saved successfully in SavedModel format.")
    except Exception as e:
        logging.error(f"Error saving the model in SavedModel format: {e}")
        try:
            # Save the model architecture to JSON
            model_json = best_model.to_json()
            with open("best_model_architecture.json", "w") as json_file:
                json_file.write(model_json)
            
            try:
                # Save the model weights to HDF5
                best_model.save_weights("best_model_weights.h5")
                logging.info("Model weights saved successfully in HDF5 format.")
            except Exception as e:
                logging.error(f"Error saving the model weights in HDF5 format: {e}")
                try:
                    # Save the model weights to JSON
                    weights = best_model.get_weights()
                    weights_json = [w.tolist() for w in weights]
                    with open("best_model_weights.json", "w") as json_file:
                        json.dump(weights_json, json_file)
                    logging.info("Model weights saved successfully in JSON format.")
                except Exception as e:
                    logging.error(f"Error saving the model weights in JSON format: {e}")
        except Exception as e:
            logging.error(f"Error saving the model architecture: {e}")



visualize_results(results)

logging.info("Starting final evaluation")
lst_result_after_clonal_selection = VGG.result(x_test,y_test,best_model)
logging.info("Final evaluation done")


print("Results before clonal selection:", lst_result_before_clonal_selection)
print("Results after clonal selection:", lst_result_after_clonal_selection)
