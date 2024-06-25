##################    Import   #######################################
import numpy as np
from tensorflow.python.keras.models import load_model
from White_box_attack import fgsm_attack, pgd_attack, deepfool_attack
import tensorflow as tf


##################   Function  #######################################



"""
brief  : generating random images as auto-reactive cells
input  : model : model we want to attack
         attack : white box attack we want
         cell_number : number of cell to generate
         image_size : size of the image generated (28 * 28 for MNIST)
return : np table of random adversarial image
"""
def generate_auto_reactive_cell(model, attack, cell_number, image_size, epsilon):
  images = np.random.rand(cell_number, image_size, image_size, 3)  # Generate random images
  auto_reactive_cells = []
  for i in range(cell_number):
    # Convert image to a tf.Tensor
    image_tensor = tf.convert_to_tensor(images[i][np.newaxis, ...])
    # Get predicted class label
    predicted_class = tf.argmax(model.predict(image_tensor), axis=1)
    # One-hot encode the predicted class label
    y = tf.one_hot(predicted_class, model.output_shape[-1])
    # Generate adversarial image using the attack function
    adversarial_image = attack(model, image_tensor, y, epsilon=epsilon)
    # Convert adversarial image back to NumPy array
    adversarial_image = adversarial_image.numpy()
    auto_reactive_cells.append(adversarial_image)
  return np.array(auto_reactive_cells)




"""
brief  : retrieving the deficient neurons in the model through negative selection algorithm
input  : model : model we want to test
         auto_reactive_cells : simulated input
         activation
return : a list of harmful neurons
"""
def negative_selection(model, auto_reactive_cells, activation_threshold):
  harmful_neurons = []
  for image in auto_reactive_cells:
    # Predict activation on the image 
    activation = model.predict(image[...])

    for i, activation_neuron in enumerate(activation[0]):
      if activation_neuron > activation_threshold:
        harmful_neurons.append(i)
  return harmful_neurons




def remove(model, harmful_neurons):
    # Iterate through each layer in the model
    for layer in model.layers:
        # Check if the layer has any weights
        if layer.get_weights():
            # Check if the layer has exactly two parameters: weights and biases
            if len(layer.get_weights()) == 2:
                weights, biases = layer.get_weights()
                # Zero out the weights of the harmful neurons
                for neuron_index in harmful_neurons:
                    if neuron_index < weights.shape[-1]:  # Ensure we are within the bounds of the layer's neurons
                        weights[:, neuron_index] = 0
                # Set the modified weights back to the layer
                layer.set_weights([weights, biases])
            else:
                print(f"Skipping layer {layer.name} with {len(layer.get_weights())} parameters")





##################   Test   ############################################################

model = load_model('mnist_vgg16_model.h5')

# Générer des cellules auto-réactives
cellules_auto_reactives = generate_auto_reactive_cell(model, fgsm_attack, 100, 32, 0.1)
bad_neurons = negative_selection(model, cellules_auto_reactives, 0.5)
print(bad_neurons)
remove(model, bad_neurons)
