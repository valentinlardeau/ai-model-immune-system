###############   import   ################################


import tensorflow as tf
import numpy as np
from scipy.optimize import differential_evolution

##############   Function   #############################



def fgsm_attack(model, x, y, epsilon=0.1): # Fast Gradient Sign Method
    with tf.GradientTape() as tape:
        tape.watch(x)
        prediction = model(x)
        loss = tf.keras.losses.categorical_crossentropy(y, prediction)
    gradient = tape.gradient(loss, x)
    signed_grad = tf.sign(gradient)
    adv_x = x + epsilon * signed_grad
    adv_x = tf.clip_by_value(adv_x, 0, 1)  
    return adv_x



def pgd_attack(model, x, y, epsilon=0.1, epsilon_iter=0.01, iterations=10): #Projected Gradient Descent
    adv_x = tf.identity(x)  
    for _ in range(iterations):
        with tf.GradientTape() as tape:
            tape.watch(adv_x)
            prediction = model(adv_x)
            loss = tf.keras.losses.categorical_crossentropy(y, prediction)
        gradient = tape.gradient(loss, adv_x)
        signed_grad = tf.sign(gradient)
        adv_x = adv_x + epsilon_iter * signed_grad
        adv_x = tf.clip_by_value(adv_x, x - epsilon, x + epsilon)  
        adv_x = tf.clip_by_value(adv_x, 0, 1)  
    return adv_x


"""
brief  : DeepFool attack implementation.
input  : model (keras.Model): Target model to be attacked.
         x (numpy.ndarray): Input image.
         y (numpy.ndarray): True label.
         max_iter (int): Maximum number of iterations.
         epsilon (float): Perturbation scale.   
return : numpy.ndarray: Adversarial example.
"""
def deepfool_attack(model, x, y, max_iter=50, epsilon=1e-2):
    x_adv = tf.identity(x)  # Initialize adversarial example with the original image
    orig_out = model.predict(tf.expand_dims(x_adv, axis=0))  # Predict the original image  
    def compute_gradients(x, y):
        with tf.GradientTape() as tape:
            tape.watch(x)
            output = model(x)
            loss = tf.keras.losses.categorical_crossentropy(y, output)
        return tape.gradient(loss, x), output 
    for i in range(max_iter):
        gradients, output = compute_gradients(tf.expand_dims(x_adv, axis=0), y)   
        perturbation = np.inf
        adversarial_class = np.argmax(output[0])
        for j in range(model.output_shape[1]):
            if j != adversarial_class:
                adversarial_gradient = gradients[0][:, adversarial_class] - gradients[0][:, j]
                w = adversarial_gradient.numpy()
                f = output[0][adversarial_class] - output[0][j]
                perturbation_j = np.abs(f) / np.linalg.norm(w.flatten())
                if perturbation_j < perturbation:
                    perturbation = perturbation_j
                    w_optimal = w
                    f_optimal = f
        perturbation = np.minimum(epsilon, perturbation)
        x_adv = tf.clip_by_value(x_adv + perturbation * w_optimal, 0.0, 1.0)  # Clip to ensure pixel values stay in [0, 1]
        # Check if the prediction changed
        if np.argmax(model.predict(tf.expand_dims(x_adv, axis=0))) != adversarial_class:
            break
    return x_adv



def differential_evolution_attack(model, x, y, epsilon=0.1):
    def loss_fn(x_adv):
        x_adv = np.reshape(x_adv, x.shape)
        prediction = model.predict(np.array([x_adv]))
        return tf.keras.losses.categorical_crossentropy(y, prediction).numpy()
    bounds = [(x - epsilon, x + epsilon) for x in np.reshape(x, (-1,))] 
    result = differential_evolution(loss_fn, bounds, maxiter=100, tol=1e-7) 
    adv_x = np.reshape(result.x, x.shape)  
    return adv_x