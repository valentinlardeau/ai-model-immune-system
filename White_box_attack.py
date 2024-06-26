# # ###############   import   ################################


# # import tensorflow as tf
# # import numpy as np
# # from scipy.optimize import differential_evolution

# # ##############   Function   #############################



# # def fgsm_attack(model, x, y, epsilon=0.1): # Fast Gradient Sign Method
# #     with tf.GradientTape() as tape:
# #         tape.watch(x)
# #         prediction = model(x)
# #         loss = tf.keras.losses.categorical_crossentropy(y, prediction)
# #     gradient = tape.gradient(loss, x)
# #     signed_grad = tf.sign(gradient)
# #     adv_x = x + epsilon * signed_grad
# #     adv_x = tf.clip_by_value(adv_x, 0, 1)  
# #     return adv_x



# # def pgd_attack(model, x, y, epsilon=0.1, epsilon_iter=0.01, iterations=10): #Projected Gradient Descent
# #     adv_x = tf.identity(x)  
# #     for _ in range(iterations):
# #         with tf.GradientTape() as tape:
# #             tape.watch(adv_x)
# #             prediction = model(adv_x)
# #             loss = tf.keras.losses.categorical_crossentropy(y, prediction)
# #         gradient = tape.gradient(loss, adv_x)
# #         signed_grad = tf.sign(gradient)
# #         adv_x = adv_x + epsilon_iter * signed_grad
# #         adv_x = tf.clip_by_value(adv_x, x - epsilon, x + epsilon)  
# #         adv_x = tf.clip_by_value(adv_x, 0, 1)  
# #     return adv_x


# # """
# # brief  : DeepFool attack implementation.
# # input  : model (keras.Model): Target model to be attacked.
# #          x (numpy.ndarray): Input image.
# #          y (numpy.ndarray): True label.
# #          max_iter (int): Maximum number of iterations.
# #          epsilon (float): Perturbation scale.   
# # return : numpy.ndarray: Adversarial example.
# # """
# # def deepfool_attack(model, x, y, max_iter=50, epsilon=1e-2):
# #     x_adv = tf.identity(x)  # Initialize adversarial example with the original image
# #     orig_out = model.predict(tf.expand_dims(x_adv, axis=0))  # Predict the original image  
# #     def compute_gradients(x, y):
# #         with tf.GradientTape() as tape:
# #             tape.watch(x)
# #             output = model(x)
# #             loss = tf.keras.losses.categorical_crossentropy(y, output)
# #         return tape.gradient(loss, x), output 
# #     for i in range(max_iter):
# #         gradients, output = compute_gradients(tf.expand_dims(x_adv, axis=0), y)   
# #         perturbation = np.inf
# #         adversarial_class = np.argmax(output[0])
# #         for j in range(model.output_shape[1]):
# #             if j != adversarial_class:
# #                 adversarial_gradient = gradients[0][:, adversarial_class] - gradients[0][:, j]
# #                 w = adversarial_gradient.numpy()
# #                 f = output[0][adversarial_class] - output[0][j]
# #                 perturbation_j = np.abs(f) / np.linalg.norm(w.flatten())
# #                 if perturbation_j < perturbation:
# #                     perturbation = perturbation_j
# #                     w_optimal = w
# #                     f_optimal = f
# #         perturbation = np.minimum(epsilon, perturbation)
# #         x_adv = tf.clip_by_value(x_adv + perturbation * w_optimal, 0.0, 1.0)  # Clip to ensure pixel values stay in [0, 1]
# #         # Check if the prediction changed
# #         if np.argmax(model.predict(tf.expand_dims(x_adv, axis=0))) != adversarial_class:
# #             break
# #     return x_adv



# # def differential_evolution_attack(model, x, y, epsilon=0.1):
# #     def loss_fn(x_adv):
# #         x_adv = np.reshape(x_adv, x.shape)
# #         prediction = model.predict(np.array([x_adv]))
# #         return tf.keras.losses.categorical_crossentropy(y, prediction).numpy()
# #     bounds = [(x - epsilon, x + epsilon) for x in np.reshape(x, (-1,))] 
# #     result = differential_evolution(loss_fn, bounds, maxiter=100, tol=1e-7) 
# #     adv_x = np.reshape(result.x, x.shape)  
# #     return adv_x

# import tensorflow as tf
# import numpy as np
# from scipy.optimize import differential_evolution

# def fgsm_attack(model, x, y, epsilon=0.1):  # Fast Gradient Sign Method
#     x = tf.convert_to_tensor(x, dtype=tf.float32)
#     y = tf.convert_to_tensor(y, dtype=tf.float32)
#     with tf.GradientTape() as tape:
#         tape.watch(x)
#         prediction = model(x)
#         loss = tf.keras.losses.categorical_crossentropy(y, prediction)
#     gradient = tape.gradient(loss, x)
#     signed_grad = tf.sign(gradient)
#     adv_x = x + epsilon * signed_grad
#     adv_x = tf.clip_by_value(adv_x, 0, 1)  
#     return adv_x

# def pgd_attack(model, x, y, epsilon=0.1, epsilon_iter=0.01, iterations=10):  # Projected Gradient Descent
#     x = tf.convert_to_tensor(x, dtype=tf.float32)
#     y = tf.convert_to_tensor(y, dtype=tf.float32)
#     adv_x = tf.identity(x)  
#     for _ in range(iterations):
#         with tf.GradientTape() as tape:
#             tape.watch(adv_x)
#             prediction = model(adv_x)
#             loss = tf.keras.losses.categorical_crossentropy(y, prediction)
#         gradient = tape.gradient(loss, adv_x)
#         signed_grad = tf.sign(gradient)
#         adv_x = adv_x + epsilon_iter * signed_grad
#         adv_x = tf.clip_by_value(adv_x, x - epsilon, x + epsilon)  
#         adv_x = tf.clip_by_value(adv_x, 0, 1)  
#     return adv_x

# def deepfool_attack(model, x, y, max_iter=50, epsilon=1e-2):
#     x = tf.convert_to_tensor(x, dtype=tf.float32)
#     y = tf.convert_to_tensor(y, dtype=tf.float32)
#     x_adv = tf.identity(x)

#     def compute_gradients(x_adv, y):
#         with tf.GradientTape() as tape:
#             tape.watch(x_adv)
#             output = model(x_adv, training=False)
#             loss = tf.keras.losses.categorical_crossentropy(y, output)
#         gradients = tape.gradient(loss, x_adv)
#         return gradients, output

#     # Ensure x_adv has the correct shape
#     # x_adv = tf.expand_dims(x_adv, axis=0)
#     # y = tf.expand_dims(y, axis=0)
#     print(np.shape(x_adv))

#     for _ in range(max_iter):
#         gradients, output = compute_gradients(x_adv, y)
#         perturbation = np.inf
#         adversarial_class = np.argmax(output[0])  # Remove batch dimension
#         for j in range(model.output_shape[1]):
#             if j != adversarial_class:
#                 adversarial_gradient = gradients[0][:, :, :, adversarial_class] - gradients[0][:, :, :, j]  # Remove batch dimension
#                 w = adversarial_gradient.numpy()
#                 f = output[0][adversarial_class] - output[0][j]  # Remove batch dimension
#                 perturbation_j = np.abs(f) / np.linalg.norm(w.flatten())
#                 if perturbation_j < perturbation:
#                     perturbation = perturbation_j
#                     w_optimal = w
#                     f_optimal = f
#         perturbation = np.minimum(epsilon, perturbation)
#         x_adv = tf.clip_by_value(x_adv + perturbation * tf.convert_to_tensor(w_optimal, dtype=tf.float32), 0.0, 1.0)
#         # Check if the prediction changed
#         if np.argmax(model(tf.convert_to_tensor(x_adv), training=False)) != adversarial_class:
#             break

#     return x_adv[0].numpy()



# def differential_evolution_attack(model, x, y, epsilon=0.1):
#     x = tf.convert_to_tensor(x, dtype=tf.float32)
#     y = tf.convert_to_tensor(y, dtype=tf.float32)
    
#     def loss_fn(x_adv):
#         x_adv = np.reshape(x_adv, x.shape)
#         prediction = model.predict(np.array([x_adv]))
#         return tf.keras.losses.categorical_crossentropy(y, prediction).numpy()
    
#     bounds = [(xi - epsilon, xi + epsilon) for xi in np.reshape(x, (-1,))] 
#     result = differential_evolution(loss_fn, bounds, maxiter=100, tol=1e-7) 
#     adv_x = np.reshape(result.x, x.shape)  
#     adv_x = tf.convert_to_tensor(adv_x, dtype=tf.float32)
#     return adv_x

import tensorflow as tf
import numpy as np
from scipy.optimize import differential_evolution
from functools import partial

def fgsm_attack(model, x, y, epsilon):  # Fast Gradient Sign Method
    print("fgsm attack")

    x = tf.convert_to_tensor(x, dtype=tf.float32)
    y = tf.convert_to_tensor(y, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(x)
        prediction = model(x, training=False)
        loss = tf.keras.losses.categorical_crossentropy(y, prediction)
    gradient = tape.gradient(loss, x)
    signed_grad = tf.sign(gradient)
    adv_x = x + epsilon * signed_grad
    adv_x = tf.clip_by_value(adv_x, 0, 1)
    return adv_x

def pgd_attack(model, x, y, epsilon, epsilon_iter=0.01, iterations=10):  # Projected Gradient Descent
    print("pgd attack")
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    y = tf.convert_to_tensor(y, dtype=tf.float32)
    adv_x = tf.identity(x)
    for _ in range(iterations):
        with tf.GradientTape() as tape:
            tape.watch(adv_x)
            prediction = model(adv_x, training=False)
            loss = tf.keras.losses.categorical_crossentropy(y, prediction)
        gradient = tape.gradient(loss, adv_x)
        signed_grad = tf.sign(gradient)
        adv_x = adv_x + epsilon_iter * signed_grad
        adv_x = tf.clip_by_value(adv_x, x - epsilon, x + epsilon)
        adv_x = tf.clip_by_value(adv_x, 0, 1)
    return adv_x

def deepfool_attack(model, x, y, max_iter, epsilon):
    print("deepfool attack")
    x = tf.convert_to_tensor(x, dtype=tf.float32)

    # Convert one-hot encoded labels to integer labels if necessary
    if y.dtype == tf.float32:
        y = tf.argmax(y, axis=1)
    y = tf.convert_to_tensor(y, dtype=tf.int64)

    input_shape = x.shape
    perturbed_x = tf.identity(x)
    
    # Forward pass to get initial predictions
    with tf.GradientTape() as tape:
        tape.watch(perturbed_x)
        logits = model(perturbed_x)
        logits_y = tf.gather_nd(logits, tf.stack((tf.cast(tf.range(logits.shape[0]), tf.int64), y), axis=1))

    k_i = tf.argmax(logits, axis=1)
    num_classes = logits.shape[1]

    # Iterate to find minimal perturbation
    for i in range(max_iter):
        if tf.reduce_all(k_i != y):
            break
        
        with tf.GradientTape() as tape:
            tape.watch(perturbed_x)
            logits = model(perturbed_x)
        
        gradients = tape.gradient(logits, perturbed_x)
        
        perturbation = tf.zeros_like(x)
        min_perturbation = tf.fill(input_shape[:1], np.inf)

        for k in range(num_classes):
            if k == y:
                continue
            
            grad_k = gradients[:, k, ...]
            grad_y = gradients[:, y[0].numpy(), ...]  # Extracting the scalar for indexing
            perturbation_k = (logits[:, k] - logits_y) / (tf.norm(grad_k - grad_y) + epsilon)
            perturbation = tf.where(perturbation_k < min_perturbation, grad_k - grad_y, perturbation)
            min_perturbation = tf.minimum(perturbation_k, min_perturbation)
        
        perturbation = tf.nn.l2_normalize(perturbation, axis=tf.range(1, len(input_shape)))
        perturbed_x = perturbed_x + perturbation * min_perturbation[..., tf.newaxis]

        with tf.GradientTape() as tape:
            tape.watch(perturbed_x)
            logits = model(perturbed_x)
            k_i = tf.argmax(logits, axis=1)
    
    return perturbed_x


def differential_evolution_attack(model, x, y, max_iter=50, epsilon=1e-6, population_size=20, mutation_factor=0.5, crossover_prob=0.7):
    """
    Performs a Differential Evolution attack on a given input.
    
    Parameters:
    - model: The neural network model (must be a subclass of tf.keras.Model).
    - x: The input tensor.
    - y: The true labels tensor (assumed to be one-hot encoded).
    - max_iter: The maximum number of iterations for the attack.
    - epsilon: A small value to avoid division by zero.
    - population_size: The number of individuals in the population.
    - mutation_factor: The mutation factor for the differential evolution.
    - crossover_prob: The crossover probability for the differential evolution.
    
    Returns:
    - Perturbed input.
    """
    print ("Differential evolution attack")
    x = tf.convert_to_tensor(x, dtype=tf.float32)

    # Convert one-hot encoded labels to integer labels if necessary
    if y.shape.rank > 0 and y.shape[-1] > 1:
        y = tf.argmax(y)
    y = tf.convert_to_tensor(y, dtype=tf.int64)

    max_iter = int(max_iter)  # Ensure max_iter is an integer
    
    def compute_fitness(perturbed_x):
        logits = model(perturbed_x)
        predictions = tf.argmax(logits, axis=1)
        correct = tf.equal(predictions, y)
        return tf.reduce_sum(tf.cast(correct, tf.float32))
    
    # Initialize the population
    input_shape = x.shape
    population = x + epsilon * tf.random.uniform((population_size,) + input_shape, -1, 1)
    best_individual = x
    best_fitness = compute_fitness(tf.expand_dims(x, 0))
    
    for iteration in range(max_iter):
        for i in range(population_size):
            # Mutation
            idxs = [idx for idx in range(population_size) if idx != i]
            a, b, c = np.random.choice(idxs, 3, replace=False)
            mutant = population[a] + mutation_factor * (population[b] - population[c])
            mutant = tf.clip_by_value(mutant, 0, 1)

            # Crossover
            cross_points = tf.random.uniform(input_shape) < crossover_prob
            if not tf.reduce_any(cross_points):
                cross_points = tf.cast(tf.one_hot(np.random.randint(0, input_shape[-1]), input_shape[-1]), tf.bool)
            trial = tf.where(cross_points, mutant, population[i])

            # Selection
            trial_fitness = compute_fitness(tf.expand_dims(trial, 0))
            if trial_fitness > compute_fitness(tf.expand_dims(population[i], 0)):
                population[i] = trial
                if trial_fitness > best_fitness:
                    best_individual = trial
                    best_fitness = trial_fitness
        
        # Early stopping if any adversarial example is found
        if best_fitness == 0:
            break

    return best_individual
