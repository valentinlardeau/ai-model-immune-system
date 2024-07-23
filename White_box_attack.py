
import tensorflow as tf
import numpy as np
# from scipy.optimize import differential_evolution
# from functools import partial


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
