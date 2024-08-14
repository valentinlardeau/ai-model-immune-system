###################   Import    ##########################
import tensorflow as tf
import numpy as np
import os
import pickle
import VGG



##################   Function   ##########################

"""
brief  : realising the Fast Gradient Sign Method
input  : model   : model we want to attack
         x       : x of the data 
         y       : y of the data
         epsilon : value of the epsilon we want to work with 
return : return the imortant metrics as an array (accuracy, recall, f1, precisions)
"""
def fgsm_attack(model, x, y, epsilon): 
    print("fgsm attack")
    # converting in tensor
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    y = tf.convert_to_tensor(y, dtype=tf.float32)
    # realising the attack
    with tf.GradientTape() as tape:
        tape.watch(x)
        prediction = model(x, training=False)
        loss = tf.keras.losses.categorical_crossentropy(y, prediction)
    # changing the values
    gradient = tape.gradient(loss, x)
    signed_grad = tf.sign(gradient)
    adv_x = x + epsilon * signed_grad
    adv_x = tf.clip_by_value(adv_x, 0, 1)
    # returning the attacked x
    return adv_x



"""
brief  : realising the Projected Gradient Descent 
input  : model        : model we want to attack
         x            : x of the data 
         y            : y of the data
         epsilon      : value of the epsilon we want to work with 
         epsilon_iter : coefficient of epsilon changes
         iterations   : number of iterations
return : return the attacked x
"""
def pgd_attack(model, x, y, epsilon, epsilon_iter=0.01, iterations=10): 
    print("pgd attack")
    # converting into the right format
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    y = tf.convert_to_tensor(y, dtype=tf.float32)
    adv_x = tf.identity(x)
    # attacking the right number of time
    for _ in range(iterations):
        # changing the data
        with tf.GradientTape() as tape:
            tape.watch(adv_x)
            prediction = model(adv_x, training=False)
            loss = tf.keras.losses.categorical_crossentropy(y, prediction)
        # actualising the changes
        gradient = tape.gradient(loss, adv_x)
        signed_grad = tf.sign(gradient)
        adv_x = adv_x + epsilon_iter * signed_grad
        adv_x = tf.clip_by_value(adv_x, x - epsilon, x + epsilon)
        adv_x = tf.clip_by_value(adv_x, 0, 1)
    # return the attacked x
    return adv_x



"""
brief  : realising the deepfool attack 
input  : model    : model we want to attack
         x        : x of the data 
         y        : y of the data
         max_iter : number of iteration we will do in the worst case 
         epsilon  : coefficient of epsilon changes
return : return the attacked x
"""
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
        batch_indices = tf.range(logits.shape[0], dtype=tf.int64)
        logits_y = tf.gather_nd(logits, tf.stack((batch_indices, y), axis=1))
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
        # doing it for every classes
        for k in range(num_classes):
            if k == y:
                continue
            grad_k = gradients[:, k, ...]
            grad_y = gradients[:, y, ...]
            perturbation_k = (logits[:, k] - logits_y) / (tf.norm(grad_k - grad_y) + epsilon)
            perturbation = tf.where(perturbation_k < min_perturbation, grad_k - grad_y, perturbation)
            min_perturbation = tf.minimum(perturbation_k, min_perturbation)
        perturbation = tf.nn.l2_normalize(perturbation, axis=tf.range(1, len(input_shape)))
        perturbed_x = perturbed_x + perturbation * min_perturbation[..., tf.newaxis]
        with tf.GradientTape() as tape:
            tape.watch(perturbed_x)
            logits = model(perturbed_x)
            k_i = tf.argmax(logits, axis=1)
    # returning the attacked x
    return perturbed_x



"""
brief  : realising the Differential Evolution attack 
input  : model           : model we want to attack
         x               : x of the data 
         y               : y of the data
         max_iter        : The maximum number of iterations for the attack
         epsilon         : value of the epsilon we want to work with 
         population_size : The number of individuals in the population
         mutation_factor : The mutation factor for the differential evolution
         crossover_prob  : The crossover probability for the differential evolution
return : return the attacked x
"""
def differential_evolution_attack(model, x, y, max_iter=50, epsilon=1e-6, population_size=20, mutation_factor=0.5, crossover_prob=0.7):
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    # Convert one-hot encoded labels to integer labels if necessary
    if y.shape.rank > 0 and y.shape[-1] > 1:
        y = tf.argmax(y)
    y = tf.convert_to_tensor(y, dtype=tf.int64)
    # Ensure max_iter is an integer
    max_iter = int(max_iter)  
    # local function to calculate the fitness
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
    # doing the attack for a determined iteration number
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



"""
brief  : realising the attacks on a dataset and saving it 
input  : model        : model we want to attack
         x            : x of the data 
         y            : y of the data
         attack_fns   : dictionnary of the attack we will do on the dataset
         save_dir     : directory where to save the dataset 
         attack_param : dictionnary of function to use
return : return the attacked dataset
"""
def generate_and_save_adversarial_examples(model, x, y, attack_fns, save_dir='attacked_datasets', **attack_params):
    # creating the repository if it does not exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # copying the data of the dataset
    combined_attacked_x = np.copy(x)
    #realising the attacks
    for attack_name, attack_fn in attack_fns.items():
        print(f"Generating adversarial examples using {attack_name}")
        attacked_x = attack_fn(model, x, y, **attack_params.get(attack_name, {}))
        combined_attacked_x = np.concatenate((combined_attacked_x, attacked_x.numpy()), axis=0)
    # saving the dataset
    save_path = os.path.join(save_dir, "combined_attacked_dataset.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(combined_attacked_x, f)
    print(f"Saved combined attacked dataset to {save_path}")
    # retunring the dataset
    return combined_attacked_x






##########   Test   #########
# attack_fns = {
#     'FGSM': fgsm_attack,
#     'PGD': pgd_attack,
#     # 'DeepFool': deepfool_attack,
#     # 'DifferentialEvolution': differential_evolution_attack
# }
# attack_params = {
#     'FGSM': {'epsilon': 0.3},
#     'PGD': {'epsilon': 0.3, 'epsilon_iter': 0.01, 'iterations': 10},
#     # 'DeepFool': {'max_iter': 50, 'epsilon': 1e-6},
#     # 'DifferentialEvolution': {'max_iter': 50, 'epsilon': 1e-6}
# }

# x_train, x_test, y_train, y_test = VGG.data_MNIST()



# model = VGG.load_mnist_vgg16_model("mnist_vgg16_model.h5")

# attacked_datasets = generate_and_save_adversarial_examples(model, x_test, y_test, attack_fns, **attack_params)
# print (attacked_datasets)
