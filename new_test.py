import tensorflow as tf

# Check if TensorFlow is built with GPU support
print("Is TensorFlow built with GPU support:", tf.test.is_built_with_gpu_support())

# List available GPUs
print("Available GPU devices:", tf.config.experimental.list_physical_devices('GPU'))

# Create a simple computation graph
a = tf.constant(2.0)
b = tf.constant(3.0)
c = a * b

print("Computation result:", c)