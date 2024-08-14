# ###############   import   ################################

import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten
from keras.src.utils import to_categorical
from PIL import Image
from tensorflow.python.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score
from keras.src.applications.vgg16 import VGG16
from keras.src.datasets import mnist
import tensorflow as tf


##############   Function   #############################

"""
brief  : changing the size of the picture to be able to use them with VGG16
input  : images : array of image we want to resize
return : array of image with the right shape
"""
def resize_images(images):
    resized_images = np.zeros((images.shape[0], 32, 32))
    # resizing all images 
    for i, image in enumerate(images):
        img = Image.fromarray(image)
        img = img.resize((32, 32))
        resized_images[i] = np.array(img)
    return resized_images


"""
brief  : retrieving the MNIST data
return : return the train and test data for x and y
"""
def data_MNIST():
    print("getting data")
    # getting the data
    (x_train, y_train), (x_test, y_test) = mnist.load_data() 
    x_train, y_train = x_train[:1000], y_train[:1000]
    x_test, y_test = x_test[:200], y_test[:200]
    x_train = resize_images(x_train)
    x_test = resize_images(x_test)
    # data normalization
    x_train = x_train.astype('float32') / 255.0 
    x_test = x_test.astype('float32') / 255.0
    # converting to one-hot
    y_train = to_categorical(y_train, 10) 
    y_test = to_categorical(y_test, 10)
    # proper data for VGG-16
    x_train = np.stack((x_train,) * 3, axis=-1) 
    x_test = np.stack((x_test,) * 3, axis=-1)
    return x_train, x_test, y_train, y_test


"""
brief  : creating the VGG architecture
input  : x_train : data for training of x
         x_test : data for the test part for x
         y_train : data for training for y
         y_test : data for testing for y
"""
def createModel(x_train, x_test, y_train, y_test):
    # getting the pretrained model
    vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3)) 
    # keeping the right architecture
    for layer in vgg_model.layers: 
        layer.trainable = False
    # creating our model
    model = Sequential() 
    # adding VGG in the model
    model.add(vgg_model) 
    # adding end of network for classes
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    # compiling the model
    model.compile(loss='categorical_crossentropy',
                optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-4),
                metrics=['accuracy']) 
    # training
    model.fit(x_train, y_train,
            batch_size=256,
            epochs=1,
            validation_data=(x_test, y_test)) 
    # saving the model
    model.save("mnist_vgg16_model.keras")
    print("Model saved successfully.")


"""
brief  : loading the VGG16 model
input  : model_path : path to the model to retrieve
return : return the model or none in case of error
"""
def load_mnist_vgg16_model(model_path):
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        print("error")
        return None


"""
brief  : calculating all the needed results
input  : x_test : data for the test part for x
         y_test : data for testing for y
         model : model we want to use 
return : return the imortant metrics as an array (accuracy, recall, f1, precisions)
"""
def result(x_test, y_test, model):
    # Convert the data to TensorFlow tensors
    x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
    y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)
    # Predictions
    y_pred = model(x_test, training=False)
    y_pred_classes = np.argmax(y_pred.numpy(), axis=1)
    y_true = np.argmax(y_test.numpy(), axis=1)
    # Confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred_classes)
    print("Confusion Matrix:")
    print(conf_matrix)
    # Classification report
    class_report = classification_report(y_true, y_pred_classes)
    print("\nClassification Report:")
    print(class_report)
    # Calculating metrics
    accuracy = accuracy_score(y_true, y_pred_classes)
    recall = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)
    f1_scores = 2 * (accuracy * recall) / (accuracy + recall)
    precision_macro = precision_score(y_true, y_pred_classes, average='macro')
    precision_micro = precision_score(y_true, y_pred_classes, average='micro')
    # return requiered
    return [accuracy, recall.mean(), f1_scores.mean(), precision_macro, precision_micro]



