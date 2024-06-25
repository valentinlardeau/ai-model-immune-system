###############   import   ################################

import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten
from keras import optimizers
from tensorflow.python.keras.utils import to_categorical
from PIL import Image
from tensorflow.python.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score
from keras.src.applications.vgg16 import VGG16
from keras.src.datasets import mnist


##############   Function   #############################

"""
brief  : changing the size of the picture to be able to use them with VGG16
input  : images : aray of image we want to resize
return : array of image with the right shape
"""
def resize_images(images):
    resized_images = np.zeros((images.shape[0], 32, 32))
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
    (x_train, y_train), (x_test, y_test) = mnist.load_data() # getting the data
    x_train = resize_images(x_train)
    x_test = resize_images(x_test)
    x_train = x_train.astype('float32') / 255.0 # data normalisation
    x_test = x_test.astype('float32') / 255.0
    y_train = to_categorical(y_train, 10) # converting in one-hot
    y_test = to_categorical(y_test, 10)
    x_train = np.stack((x_train,) * 3, axis=-1) # proper data for VGG-16
    x_test = np.stack((x_test,) * 3, axis=-1)
    return x_train, x_test, y_train, y_test



"""
brief  : retrieving the VGG architecture
input  : x_train : data for training of x
         x_test : data for the test part for x
         y_train : data for training for y
         y_test : data for testing for y
"""
def createModel(x_train, x_test, y_train, y_test):
    vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3)) # getting the pretrained model
    for layer in vgg_model.layers: # keeping the right architecture
        layer.trainable = False
    model = Sequential() # creating our model
    model.add(vgg_model) # adding VGG in the model
    # adding end of network for classes
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                optimizer=optimizers.RMSprop(lr=1e-4),
                metrics=['accuracy']) # compiling the model
    model.fit(x_train, y_train,
            batch_size=128,
            epochs=5,
            validation_data=(x_test, y_test)) # training
    model.save("mnist_vgg16_model.h5")
    print("Modèle sauvegardé avec succès.")



"""
brief  : retrieving the VGG16 model
input  : model_path : path to the model to retrieve
return : return the model
"""
def load_mnist_vgg16_model(model_path):
    model = load_model(model_path)
    return model



"""
brief  : calculating all the needed results
input  : x_test : data for the test part for x
         y_test : data for testing for y
         model : model we want to use 
"""
def result(x_test, y_test, model):
    # predictions
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)
    # confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred_classes)
    print("Matrix:")
    print(conf_matrix)
    # classification report
    class_report = classification_report(y_true, y_pred_classes)
    print("\classification report:")
    print(class_report)
    # calcul of valuable intels
    accuracy = accuracy_score(y_true, y_pred_classes)
    recall = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    f1_score = 2 * (accuracy * recall) / (accuracy + recall)
    precision_macro = precision_score(y_true, y_pred_classes, average='macro')
    precision_micro = precision_score(y_true, y_pred_classes, average='micro')
    # print of results
    print("\naccuracy by classe:")
    print(accuracy)
    print("\nRecall by classe:")
    print(recall)
    print("\nF1-score by classe:")
    print(f1_score)
    print("\nprecision (macro) by classe:")
    print(precision_macro)
    print("\nprecision (micro) by classe:")
    print(precision_micro)


##############   Test part   ########################



# x_train, x_test, y_train, y_test = data_MNIST()
# model = load_mnist_vgg16_model("mnist_vgg16_model.h5")
# # Évaluer le modèle sur les données de test
# score = model.evaluate(x_test, y_test, verbose=0)
# result(x_test,y_test,model)

