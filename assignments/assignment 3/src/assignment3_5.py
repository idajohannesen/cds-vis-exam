# system tools
import os

# tf tools
import tensorflow as tf

# image processsing
from tensorflow.keras.preprocessing.image import (load_img,
                                                  img_to_array,
                                                  ImageDataGenerator)
# VGG16 model
from tensorflow.keras.applications.vgg16 import (preprocess_input,
                                                 decode_predictions,
                                                 VGG16)
# layers
from tensorflow.keras.layers import (Flatten, 
                                     Dense, 
                                     Dropout, 
                                     BatchNormalization)

# model object
from tensorflow.keras.models import Model

# optimizers
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import SGD, Adam

# scikit-learn
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# visualizations
import numpy as np
import matplotlib.pyplot as plt

def create_dir():
    """
    Navigates into the input folder

    Returns:
        subfolders: list of the ten subfolders in the dataset
        filepath: path to dataset
    """
    filepath = os.path.join("input", "Tobacco3482")
    subfolders = os.listdir(filepath) # create a list of subfolders
    return subfolders, filepath

def load_data(subfolders, filepath):
    """
    Loads the data from the input folder's subfolders

    Arguments:
        subfolders: list of the ten subfolders in the dataset
        filepath: path to dataset
    Returns:
        img: list of all images in the dataset
        labels: list with labels for every image
    """
    img = [] # a list for collecting all images
    labels = [] # a list for the 10 labels
    label = 0 # start a label counter at 0

    #load data
    for folder in subfolders: # navigate into the input folder
        folderpath = os.path.join(filepath, folder)
        data = os.listdir(folderpath)
        for file in data:
            if file.endswith(".jpg"): # only load images
                imagepath = os.path.join(folderpath, file)
                image = load_img(imagepath, target_size=(224, 224)) # loading the data in the target size that works for the VGG16 model
                img.append(image) # add to the list for all images
                labels.append(label) # add a label to the label list
        label += 1
    return img, labels

def preprocess(img):
    """
    Preprocesses the images for the VGG16 model

    Arguments:
        img: list of all images in the dataset
    Returns:
        preprocessed_img: list of preprocessed images
    """
    img_array = np.array(img) # converting the list to an array
    preprocessed_img = preprocess_input(img_array) # preprocessing
    return preprocessed_img

def split(preprocessed_img, labels):
    """
    Creates a train-test split

    Arguments:
        preprocessed_img: list of preprocessed images
        labels: list with labels for every image
    Returns:
        X_train: images for training
        X_test: imeages for testing
        y_train: labels for training data
        y_test: labels for testing data
    """
    (X_train, X_test, y_train, y_test) = train_test_split(preprocessed_img,
                                                      labels, 
                                                      test_size=0.1) # creating a 90/10 split
    X_train = X_train.astype("float") / 255.
    X_test = X_test.astype("float") / 255.
    
    lb = LabelBinarizer() #binarize labels to create one-hot vectors
    y_train = lb.fit_transform(y_train)
    y_test = lb.fit_transform(y_test)
    return X_train, X_test, y_train, y_test

def labeling(): # defining our 10 labels
    """
    Defines the 10 labels

    Returns:
        labelnames: names of the 10 labels
    """
    labelnames = ['advertisement', 'email', 
            'form', 'letter', 
            'memo', 'news', 
            'note', 'report', 
            'resume', 'scientific paper']
    return labelnames

def load_model():
    """
    Loads the VGG16 model without classifier layers

    Returns:
        model: VGG16 model
    """
    model = VGG16(include_top=False, 
              pooling='max', # max pooling
              input_shape=(224, 224, 3))
    
    # mark loaded layers as not trainable
    for layer in model.layers:
        layer.trainable = False
    
    # adding new classifier layers to the model
    flat1 = Flatten()(model.layers[-1].output) #flatten the output of the model's last layer
    bn = BatchNormalization()(flat1) # batch normalization
    class1 = Dense(128, activation='relu')(bn) # adding one relu layer
    output = Dense(10, activation='softmax')(class1) # using a softmax layer for the output layer

    # update model
    model = Model(inputs=model.inputs, 
                  outputs=output)

    # optimization algorithm
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.01,
        decay_steps=10000,
        decay_rate=0.9)
    sgd = SGD(learning_rate=lr_schedule)

    #compiling the model
    model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    return model

def data_augmentation(X_train):
    """
    Expands the dataset using data augmentation

    Arguments:
        X_train: images for training
    Returns:
        datagen: image data generator for data augmentation
        X_train: images for training, augmented
    """
    datagen = ImageDataGenerator(horizontal_flip=True, # mirrors the images
                                 rotation_range=20, #rotates
                                 width_shift_range=0.2, # shift width
                                 height_shift_range=0.2, # shift height
                                 validation_split=0.1)
    datagen.fit(X_train) # fitting the data generator to the dataset
    return X_train, datagen

def train_model(model, X_train, y_train, datagen):
    """
    Trains the model

    Arguments:
        model: VGG16 model
        X_train: images for training, augmented
        y_train: labels for training data
        datagen: image data generator for data augmentation
    Returns:
        H: trained model
    """
    H = model.fit(X_train, y_train, 
            validation_split=0.1, # split off 10% of the data for validation
            batch_size=64,
            epochs=40, # running for 40 epochs
            verbose=1) # give updates on progress
    return H

def plot_history(H):
    """
    Plots the loss curves and accuracy curves

    Arguments:
        H: trained model
    Returns:
    """
    epochs = 40
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss", linestyle=":")
    plt.title("Loss curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.legend()
    #plt.clf()

    plt.subplot(1,2,2)
    plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc", linestyle=":")
    plt.title("Accuracy curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend()
    plt.show()

    plt.savefig('output/plot_6.png') # save output
    plt.show()

def classification(model, X_test, y_test, labelnames):
    """
    Creates a classification report

    Arguments:
        model: VGG16 model
        X_test: imeages for testing
        y_test: labels for testing data
        labelnames: names of the 10 labels
    """
    predictions = model.predict(X_test, batch_size=128)
    classifier_metrics = classification_report(y_test.argmax(axis=1),
                                predictions.argmax(axis=1),
                                target_names=labelnames)
    print(classifier_metrics)

    # saving classification report as a .txt file
    text_file = open(r'output/classification_report_6.txt', 'w')
    text_file.write(classifier_metrics)
    text_file.close()

def main():
    subfolders, filepath = create_dir()
    img, labels = load_data(subfolders, filepath)
    preprocessed_img = preprocess(img)
    X_train, X_test, y_train, y_test = split(preprocessed_img, labels)
    labelnames = labeling()
    model = load_model()
    X_train, datagen = data_augmentation(X_train)
    H = train_model(model, X_train, y_train, datagen)
    plot_history(H)
    classification(model, X_test, y_test, labelnames)

if __name__=="__main__":
    main()