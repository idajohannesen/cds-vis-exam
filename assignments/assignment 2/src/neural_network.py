# system tools
import os

# for preprocessing
import cv2

# making arrays
import numpy as np

# visualizations
import matplotlib.pyplot as plt

# util functions
from imutils import jimshow as show
from imutils import jimshow_channel as show_channel
import classifier_utils as clf_util

# scikit-learn
from sklearn import metrics
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# cifar10 dataset
from tensorflow.keras.datasets import cifar10

def load_data():
    """
    Loads the cifar10 dataset

    Returns:
        X_train: images for training
        X_test: images for testing
        y_train: labels for training data
        y_test: labels for testing data
    """
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    return X_train, y_train, X_test, y_test
    
def preprocess(X_train, X_test):
    """
    Preprocesses the data

    Arguments:
        X_train: images for training
        X_test: images for testing
    Returns:
        X_preprocess_train: images for training, preprocessed
        X_preprocess_test: images for testing, preprocessed
    """
    X_train_list=[]
    X_test_list=[]

    for image in X_train: # preprocessing the training images
        gray_train = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # convert to grayscale
        normalized = gray_train/255 # normalize
        X_train_list.append(normalized) # add to list
    X_preprocess_train_array = np.array(X_train_list) # turn list back into an array
    X_preprocess_train = X_preprocess_train_array.reshape(-1, 1024) # reshape

    for image in X_test: # preprocessing the testing images
        gray_test = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # convert to grayscale
        normalized = gray_test/255 # normalize
        X_test_list.append(normalized) # add to list
    X_preprocess_test_array = np.array(X_test_list) # turn list back into an array
    X_preprocess_test = X_preprocess_test_array.reshape(-1, 1024) # reshape
    return X_preprocess_train, X_preprocess_test

def mlp_classifier(X_preprocess_train, y_train):
    """
    Creates a neural network classifier

    Arguments:
        X_preprocess_train: images for training, preprocessed
        y_train: labels for training data
    Returns:
        classifier: neural network classifier
        """
    classifier = MLPClassifier(activation = "logistic",
                               hidden_layer_sizes = (30,), # 30 neurons in our hidden layer
                               max_iter=1000, # 1000 epochs
                               random_state = 9) # random state for reproducibility
    classifier.fit(X_preprocess_train, y_train) # fit the classifier to the data
    return classifier
    
def labels():
    """
    Defines labels for the 10 classes

    Returns:
        classes: labels of the 10 classes
    """
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    return classes

def predictions(classifier, X_preprocess_test):
    """
    Makes predictions

    Arguments:
        X_preprocess_test: images for testing, preprocessed
    Returns:
        y_pred: predictions
    """
    y_pred = classifier.predict(X_preprocess_test)
    return y_pred

def classification_report(y_pred, y_test, classes):
    """
    Gets classification report

    Arguments:
        y_pred: predictions
        y_test: labels for testing data
        classes: labels of the 10 classes
    """
    cm = metrics.classification_report(y_test, y_pred, target_names = classes)
    print(cm)

    # saving classification report as a .txt file
    text_file = open(r'output/MLP_classification_report.txt', 'w')
    text_file.write(cm)
    text_file.close()

def loss_curves(classifier):
    """
    Plots loss curves

    Arguments:
        classifier: neural network classifier
    """
    plt.plot(classifier.loss_curve_)
    plt.title("Loss curve during training", fontsize=14)
    plt.xlabel('Iterations')
    plt.ylabel('Loss score')
    # save output
    plt.savefig('output/loss_curve.png')
    plt.show()

def main():
    X_train, y_train, X_test, y_test = load_data()
    X_preprocess_train, X_preprocess_test = preprocess(X_train, X_test)
    classifier = mlp_classifier(X_preprocess_train, y_train)
    classes = labels()
    y_pred = predictions(classifier, X_preprocess_test)
    classification_report(y_pred, y_test, classes)
    loss_curves(classifier)

if __name__ =="__main__":
    main()