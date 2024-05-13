# Assignment 2

# Short description:
The code trains two classifiers on the Cifar10 dataset. The two scripts contains a logistic regression classifier and a neural network classifier, respectively. The data has been preprocessed through grayscaling, normalizing, and reshaping before it was used for training. The 'output' folder contains a classification report for both models as well as a loss curve graph for the neural network.

# Data:
The models were trained on the Cifar10 dataset which contains 50000 training images and 10000 test images. It is loaded as part of the scripts and does not need to be downloaded.
More information about the dataset here: https://www.cs.toronto.edu/~kriz/cifar.html

# Reproducing:
A setup file has been included which creates a virtual environment with the necessary requirements, as well as a ```run.sh``` file which activates the virtual environment and runs both scripts.

Start by running the setup file by inputting ```bash setup.sh``` in the terminal. 
The code can then be run through the terminal by inputting ```bash run.sh```. This runs both scripts.
Both of these lines should be executed from the ```assignment 2``` folder.

The ```src``` folder also includes two necessary utils scripts which do not have to be run manually.
The code has been written and tested on a Windows laptop and may not work correctly on other operating systems.

# Discussion/summary:
Out of the two options, the neural network obtains a higher f1 score. With 30 epochs and a logistic regression activation layer it reached 38%, while the logistic regression classifier reaches 31%. These are still low scores, and there is likely room for improvement if the neural network classifier is further adjusted. The ```output``` folder also contains classification reports for a neural network with either a ReLU or a tanh activation layer, both of which had lower accuracy scores than the logistic layer.
Additionally, the logistic regression's parameters have also been tested using a gridsearch. This was used to test if different tolerances or adding a l1 or l2 penalty would improve performance. The penalties did not improve performance, and a tolerance of 0.1 obtained the best results for f1 accuracy.
The neural network's performance can also be checked by looking at the loss curves, located in the ```output``` folder. Loss curves for neural networks should ideally have a smooth, steep downwards slope which ends in a plateau. This curve does not quite plateau, which could indicate that more iterations would improve the performance.