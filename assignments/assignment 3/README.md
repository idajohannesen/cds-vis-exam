# Assignment 3

# Short description:
This script is meant to classify document types based on what they look like rather than the text they contain. To do this, the convolutional neural networks model called VGG16 has been used for the classification. The script uses functions imported from tensorflow to carry out this classification task.

# Data:
This code uses the ```Tobacco3482``` dataset, which consists of 3842 images spread across 10 different types of documents. It should be inserted into the ```input``` folder as a folder named ```Tobacco3482```, containing all ten document types as subfolders with the images inside.
https://www.kaggle.com/datasets/patrickaudriaz/tobacco3482jpg?resource=download

# Reproducing: 
A setup file has been included which creates a virtual environment with the necessary requirements, as well as a ```run.sh``` file which activates the virtual environment and runs the script. The script requires that the data is located in the ```input``` folder with the structure mentioned above. 

Start by running the setup file by inputting ```bash setup.sh``` in the terminal. 
The code can then be run through the terminal by inputting ```bash run.sh```.
Both of these lines should be executed from the ```assignment 3``` folder.

The code has been written and tested on a Windows laptop and may not work correctly on other operating systems.

# Discussion/summary:
Multiple classification reports have been saved in the ```output``` folder as ```classification_report_example.txt```. The model achieves an f1 accuracy score of 76%. Some notable parameters include a single relu layer, a batch size of 64, and 40 epochs. The loss curves show that the model plateaus earlier for validation accuracy than training accuracy which has yet to plateau after 40 epochs. The model is therefore also overfitting to its training data. During testing, adding more than 40 epochs did not significantly improve the validation accuracy and only led to more overfitting. In order to lessen overfitting, data augmentation has been used to expand the dataset. The data has been rotated, flipped, and shifted both horizontally and vertically as part of the augmentation. Throughout testing and adjusting the parameters, smaller batch sizes, more epochs, and adding data augmentation have generally led to improved results for the model, with 76% being the best final f1 score despite the overfitting.
Notably, the model consistently does worse with classifying emails and scientific papers, and to a lesser degree, with memos and forms. This could be because those four document types vary a lot or look very similar to another type. 
For example, the scientific papers vary a lot in their presentation. Some files are sorted into columns while others are not. Some contain a head for the name of the writer and other relevant information while others are mostly a block of text. The inconsistent visuals may be causing lower scores for these four types of documents, regardless of the model's parameters and general performance.