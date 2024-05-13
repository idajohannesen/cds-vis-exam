# Assignment 1

# Short description:
The code allows you to pick an image of a flower and compare it with all of the flower images in the dataset. Using this comparison, the 5 most similar images will be shown.
There are two scripts included in this task, the first of which is the ```assignment1.py``` file. This script uses histograms to calculate distance scores for every flower and provides the 5 flowers with the smallest distance scores. 
The second script, ```vgg16.py```, uses VGG16 to find the 5 most similar flowers. This is done through a k-nearest neighbour function.

The results can be found in the ```ouput``` folder as images. The distance scores for the 5 most similar flowers from the first script are also saved as a .csv file. 

# Data:
The dataset is from the Visual Geometry Group at the University of Oxford, and it consists of more than 1000 flower images across 17 different species. It should be inserted in the input folder as a folder named ```flowers```.
The data can be downloaded from here: https://www.robots.ox.ac.uk/~vgg/data/flowers/17/

# Reproducing:
A setup file has been included which creates a virtual environment with the necessary requirements, as well as a ```run.sh``` file which activates the virtual environment and runs the script. The script requires that the data is located in the input folder with the structure mentioned above. 

Start by running the setup file by inputting ```bash setup.sh``` in the terminal. 
The code can then be run through the terminal by inputting ```bash run.sh```. This runs both scripts. This file can be edited with your chosen flower's filename before running, replacing the file ```/image_0001.jpg``` which has been inserted as a default option. Use the full filename, including the filetype and a preceding forward slash.
Both of these lines should be executed from the ```assignment 1``` folder.

The ```src``` folder also includes a necessary utils script which does not have to be run manually.
The code has been written and tested on a Windows laptop and may not work correctly on other operating systems.

# Discussion/summary:
The distance between images is calculated based on histograms, which means that they're primarily based on the colours found in the image rather than edge detection, textures, or other distinctions between flowers. The output may therefore not be the same species of flower as the input or even resemble it. For example, the histograms may be similar due to similar colours in the background of the images, rather than the flowers themselves. This can be seen in the ```results_example.png``` file in the ```output``` folder, where the most similar flowers are all notably different from the target flower. The flowers are generally yellow or white, with dark green backgrounds, but the flowers themselves are not similar in terms of shape or type of flower.
It works as a simple way to compare images, but more detailed approaches, such as machine learning, would likely yield better results.

The script using VGG16 performs much better. The results are found in the ```results_vgg16_example.png``` file in the ```output``` folder. The target flower is a daffodil, and the model identified other examples of daffodils in the dataset to output as being the most similar. The model therefore appears to take type of flower and visual characteristics into account when similarities are calculated, instead of only using colour as an indicator of similarity between images.

In general, the VGG16 model performs much better than the histogram-based approach and can correctly identify flowers of the same type as the target flower. 