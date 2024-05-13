# system tools
import os
import sys

# arguments
import argparse

# working with images
import cv2

# numpy
# import numpy as np

# util functions
from imutils import jimshow as show
from imutils import jimshow_channel as show_channel

# visualizations
import matplotlib.pyplot as plt

# making dataframes
import pandas as pd

def parser():
    """
    Creates arguments for the input flower

    Returns:
        arg: filename for the chosen flower
    """
    parser = argparse.ArgumentParser(description="Choose a file from the dataset")
    parser.add_argument("--input",
                        "-i",
                        required=True,
                        help="Filename for a flower. use full name of file + .jpg")
    arg = parser.parse_args()
    return arg

def create_path(arg):
    """
    Creates a filepath to the directory

    Arguments:
        arg: filename for the chosen flower
    Returns:
        image_flower: image of chosen flower
        filepath: path to chosen flower file
    """
    filepath = "input/flowers" + arg.input
    image_flower = cv2.imread(filepath)
    return image_flower, filepath

def histogram(image_flower):
    """
    Creates a normalized histogram for the chosen flower

    Arguments:
        image_flower: image of chosen flower
    Returns:
        normalized_image_flower: image of chosen flower, normalized
    """
    hist_flower = cv2.calcHist([image_flower], [0,1,2], None, [255,255,255], [0,256, 0,256, 0,256]) # make histogram
    normalized_hist_flower = cv2.normalize(hist_flower, hist_flower, 0, 1.0, cv2.NORM_MINMAX) # normalize
    return normalized_hist_flower

def calculate_distance(normalized_hist_flower):
    """
    Creates normalized histograms for every file in the dataset, 
    used to calculate distance from the input image,
    creates a dataframe of 5 most similar flowers

    Arguments:
        normalized_image_flower: image of chosen flower, normalized
    Returns:
        datapath_data: path to folder with all flowers
        df: dataframe with chosen flower + 5 most similar flowers
    """
    datapath_data = os.path.join("input", "flowers") # navigate into folder with all flowers
    filelist_data = sorted(os.listdir(datapath_data)) # create list of all flowers
    distance_scores = [] # make empty list for distance scores to be added to

    for files in filelist_data: # for loop going through every flower's file
        filepath_data = datapath_data + "/" + files # navigate to file
        image_data = cv2.imread(filepath_data) # read image
        hist_data = cv2.calcHist([image_data], [0,1,2], None, [255,255,255], [0,256, 0,256, 0,256]) # make histogram
        normalized_hist_data = cv2.normalize(hist_data, hist_data, 0, 1.0, cv2.NORM_MINMAX) # normalize
        distance = round(cv2.compareHist(normalized_hist_flower, normalized_hist_data, cv2.HISTCMP_CHISQR), 2) # calculate distance score

        # create list for every file
        file_info = [files, distance]
        # append the file's info to the collected list for the whole folder's info
        distance_scores.append(file_info)
    
    # make dataframe
    df = pd.DataFrame(distance_scores,
                        columns=["Filename", "Distance"])

    df = df.sort_values(by=['Distance'])[:6] # sort the dataframe by shortest distance, and show the first 5 results and the flower we selected in the beginning

    outpath = os.path.join("output", "flowercomparison.csv")
    df.to_csv(outpath, index=False) # upload dataframe to output folder
    return datapath_data, df

def show_flowers(df, datapath_data, filepath):
    """
    Shows the target flower and the 5 most similar flowers,
    saves the results

    Arguments:
        df: dataframe with chosen flower + 5 most similar flowers
        datapath_data: path to folder with all flowers
        filepath: path to chosen flower file
    """
    #creating a grid of the results
    fig = plt.figure(figsize=(10, 7)) 

    # setting the number of rows and columns
    rows = 2
    columns = 3

    counter = 2 # counter will be used to indicate where to place the image in the grid

    image_flower = cv2.imread(filepath)
    fig.add_subplot(rows, columns, 1) # add new plot to grid
    plt.imshow(image_flower[:,:,::-1]) # showing flower, switching from BGR to RGB
    plt.axis('off') # dont show axes
    plt.title("Chosen flower")

    for flower in df["Filename"][1:6]:
        similar_flower = datapath_data + "/" + flower
        image_similar_flower = cv2.imread(similar_flower)
        fig.add_subplot(rows, columns, counter) # add new plot to grid
        plt.imshow(image_similar_flower[:,:,::-1]) # showing flower, switching from BGR to RGB
        plt.axis('off') # dont show axes
        counter += 1

    plt.savefig('output/results.png') # save results

def main():
    arg = parser()
    image_flower, filepath = create_path(arg)
    normalized_hist_flower = histogram(image_flower)
    datapath_data, df = calculate_distance(normalized_hist_flower)
    show_flowers(df, datapath_data, filepath)

if __name__ =="__main__":
    main()