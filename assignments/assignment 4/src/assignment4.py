# system tools
import os

# creating dataframes
import pandas as pd

# visualization
import matplotlib.pyplot as plt

# loading model
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch

# loading data
from PIL import ImageFile, Image

def load_model():
    """
    Loads the models needed for face detection

    Returns:
        mtcnn: face detector
    """
    mtcnn = MTCNN(keep_all=True) # Initialize MTCNN for face detection

    resnet = InceptionResnetV1(pretrained='casia-webface').eval() # Load pre-trained FaceNet model
    return mtcnn

def create_path():
    """
    Navigates into the input folder
    
    Returns:
        subfolders: list of the three subfolders in the dataset
        filepath: path to dataset
    """
    filepath = os.path.join("input", "newspapers")
    subfolders = os.listdir(filepath) # create a list of subfolders
    subfolders.remove('README-images.txt') # remove the README file from the list
    return subfolders, filepath

def face_detection(subfolders, filepath, mtcnn):
    """
    Detects how many faces there are on every page and creates a csv file with the data
    
    Arguments:
        subfolders: list of the three subfolders in the dataset
        filepath: path to dataset
        mtcnn: face detector
    """
    ImageFile.LOAD_TRUNCATED_IMAGES = True # load truncated images too. without this, an error will occur

    #load data
    for folder in subfolders:
        newspaper_info = [] # an empty file to collect the name of the file, the decade, and the amount of detected faces
        folderpath = os.path.join(filepath, folder)
        data = os.listdir(folderpath)
        for file in data:
            imagepath = os.path.join(folderpath, file)
            image = Image.open(imagepath) # load the image
            boxes, _ = mtcnn.detect(image) # detect faces
            face_count = 0 # reset the variable and keep it at 0 if boxes = None
            if boxes is not None:
                face_count = len(boxes) #counting the faces based on the length of the boxes variable
                face_presence = 1 #if any faces are present, set this variable as 1...
            else:
                face_presence = 0 # ... and if not, set it to 0. is used to count the total number of pages with faces on them
            decade = file[4:7] + "0" # getting the decade by finding the first 3 numbers of the year in the filename
            file_info = [file, decade, face_count, face_presence] #compile info for every page
            newspaper_info.append(file_info) # add to list with the info for the entire newspaper
        df = pd.DataFrame(newspaper_info, # creating a dataframe for every newspaper using pandas
                        columns=["Filename", "Decade", "Total number of faces", "Is a face present?"])
        # upload dataframe to output folder
        outpath = os.path.join("output", "data_unaltered", folder + ".csv")
        df.to_csv(outpath, index=False)

def group_by_decade():
    """
    Groups the data by decade, 
    calculates the total number of faces per decade, 
    calculates the percentage of pages with a face present. 
    Adds this to a csv file.
    """
    output = os.path.join("output", "data_unaltered") # navigate to the previously created csv files
    output = os.listdir(output) # create a list of the files

    for csv in output:
        newspaper = pd.read_csv("output/data_unaltered/" + csv, keep_default_na=False) # read csv files
        sorted = newspaper.sort_values('Filename', inplace=False) # sort the dataframe by decade
        all_decades = [] # list to save the amount of faces for every decade
        for decade in sorted["Decade"]:
            sliced = sorted[sorted['Decade'] == decade] #slice the dataframe by decades
            decade_sum = sum(sliced['Total number of faces']) # total number of face for that decade
            
            # calculate percentage of pages containing a face
            total_pages = len(sliced) # find the total amount of pages
            pages_with_faces = sum(sliced['Is a face present?']) # counts how many pages has one or more faces
            percentage = pages_with_faces/total_pages*100
            
            decade_info = [decade, decade_sum, percentage] # compile info for every decade
            all_decades.append(decade_info) # add to list with info for all decades
        
        decade_df = pd.DataFrame(all_decades, # creating a dataframe for every newspaper using pandas
                        columns=["Decade", "Total number of faces", "% of pages with faces"])
        decade_df.drop_duplicates(inplace=True) # get rid of duplicates
        
        # upload dataframe to output folder
        outpath = os.path.join("output", "decade_info", "decade_info_" + csv)
        decade_df.to_csv(outpath, index=False)

def plot():
    """
    Plots the percentage of pages with a face detected per decade.
    Repeats for all 3 newspapers
    """
    plot_input = os.path.join("output", "decade_info") # navigate into the folder with the data for every decade
    plot_input = os.listdir(plot_input) # create a list of the files
    for csv in plot_input:
        newspaper = pd.read_csv("output/decade_info/" + csv, keep_default_na=False) # read csv files
        newspaper_name = csv[12:15] # grab the name of the newspaper from the filenames
        
        # define axes
        x = newspaper['Decade']
        y = newspaper['% of pages with faces']

        # plot
        fig, ax = plt.subplots()
        ax.bar(x, y, width=5, edgecolor="white", linewidth=0.7)
        plt.xlabel('Decades')
        plt.ylabel('Percentage')
        plt.title(newspaper_name + "'s percentage of pages with faces")

        plt.savefig('output/' + newspaper_name + '.png') # save output
        plt.show()
        plt.clf()

def main():
    mtcnn = load_model()
    subfolders, filepath = create_path()
    face_detection(subfolders, filepath, mtcnn)
    group_by_decade()
    plot()

if __name__=="__main__":
    main()