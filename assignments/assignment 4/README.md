# Assignment 4

# Short description:
The code performs face detection on a dataset covering 3 different newspapers throughout the decades of their publishing. The number of faces per page are counted and the final results are grouped together by decade. The ```output``` folder contains plots for how large a percentage of pages contain faces for all the available decades a newspaper was published in. In the subfolder ```data_unaltered``` there is a csv file for the data for each newspaper, including the number of faces per page and a binary setting of whether or not a page contains a face. In the other subfolder, called ```decade_info```, there is a csv file for every newspaper containing the face detection data grouped into decades.

# Data:
The models were trained on a corpus of Swiss newspapers. It includes three newspapers: the Journal de Genève (JDG,operating from 1826 to 1994), the Gazette de Lausanne (GDL, 1804-1991), and the Impartial (IMP, 1881-2017). The dataset should be placed in the ```input``` folder.
Download and find more information about the dataset here: https://zenodo.org/records/3706863

# Reproducing:
A setup file has been included which creates a virtual environment with the necessary requirements, as well as a ```run.sh``` file which activates the virtual environment and runs both scripts.

Start by running the setup file by inputting ```bash setup.sh``` in the terminal. 
The code can then be run through the terminal by inputting ```bash run.sh```. This runs both scripts.
Both of these lines should be executed from the ```assignment 4``` folder.

The code has been written and tested on a Windows laptop and may not work correctly on other operating systems.

# Discussion/summary:
Generally, the results show a rising trend, as the highest percentage of pages with faces on them happen in the most recent publications for all three newspapers. JDG falls slightly from the 80s to the 90s but both are still the two highest percentages for that newspaper. This overall trend is likely due to improving technology for printing and taking photos, which made it easier and cheaper to add visuals instead of, for example, illustrating them.
There are, however, certain limitation to this approach. As with any quantitative data analysis, it is impossible to doublecheck the data. In this case, we do not have the time to check if every face has been counted, as the dataset does include some faces that might not be picked up by the detector, such as illustrations of faces, faces in poor lighting, or faces wearing helmets or other face-obscuring accessories. The image of the individual page may also be too low quality for any faces to be detected.
It is also possible for the model to detect faces where there are none. This becomes clear when looking at GDL's graph, which shows that 25% of all pages published from 1800-1809 contain a face. However, under further inspection, none of these pages contain any faces and the model is likely detecting faces in the text. Therefore, we should also disregard the high percentages in the start of GDL's graph, as they are false positives.