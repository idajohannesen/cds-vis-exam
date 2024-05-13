# activate the environment
source ./env_assignment/bin/activate
# run the code
python src/assignment1.py -i /image_0001.jpg
python src/vgg16.py -i /image_0001.jpg
# close the environment
deactivate