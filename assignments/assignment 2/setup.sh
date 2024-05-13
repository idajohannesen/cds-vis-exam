# create virtual env
python -m venv env_assignment
# activate env
source ./env_assignment/bin/activate
# install opencv
sudo apt-get update
sudo apt-get install -y python3-opencv
# install requirements
pip install --upgrade pip
pip install -r requirements.txt