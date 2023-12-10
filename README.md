# Ear recognition using deep learning

## Setup project environment

Ensure you have Python 3.8 or later installed on your Linux machine. You can download it from the official Python website https://www.python.org/downloads/.
To check if Python is installed, use in terminal: `python --version` or `python3 --version`.

It's recommended to create a virtual environment to install the project's dependencies. Here's how you can do it:
1. Install the virtual environment package: `pip install virtualenv --user`
2. In terminal navigate to your project directory and create a virtual environment with: `virtualenv env`
3. Activate the virtual environment: `source env/bin/activate` (You should see (env) prefixed to your terminal indicating that the virtual environment is now active.)

Install the project's dependencies with: `pip install opencv-python numpy tqdm scikit-learn` 

## Database selection and preparation:
Ear images are downloaded from https://webctim.ulpgc.es/research_works/ami_ear_database/ and stored in the `data/01_raw` folder.

In terminal go to `src/data` directory and run `python3 prepare_data.py` to prepare the data for model. The script will:
- resize images to 128 pixels in width and store it in `data/02_resized` directory,
- resized images will convert to grayscale and stored in `data/03_grayscale` directory,
- normalize to have values between 0 and 1 and saved as numpy arrays in `data/04_normalized` directory,
- take grayscale images and create arrays of images and labels split for model to train, test and validate.
