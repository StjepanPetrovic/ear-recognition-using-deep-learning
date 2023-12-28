# Ear recognition using deep learning

## Setup project environment (Linux)

Ensure you have Python 3.8 and pip python package manager or later installed on your Linux machine. You can download it from the official Python website https://www.python.org/downloads/.
To check if Python is installed, use in terminal: `python --version` or `python3 --version`. To install pip package manager run: `sudo apt install python3-pip`.

It's recommended to create a virtual environment to install the project's dependencies. Here's how you can do it:
1. Install the virtual environment package: `pip install virtualenv` (if you get a warning that virtualenv script is installed in directory which is not in $PATH, you should go into directory where it is installed - for me it was `/home/stjepan/.local/bin` - and move it into `/usr/local/bin`)
2. In terminal navigate to your project directory and create a virtual environment with: `virtualenv env-chosen-name` - if it is successful, you should see new directory in your project root named `env-chosen-name`
3. Activate the virtual environment: `source env-chosen-name/bin/activate` (You should see (env) prefixed to your terminal indicating that the virtual environment is now active.)

Install the project's dependencies with: `pip install numpy opencv-python tqdm scikit-learn keras`.
If you want to use PyCharm IDE for running scripts on click you need to install it with PyCharm going to "Interpreter Settings". Before installing it with PyCharm if you created new virtual environment it is recommended to add new python interpreter for virtual environment by going to "Add interpreter":

![img.png](images-for-readme/new-interpreter-for-virt-env.png)

## Database selection and preparation:
Ear images are downloaded from https://webctim.ulpgc.es/research_works/ami_ear_database/ and stored in the `data/01_raw` folder.

In terminal go to `src/data` directory and run `python3 prepare_data.py` to prepare the data for model, or you can open script `prepare_data.py` with PyCharm and click green triangle to run the script. The script will:
- resize images to 128 pixels in width and store it in `data/02_resized` directory,
- resized images will convert to grayscale and stored in `data/03_grayscale` directory,
- normalize to have values between 0 and 1 and saved as numpy arrays in `data/04_normalized` directory,
- take grayscale images and create arrays of images and labels split for model to train, test and validate.
