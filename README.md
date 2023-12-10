# Ear recognition using deep learning

## Database selection and preparation:
Ear images are downloaded from https://webctim.ulpgc.es/research_works/ami_ear_database/ and stored in the `data/01_raw` folder.

In terminal go to `src/data` directory and run `python3 prepare_data.py` to prepare the data for model. The script will:
- resize images to 128 pixels in width and store it in `data/02_resized` directory,
- resized images will convert to grayscale and stored in `data/03_grayscale` directory,
- normalize to have values between 0 and 1 and saved as numpy arrays in `data/04_normalized` directory,
- take grayscale images and create arrays of images and labels split for model to train, test and validate.

## Setup project environment
