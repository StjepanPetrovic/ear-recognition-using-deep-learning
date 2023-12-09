import os
import cv2
import numpy as np
from tqdm import tqdm


def resize_images(raw_data_directory):
    raw_ear_images = os.listdir(raw_data_directory)

    resized_data_directory = '../../data/02_resized'

    if not os.path.exists(resized_data_directory):
        os.makedirs(resized_data_directory)

    for filename in tqdm(raw_ear_images):
        img_path = os.path.join(raw_data_directory, filename)

        img = cv2.imread(img_path)

        h, w = img.shape[:2]
        aspect_ratio = h / w

        new_width = 128
        new_height = int(new_width * aspect_ratio)

        resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

        new_img_path = os.path.join(resized_data_directory, filename)

        cv2.imwrite(new_img_path, resized_img)

    print("Resizing is completed.")


def convert_to_grayscale(resized_data_directory):
    resized_ear_images = os.listdir(resized_data_directory)

    grayscale_data_directory = '../../data/03_grayscale'

    if not os.path.exists(grayscale_data_directory):
        os.makedirs(grayscale_data_directory)

    for filename in tqdm(resized_ear_images):
        img_path = os.path.join(resized_data_directory, filename)

        img = cv2.imread(img_path)

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        new_img_path = os.path.join(grayscale_data_directory, filename)

        cv2.imwrite(new_img_path, gray_img)

    print("Converting to grayscale is completed.")


def normalize_images(grayscale_data_directory):
    grayscale_ear_images = os.listdir(grayscale_data_directory)

    normalized_data_directory = '../../data/04_normalized'

    if not os.path.exists(normalized_data_directory):
        os.makedirs(normalized_data_directory)

    for filename in tqdm(grayscale_ear_images):
        img_path = os.path.join(grayscale_data_directory, filename)

        filename = filename.split('.')[0] + '.npy'

        new_img_path = os.path.join(normalized_data_directory, filename)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        normalized_img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        np.save(new_img_path, normalized_img)

    print("Normalizing is completed.")


if __name__ == '__main__':

    resize_images('../../data/01_raw/')
    convert_to_grayscale('../../data/02_resized/')
    normalize_images('../../data/03_grayscale/')
