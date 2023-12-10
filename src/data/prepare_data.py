import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split


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


def split_dataset(grayscale_data_directory):
    grayscale_imgs = os.listdir(grayscale_data_directory)
    grayscale_imgs.sort()

    imgs_array = []
    labels = []

    for filename in tqdm(grayscale_imgs):
        img_path = os.path.join(grayscale_data_directory, filename)

        label = filename.split('_')[0]
        labels.append(int(label))

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        imgs_array.append(img)

    imgs_array = np.array(imgs_array)
    labels = np.array(labels)

    img_train, img_temp, label_train, label_temp = train_test_split(imgs_array, labels, test_size=0.4, random_state=42)
    img_val, img_test, label_val, label_test = train_test_split(img_temp, label_temp, test_size=0.5, random_state=42)

    return (img_train, label_train), (img_val, label_val), (img_test, label_test)


if __name__ == '__main__':

    resize_images('../../data/01_raw/')
    convert_to_grayscale('../../data/02_resized/')
    normalize_images('../../data/03_grayscale/')
    ((img_training_set, label_training_set),
     (img_validation_set, label_validation_set),
     (img_test_set, label_test_set)) = split_dataset('../../data/03_grayscale/')
