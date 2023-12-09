import os
import cv2
from tqdm import tqdm


def resize_images(raw_data_directory):
    raw_ear_images = os.listdir(raw_data_directory)

    for filename in tqdm(raw_ear_images):
        img_path = os.path.join(raw_data_directory, filename)

        resized_data_directory = '../../data/resized'
        new_filename = filename.replace('.jpg', '_resized.jpg')
        new_img_path = os.path.join(resized_data_directory, new_filename)

        img = cv2.imread(img_path)

        h, w = img.shape[:2]
        aspect_ratio = h / w

        new_width = 128
        new_height = int(new_width * aspect_ratio)

        resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

        if not os.path.exists(resized_data_directory):
            os.makedirs(resized_data_directory)

        cv2.imwrite(new_img_path, resized_img)

    print("Resizing is completed.")


def convert_to_grayscale(resized_data_directory):
    resized_ear_images = os.listdir(resized_data_directory)

    for filename in tqdm(resized_ear_images):
        img_path = os.path.join(resized_data_directory, filename)

        grayscale_data_directory = '../../data/grayscale'
        new_filename = filename.replace('.jpg', '_grayscale.jpg')
        new_img_path = os.path.join(grayscale_data_directory, new_filename)

        img = cv2.imread(img_path)

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if not os.path.exists(grayscale_data_directory):
            os.makedirs(grayscale_data_directory)

        cv2.imwrite(new_img_path, gray_img)

    print("Converting to grayscale is completed.")


if __name__ == '__main__':

    resize_images('../../data/raw/')
    convert_to_grayscale('../../data/resized/')
