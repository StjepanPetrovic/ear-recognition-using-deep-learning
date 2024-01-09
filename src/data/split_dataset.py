import os

from matplotlib import pyplot as plt
from tqdm import tqdm
import cv2
import numpy as np
from sklearn.model_selection import train_test_split


def split_dataset(grayscale_data_directory):
    grayscale_imgs = os.listdir(grayscale_data_directory)
    grayscale_imgs.sort()

    imgs_array = []
    labels = []

    for filename in tqdm(grayscale_imgs):
        img_path = os.path.join(grayscale_data_directory, filename)

        label = int(filename.split('_')[0])

        labels.append(label)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        img = normalize_image(img)

        imgs_array.append(img)

        # check if the image is correct
        #
        # if len(imgs_array) % 50 == 0:
        #     print(grayscale_imgs[len(imgs_array) - 1])
        #
        #     plt.figure()
        #     plt.imshow(img, cmap='gray')
        #     plt.title(f'Image {len(imgs_array)} - Label {label}')
        #     plt.show()

    imgs_array = np.array(imgs_array)
    labels = np.array(labels)

    img_train, img_temp, label_train, label_temp = train_test_split(
        imgs_array,
        labels,
        test_size=0.4,
        random_state=42,
        shuffle=True,
        stratify=labels
    )

    img_val, img_test, label_val, label_test = train_test_split(
        img_temp,
        label_temp,
        test_size=0.5,
        random_state=42,
        shuffle=True,
        stratify=label_temp
    )

    return (img_train, label_train), (img_val, label_val), (img_test, label_test)


def normalize_image(img):
    return cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
