import os

import cv2
from tqdm import tqdm
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, save_img


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


def convert_to_grayscale(augmented_data_directory, resized_data_directory):
    augmented_ear_images = os.listdir(augmented_data_directory)
    resized_ear_images = os.listdir(resized_data_directory)

    all_images = augmented_ear_images + resized_ear_images

    grayscale_data_directory = '../../data/04_grayscale'

    if not os.path.exists(grayscale_data_directory):
        os.makedirs(grayscale_data_directory)

    for filename in tqdm(all_images):
        if filename in augmented_ear_images:
            img_path = os.path.join(augmented_data_directory, filename)
        else:
            img_path = os.path.join(resized_data_directory, filename)

        img = cv2.imread(img_path)

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        new_img_path = os.path.join(grayscale_data_directory, filename)

        cv2.imwrite(new_img_path, gray_img)

    print("Converting to grayscale is completed.")


def augment_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    datagen = ImageDataGenerator(
        rotation_range=40,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2])

    for filename in tqdm(os.listdir(input_dir)):
        person_num = int(filename.split('_')[0])

        if person_num < 4:
            img_path = os.path.join(input_dir, filename)

            # Load image and reshape it to a 4D tensor
            image = load_img(img_path)
            x = img_to_array(image)
            x = x.reshape((1,) + x.shape)

            # Iterate over augmented images and save them to the output directory
            i = 0
            for batch in datagen.flow(x, batch_size=1):
                aug_img_path = os.path.join(output_dir, filename + '_aug_' + str(i) + '.jpg')
                save_img(aug_img_path, batch[0])
                i += 1

                # here we define how many augmented images for one original image we want
                if i >= 20:
                    break

    print("Augmenting is completed.")


def delete_left_ears(raw_data_directory):
    for filename in tqdm(os.listdir(raw_data_directory)):
        if 'left' in filename:
            file_path = os.path.join(raw_data_directory, filename)
            os.remove(file_path)

    print("Deleting left ears is completed.")


if __name__ == '__main__':

    delete_left_ears('../../data/01_raw/')

    resize_images('../../data/01_raw/')

    augment_images('../../data/02_resized', '../../data/03_augmented/')

    convert_to_grayscale('../../data/03_augmented/', '../../data/02_resized')
