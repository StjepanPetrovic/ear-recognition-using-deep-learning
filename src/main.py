import numpy as np
from keras.src.utils.np_utils import to_categorical

from model.model_cnn import create_model
from data.prepare_data import split_dataset
from src.plot import plot


def main():
    ((img_training_set, label_training_set),
     (img_validation_set, label_validation_set),
     (img_test_set, label_test_set)) = split_dataset('../data/03_grayscale/')

    label_training_set = np.where(label_training_set > 49, 50, label_training_set)
    label_validation_set = np.where(label_validation_set > 49, 50, label_validation_set)
    label_test_set = np.where(label_test_set > 49, 50, label_test_set)

    num_classes = 51

    training_labels = to_categorical(label_training_set, num_classes)
    test_labels = to_categorical(label_test_set, num_classes)
    validation_labels = to_categorical(label_validation_set, num_classes)

    model = create_model((182, 128, 1), num_classes)

    print(model.summary())

    epochs = 10

    history = model.fit(
        img_training_set,
        training_labels,
        epochs=epochs,
        validation_data=(img_validation_set, validation_labels)
    )

    loss, accuracy = model.evaluate(img_test_set, test_labels)

    print('Test loss:', loss)
    print('Test accuracy:', accuracy)

    model.save('model.keras')

    plot(history)


if __name__ == '__main__':

    main()
