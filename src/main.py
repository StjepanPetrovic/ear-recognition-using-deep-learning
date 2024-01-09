import numpy as np
from keras.src.utils.np_utils import to_categorical
from sklearn.metrics import confusion_matrix

from model.model_cnn import create_model
from data.split_dataset import split_dataset
from src.plot import plot, plot_confusion_matrix


def preprocess_labels(label_dataset, num_classes):
    return np.where(
        label_dataset > num_classes - 2,
        num_classes - 1,
        label_dataset
    )


def main():

    #
    # DATA PREPARATION
    #

    ((img_training_set, label_training_set),
     (img_validation_set, label_validation_set),
     (img_test_set, label_test_set)) = split_dataset('../data/04_grayscale/')

    NUMBER_OF_CLASSES = 5

    label_training_set = preprocess_labels(label_training_set, NUMBER_OF_CLASSES)
    label_validation_set = preprocess_labels(label_validation_set, NUMBER_OF_CLASSES)
    label_test_set = preprocess_labels(label_test_set, NUMBER_OF_CLASSES)

    training_labels = to_categorical(label_training_set, NUMBER_OF_CLASSES)
    validation_labels = to_categorical(label_validation_set, NUMBER_OF_CLASSES)

    #
    # WORKING WITH MODEL
    #

    model = create_model((182, 128, 1), NUMBER_OF_CLASSES)

    print(model.summary())

    history = model.fit(
        x=img_training_set,
        y=training_labels,
        epochs=1,
        validation_data=(img_validation_set, validation_labels),
        verbose=2
    )

    predictions = model.predict(
        x=img_test_set,
        verbose=2
    )

    model.save('model/output/model.keras')

    #
    # PLOTTING FOR REPORTS
    #

    cm_plot_labels = [str(i) for i in range(NUMBER_OF_CLASSES)]

    cm = confusion_matrix(
        y_true=label_test_set,
        y_pred=np.argmax(predictions, axis=-1)
    )

    plot_confusion_matrix(cm=cm, classes=cm_plot_labels, dirname='reports/')

    plot(history)


if __name__ == '__main__':
    main()
