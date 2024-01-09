import numpy as np
from keras.src.utils.np_utils import to_categorical
from sklearn.metrics import confusion_matrix

from model.model_cnn import create_model
from data.split_data_set import split_dataset
from src.plot import plot, plot_confusion_matrix


def main():
    ((img_training_set, label_training_set),
     (img_validation_set, label_validation_set),
     (img_test_set, label_test_set)) = split_dataset('../data/04_grayscale/')

    num_classes = 5

    label_training_set = np.where(
        label_training_set > num_classes - 2,
        num_classes - 1,
        label_training_set
    )

    label_validation_set = np.where(
        label_validation_set > num_classes - 2,
        num_classes - 1,
        label_validation_set
    )

    label_test_set = np.where(
        label_test_set > num_classes - 2,
        num_classes - 1,
        label_test_set
    )

    training_labels = to_categorical(label_training_set, num_classes)
    test_labels = to_categorical(label_test_set, num_classes)
    validation_labels = to_categorical(label_validation_set, num_classes)

    model = create_model((182, 128, 1), num_classes)

    print(model.summary())

    history = model.fit(
        x=img_training_set,
        y=training_labels,
        epochs=10,
        validation_data=(img_validation_set, validation_labels),
        verbose=2
    )

    predictions = model.predict(
        x=img_test_set,
        verbose=2
    )

    model.save('model.keras')

    cm = confusion_matrix(y_true=label_test_set, y_pred=np.argmax(predictions, axis=-1))
    cm_plot_labels = ['0', '1', '2', '3', '4']
    plot_confusion_matrix(cm=cm, classes=cm_plot_labels, dirname='reports/')

    plot(history)


if __name__ == '__main__':

    main()
