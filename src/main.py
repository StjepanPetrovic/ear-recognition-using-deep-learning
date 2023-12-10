import cv2
import numpy as np
from matplotlib import pyplot as plt

from model.model_cnn import create_model
from data.prepare_data import split_dataset, normalize_image


def main():
    ((img_training_set, label_training_set),
     (img_validation_set, label_validation_set),
     (img_test_set, label_test_set)) = split_dataset('../data/03_grayscale/')

    training_labels = np.asarray(label_training_set).astype('float32').reshape((-1, 1))
    test_labels = np.asarray(label_test_set).astype('float32').reshape((-1, 1))
    validation_labels = np.asarray(label_validation_set).astype('float32').reshape((-1, 1))

    model = create_model((182, 128, 1), 106)

    print(model.summary())

    EPOCHS = 10

    history = model.fit( # ovdje PUCAAAAAAAAAAAAAA nesto nije ok
        img_training_set,
        training_labels,
        epochs=EPOCHS,
        validation_data=(img_test_set, test_labels)
    )

    loss, accuracy = model.evaluate(img_validation_set, validation_labels)

    print('Test loss:', loss)
    print('Test accuracy:', accuracy)

    # write code for importing new image and predicting the class
    new_img = cv2.imread('../data/new_data.jpg', cv2.IMREAD_GRAYSCALE)
    new_img = normalize_image(new_img)

    predictions = model.predict(new_img)
    predicted_label = np.argmax(predictions, axis=1)

    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.tight_layout()
    plt.savefig('reports/figures/report_plot.png', dpi=300)
    plt.close()


main()
