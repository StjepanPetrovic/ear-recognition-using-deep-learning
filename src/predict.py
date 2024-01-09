import numpy as np
from keras.models import load_model
import cv2

from src.data.split_dataset import normalize_image

if __name__ == '__main__':

    model = load_model('model/output/model_15.keras')

    new_img = cv2.imread('../data/new_data.jpg', cv2.IMREAD_GRAYSCALE)
    new_img = normalize_image(new_img)
    new_img = np.expand_dims(new_img, axis=0)

    predictions = model.predict(new_img)
    predicted_label = np.argmax(predictions, axis=1)

    print('Predicted label:', predicted_label)
