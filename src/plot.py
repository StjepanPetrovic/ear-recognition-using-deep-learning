import itertools

import numpy as np
from matplotlib import pyplot as plt


def plot(history):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.tight_layout()
    plt.savefig('reports/report_plot.png', dpi=300)
    plt.close()


def plot_confusion_matrix(cm,
                          classes,
                          dirname,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues
                          ):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(f'{dirname}/confusion_matrix.png', dpi=300)
    plt.close()


def print_FAR_FRR(cm):
    num_classes = cm.shape[0]
    for i in range(num_classes):
        # get true positives, false negatives, false positives, and true negatives
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = cm.sum() - TP - FN - FP

        # calculate FAR and FRR
        FAR = FP / (FP + TN) if FP + TN > 0 else 0
        FRR = FN / (FN + TP) if FN + TP > 0 else 0

        print(f"Class {i}")
        print(f"True Positives (TP): {TP}")
        print(f"False Negatives (FN): {FN}")
        print(f"False Positives (FP): {FP}")
        print(f"True Negatives (TN): {TN}")
        print(f"False Acceptance Rate (FAR): {FAR}")
        print(f"False Rejection Rate (FRR): {FRR}")
        print("---------------------------------")
