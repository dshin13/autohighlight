from utils.video_generator import VideoGenerator
from keras.models import load_model
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt


def return_classes_from_pred(y):
    """Converts a list of batches of class labels into 1D array

    Parameters
    ----------
    y : list
        List of batches of class labels
        (e.g. [[0,1], [1,0], [1,1], [1,0]] for a batch size of 2)
    """

    y = np.array(y)
    y = y.reshape(-1, y.shape[-1])
    y = np.argmax(y, 1)
    return y


def evaluate_model(steps, validation_generator, model):
    """Evaluates model using Keras model.predict method

    Parameters
    ----------
    steps : int
        Number of iterations to run the generator
    validation_generator : A generator object from VideoGenerator class
        Generates data for evaluation
    model : Keras model
        The model to run inference with
    """

    counter = 0

    y_pred = []
    y_true = []

    for X, y in validation_generator:

        yp = model.predict(X)

        y_pred.append(yp)
        y_true.append(y)
        counter += 1
        if counter > steps:
            break

    y_true = return_classes_from_pred(y_true)
    y_pred = return_classes_from_pred(y_pred)

    return y_true, y_pred


# Confusion matrix plotting code from sklearn examples
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """ 
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    plt.savefig('./confusion_test.png')
    return ax


if __name__ == '__main__':
    # Directories where training and test sets reside
    val_dir = './data/split_clips/test'
    train_dir = './data/split_clips/train'
    dims = (150,224,224,3)
    batch_size = 6

    videogen = VideoGenerator(train_dir, val_dir, dims, batch_size)

    validation_generator = videogen.generate(train_or_val="val")
    steps = len(videogen.filenames_val) // batch_size

    # Load model
    model = load_model('./models/default.hdf5')

    # Generate prediction
    y_true, y_pred = evaluate_model(steps, validation_generator, model)

    # Labels
    class_labels = [v for _, v in sorted(videogen.classname_by_id.items(), key=lambda x: x[0])]

    # Generate confusion matrix
    plot_confusion_matrix(y_true, y_pred, class_labels)
