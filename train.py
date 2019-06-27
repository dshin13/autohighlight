"""Training function for event detector model

"""

# python utility
import os
import argparse

# Generator class
from utils.video_generator import VideoGenerator

# Keras load_model and optimizer
from keras.models import load_model
from keras.optimizers import SGD

# Keras callbacks
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger

# helper function to freeze layers
def freeze_RGB_model(model, depth=152, trainable=False):
    """Freezes or unfreezes layers in Keras model for training

    Parameters
    ----------
    model : Keras model object
        Model whose layers are to be rendered trainable/untrainable
    depth : int
        Starting index of layers (from model.layers) to modify
    trainable : bool
        If set to True, unfreezes layers to allow weights to be updated
        during training; otherwise, freezes layers

    Notes
    -----
    Indices for Inception blocks are as follows:

    End of block 0: layer 31 (0-indexed)
    End of block 1: layer 51
    End of block 2: layer 72
    End of block 3: layer 92
    End of block 4: layer 112
    End of block 5: layer 132
    End of block 6: layer 152
    End of block 7: layer 173
    End of block 8: layer 193
    """

    for i, l in enumerate(model.layers):
        if i < depth:
            l.trainable = False
        else:
            l.trainable = trainable

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("train_dir", type=str,
                        help="directory where training set is located")
    parser.add_argument("val_dir", type=str,
                        help="directory where validation set is located")
    parser.add_argument("-b", "--batch_size", type=int, default=6,
                        help="batch size")
    parser.add_argument("-i", "--input", type=str, default="test_model.hdf5",
                        help="file path of serialized model definition and weights")
    parser.add_argument("-o", "--output", type=str, default='test_model_trained.hdf5',
                        help="file path to save the trained model as")
    parser.add_argument("-e", "--epochs", type=int, default=10,
                        help="number of epochs")
    args = parser.parse_args()


    # Directories where training and test sets reside
    train_dir = args.train_dir
    val_dir = args.val_dir

    # load model
    test_model = load_model(args.input)

    # extract input dimensions
    _, num_frames, num_height, num_width, num_channels = test_model.input.shape
    dims = (int(num_frames), int(num_height), int(num_width), int(num_channels))

    # batch size
    batch_size = args.batch_size

    videogen = VideoGenerator(train_dir, val_dir, dims, batch_size)
    classes = videogen.classname_by_id

    class_weight={}
    for id, name in classes.items():
        class_weight[id] = len(os.listdir(os.path.join(train_dir, name)))
    train_count = sum(class_weight.values())

    # Class weights to adjust class contribution to loss function
    # Weights are set proportional to inverse frequency
    for c in class_weight:
        class_weight[c] = train_count//class_weight[c]

    print(class_weight)
    
    # training/testing data generators and hyperparameters
    training_generator = videogen.generate(train_or_val='train', horizontal_flip=True, random_crop=True, random_start=True)
    training_steps_per_epoch = len(videogen.filenames_train) // batch_size
    validation_generator = videogen.generate(train_or_val="val")
    validation_steps_per_epoch = len(videogen.filenames_val) // batch_size

    # Stop process if model size does not match number of classes
    n_classes = len(classes)
    model_n_classes = int(test_model.output.shape[1])

    if model_n_classes != n_classes:
        raise Exception('Classes in training set : {}, expected : {}'.format(n_classes, model_n_classes))
    
    # Freeze I3D model (unfreeze last 2 layers only)
    freeze_RGB_model(test_model, depth=152, trainable=True)
    
    # Define optimizer
    #opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    opt = SGD(lr=0.001, decay=1e-7, momentum=0.9, nesterov=True)
    
    # Compile model
    test_model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    
    # Set the callbacks
    checkpointer = ModelCheckpoint(filepath='./weights/weights.{epoch:02d}-{val_loss:.2f}.hdf5', verbose=1)
    csv_logger = CSVLogger('log.csv', append=True, separator=';')
    
    # Fit the model
    test_model.fit_generator(training_generator,
                             steps_per_epoch=training_steps_per_epoch,
                             epochs=10,
                             class_weight=class_weight,
                             validation_data=validation_generator,
                             validation_steps=validation_steps_per_epoch,
                             callbacks=[csv_logger, checkpointer])
   
    # Save final model
    test_model.save('test_model_trained.hdf5')
