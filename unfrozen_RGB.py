# Generator class
import utils.video_generator

# I3D model functions
from models.i3d_inception import Inception_Inflated3d
from models.i3d_inception import generate_logit

# Keras load_model and optimizer
from keras.models import load_model
from keras.optimizers import Adam

# Keras callbacks
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger

# helper code to freeze layers
def freeze_RGB_model(model, trainable=False):
    freeze_layers = model.layers[:-5]
    for layer in freeze_layers:
        layer.trainable=trainable
        

if __name__ == '__main__':
    
    # Directories where training and test sets reside
    test_dir = './data/val'
    train_dir = './data/train'
    dims = (250,224,224,3)
    batch_size = 16
    videogen = video_generator.VideoGenerator(train_dir, test_dir, dims, batch_size)
    
    # training/testing data generators and hyperparameters
    training_generator = videogen.generate(train_or_test='train')
    training_steps_per_epoch = len(videogen.filenames_train) // batch_size
    testing_generator = videogen.generate(train_or_test="test")
    testing_steps_per_epoch = len(videogen.filenames_test) // batch_size
    
    # Load model
    test_model = load_model('./weights/0612_RGBonly_3epochs.hdf5')
    
    # Unfreeze I3D model
    freeze_RGB_model(test_model)
    
    # Define optimizer
    opt = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    
    # Compile model
    test_model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    
    # Set the callbacks
    checkpointer = ModelCheckpoint(filepath='/tmp/weights.hdf5', verbose=1, save_best_only=True)
    csv_logger = CSVLogger('log.csv', append=True, separator=';')
    
    # Fit the model
    test_model.fit_generator(training_generator,
                             steps_per_epoch=training_steps_per_epoch,
                             epochs=10,
                             validation_data=testing_generator,
                             validation_steps=testing_steps_per_epoch,
                             callbacks=[csv_logger, checkpointer])
   
