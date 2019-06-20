# os utility
import os

# Generator class
from utils.video_generator import VideoGenerator

# Keras load_model and optimizer
from keras.models import load_model
from keras.optimizers import SGD

# Keras callbacks
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger

# helper code to freeze layers
def freeze_RGB_model(model, depth=0, trainable=False):
    freeze_layers = model.layers
    for layer in freeze_layers[depth:]:
        layer.trainable=trainable
        

if __name__ == '__main__':
    
    # Directories where training and test sets reside
    val_dir = './data/split_clips/val'
    train_dir = './data/split_clips/train'
    dims = (150,224,224,3)
    batch_size = 6
    videogen = VideoGenerator(train_dir, val_dir, dims, batch_size)
    classes = videogen.classname_by_id

    class_weight={}
    for id, name in classes.items():
        class_weight[id] = len(os.listdir(os.path.join(train_dir, name)))
    train_count = sum(class_weight.values())

    # Class weights to balance contribution to loss fcn
    for c in class_weight:
        class_weight[c] = train_count//class_weight[c]

    print(class_weight)
    
    # training/testing data generators and hyperparameters
    training_generator = videogen.generate(train_or_val='train', horizontal_flip=True, random_crop=True, random_start=True)
    training_steps_per_epoch = len(videogen.filenames_train) // batch_size
    validation_generator = videogen.generate(train_or_val="val")
    validation_steps_per_epoch = len(videogen.filenames_val) // batch_size
    
    # Load model
    test_model = load_model('./weights/0618_finetune/weights.08-0.64.hdf5')
    
    # Freeze I3D model (unfreeze last 2 layers only)
    # freeze_RGB_model(test_model, depth=132, trainable=False)
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
