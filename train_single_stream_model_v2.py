# Generator class
import video_generator

# I3D model functions
from i3d_inception import Inception_Inflated3d
from i3d_inception import conv3d_bn

# Keras models and layers
from keras import backend as K
from keras.models import Model
from keras.layers import Activation
from keras.layers import Add
from keras.layers import Dropout
from keras.layers import Reshape
from keras.layers import Lambda

# Keras callbacks
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger

# 1x1x1 3d convolution + logit function for video segment classification
def generate_logit(x, last_conv3d_name, classes):
    
    x = conv3d_bn(x, classes, 1, 1, 1, padding='same',
                   use_bias=True, use_activation_fn=False, use_bn=False, name=last_conv3d_name)

    num_frames_remaining = int(x.shape[1])
    x = Reshape((num_frames_remaining, classes))(x)

    x = Lambda(lambda x: K.mean(x, axis=1, keepdims=False),
               output_shape=lambda s: (s[0], s[2]))(x)    
    
    return x

# Single stream 3D convolution model in RGB channel using ImageNet/Kinetics weights
def RGB_model(NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, NUM_CLASSES, dropout_prob):

    rgb_model = Inception_Inflated3d(
        include_top=False,
        weights='rgb_imagenet_and_kinetics',
        input_shape=(NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, 3))

    x1 = rgb_model.layers[-1].output
    x1 = Dropout(dropout_prob)(x1)

    x1 = generate_logit(x1, '1x1_Conv3d_rgb_logits', NUM_CLASSES)

    x = Activation('softmax', name='prediction')(x1)

    model = Model(input=rgb_model.input, output=x)

    return model

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
    batch_size = 4
    videogen = video_generator.VideoGenerator(train_dir, test_dir, dims, batch_size)
    
    # training/testing data generators and hyperparameters
    training_generator = videogen.generate(train_or_test='train')
    training_steps_per_epoch = len(videogen.filenames_train) // batch_size
    testing_generator = videogen.generate(train_or_test="test")
    testing_steps_per_epoch = len(videogen.filenames_test) // batch_size
    
    # Input dimensions and dropout probability
    NUM_FRAMES=250
    FRAME_HEIGHT=224
    FRAME_WIDTH=224
    NUM_CLASSES=2
    dropout_prob=0.5
    
    # Instantiate model
    test_model = RGB_model(NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, NUM_CLASSES, dropout_prob)
    
    # freeze I3D model
    freeze_RGB_model(test_model)
    
    # Compile model
    test_model.compile(optimizer='adam',
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
   
