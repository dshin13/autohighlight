# Python code to build an RGB-only Inflated3D network
# Outputs a file ('test_model.hdf5') in the same directory

# I3D model functions
from i3d_inception import Inception_Inflated3d
from i3d_inception import generate_logit

# Keras models and layers
from keras.models import Model
from keras.models import load_model
from keras.layers import Activation
from keras.layers import Dropout

# Single stream 3D convolution model from pretraind weights (binary classifier trained in 6/14)
def RGB_model(NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, NUM_CLASSES, dropout_prob):

    rgb_model = load_model('../weights/0614_run_2_finetune/weights.03-0.34.hdf5')

    x1 = rgb_model.layers[-6].output # extract output of last avg pool layer
    x1 = Dropout(dropout_prob)(x1)

    x1 = generate_logit(x1, '1x1_Conv3d_rgb_logits', NUM_CLASSES)

    x = Activation('softmax', name='prediction')(x1)

    model = Model(input=rgb_model.input, output=x)

    return model

if __name__ == '__main__':
    
    # Input dimensions and dropout probability
    NUM_FRAMES=150
    FRAME_HEIGHT=224
    FRAME_WIDTH=224
    NUM_CLASSES=4
    dropout_prob=0.5
    
    # Instantiate model
    test_model = RGB_model(NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, NUM_CLASSES, dropout_prob)    
    
    # Save the model
    test_model.save('test_model_4classes.hdf5')
   
