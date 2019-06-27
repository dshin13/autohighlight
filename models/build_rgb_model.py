# Python code to build an RGB-only Inflated3D network
# Outputs a file ('test_model.hdf5') in the same directory

# I3D model functions
from i3d_inception import Inception_Inflated3d
from i3d_inception import generate_logit

# Keras models and layers
from keras.models import Model
from keras.layers import Activation
from keras.layers import Dropout

import argparse

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

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("NUM_FRAMES", type=int,
                        help="number of frames processed by model during inference")
    parser.add_argument("FRAME_HEIGHT", type=int,
                        help="height of frames processed by model")
    parser.add_argument("FRAME_WIDTH", type=int,
                        help="width of frames processed by model")
    parser.add_argument("NUM_CLASSES", type=int,
                        help="number of event classes including neg/background")
    parser.add_argument("-d", "--dropout", type=float, default=0.5,
                        help="dropout probability for classification layer")
    parser.add_argument("-o", "--output", type=str, default='test_model.hdf5',
                        help="file path to save the model")
    args = parser.parse_args()

    # Input dimensions and dropout probability
    NUM_FRAMES=args.NUM_FRAMES
    FRAME_HEIGHT=args.FRAME_HEIGHT
    FRAME_WIDTH=args.FRAME_WIDTH
    NUM_CLASSES=args.NUM_CLASSES
    dropout_prob=args.dropout
    
    # Instantiate model
    test_model = RGB_model(NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, NUM_CLASSES, dropout_prob)
    
    # Save the model
    test_model.save(args.output)

    print('Model built for {}-class classification of ({}x{}x{}x3) video clips.'.format(NUM_CLASSES, NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH))
    print('Saved as {}'.format(args.output))
