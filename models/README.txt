This folder contains functions relevant to creating model definitions.

i3d_inception.py : I3D network definition from I3D-keras repository
                   (https://github.com/dlpbc/keras-kinetics-i3d)
                   Includes a custom function generate_logit to define
                   a softmax output for user-defined number of classes.

build_rgb_model.py : Contains function to create new model definitions
                     using the I3D backbone and pre-trained weights from
                     the Kinetics dataset.

default.hdf5 : example pre-trained weights for detecting SoccerNet events.
               Class labels:
                     0: substitutions
                     1: goals
                     2: cards (yellow/red cards)
                     3: background (non-events)
