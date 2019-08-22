""" Intermediate activation visualizer for I3D backbone
This code extracts intermediate activations from I3D model
(Concatenated stack of outputs from all filters at the end of each block)
by running inference on an input clip.
Video processing is done using a set of functions provided in 
utils.video_writer.py.

Usage
-----

$ python visualize_activation.py <videopath> -o <output>
(if output not specified, defaults to saving output as overlay.mp4 in pwd)

For other optional arguments, run:
$ python visualize_activation.py -h
"""

from utils.video_generator import VideoGenerator
from utils.video_writer import *

import numpy as np

import argparse

# locate all concat layers for visualization
def return_concat_layers(layers):
    """Returns a list of every concatenation layer at the end of each I3D block
    
    Parameters
    ----------
    layers : list
        List of Layer objects retrieved from Model.layers method
    
    Returns
    -------
        outputs : list of Layer objects corresponding to concatenation layers
    """
    outputs = []
    for i, l in enumerate(layers):
        if 'Mixed' in l.name:
            outputs.append(l.output)
            
    return outputs

# build probe model
def build_probe_model(modelpath):
    """Builds probe model which returns all intermediate outputs from I3D blocks.
    
    Parameters
    ----------
    modelpath : str
        Path of model to be loaded
        
    Returns
    -------
    probe_model : Keras Model object
    
    """
    
    from keras.models import load_model
    model = load_model(modelpath)
    layers = model.layers
    
    outputs = return_concat_layers(layers)
    
    # append final output to keep track of prediction
    outputs.append(model.output)

    from keras.models import Model
    probe_model = Model(input=model.input, output=outputs)
    
    return probe_model

def extract_activation(outputs, block, rule):
    """Extracts outputs from a specified I3D block and filter rule.
    
    Parameters
    ----------
    outputs : list
        List containing outputs at each stage of I3D block
    block : int
        Index of block to be retrieved (range [0, 8])
    rule : str
        Specifies channel selection rule.
            'max' : retrieves channel having an element with the highest response
            'mean' : retrieves channel with the highest average response over space and time
            
    Returns
    -------
        Numpy array : an array of shape (n_frames, height, width)
    """

    block_output = outputs[block][0]
    if rule=="max":
        # find out which filter has the largest activation (based on sum over H x W)
        filt = np.argmax(np.max(block_output[2:-2,:,:,:], axis=(0, 1, 2)))
    if rule=="mean":
        filt = np.argmax(np.sum(block_output, axis=(0, 1, 2)))

    activation = block_output[:,:,:,filt]

    # normalize to fall within range (0, 1)
    activation = (activation - activation.min()) / activation.max()

    return activation


# build model
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("videopath", help="input video path")
    parser.add_argument("-o", "--output", default="overlay.mp4",
                        help="output video path")
    parser.add_argument("-m", "--model", default="models/default.hdf5",
                        help="model file for inspection")
    parser.add_argument("-b", "--block", default=3, type=int,
                        help="I3D block (0-indexed) for probing")
    parser.add_argument("-r", "--rule", default="max",
                        help="rule for selecting output channel (select max activation or mean activation)")
    parser.add_argument("-a", "--alpha", type=float, default=0.5,
                        help="alpha value (fraction of overlay contribution in image)")    
    args = parser.parse_args()

    # Step 1: build probe model for activation signal extraction
    probe_model = build_probe_model(args.model)    
    
    # Directories where training set and input data reside
    videopath = args.videopath
    dims = (150,224,224,3)

    # Step 2: use vid2npy class from VideoGenerator to extract frames from video
    vgen = VideoGenerator('.', '.', dims, 1)
    X = vgen.vid2npy(videopath, dims[0])
    X_in = np.array([X])
        
    # run prediction to generate activations
    y = probe_model.predict(X_in)

    # Extract activations of interest
    activation = extract_activation(y, args.block, args.rule)
    
    # match frames and dimensions (propagate activation over all 3 RGB channels)
    vid_overlay = match_frames_and_dims(activation,
                                        X,
                                        overlay=True,
                                        alpha=args.alpha)
    
    # export overlay frames into video
    write_to_video(vid_overlay, args.output)
