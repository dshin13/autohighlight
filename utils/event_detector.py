""" 
Returns an array of probability for each class at timepoints sampled at the specified rate.
e.g. event_detector(filename, model, 40) will sample the input file at 40 frame intervals
and concatenate
"""

import cv2
import numpy as np
from keras.models import load_model


def videoscan(filename, model_file, sampling_interval, resize=True, verbose=False):
    """Scans video files and performs inference using the provided model

    Approach
    --------
    Inference is performed on clips of size specified by the input size
    of the model. The entire video is sampled at the rate specified by
    the parameter sampling_interval.

    Usage
    -----
    arr = videoscan(video_filename, model, interval)
    np.save(arr, 'labels.npy')


    Parameters
    ----------
    filename : str
        Path of the video source to be scanned
    model_file : serialized Keras model file (hdf5)
        A serialized file containing Keras model definition and weights
    sampling_interval : int
        Number of frames to shift after each inference
    resize : bool (default = True)
        Toggles resize (set to False if spatial cropping has been used)
    verbose : bool (default = False)
        Flag to indicate whether to print progress

    Returns
    -------
        A numpy array containing class inference result for each time interval
        with the shape (number of time steps X number of classes)
    """

    model = load_model(model_file) # Loads Keras model/weights for prediction
    input_frames = int(model.input.shape[1]) # number of frames to collect
    dim_h = int(model.input.shape[2]) # height of input frames
    dim_w = int(model.input.shape[3]) # width of input frames

    cap = cv2.VideoCapture(filename)
    num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    
    steps_total = int(num_frames // sampling_interval)
    
    frames = []
    probs = []
    
    count = 0

    while True:
        ret, frame = cap.read()
        
        if ret == True:

            if resize:
                frame = cv2.resize(frame, (dim_w, dim_h), interpolation=cv2.INTER_AREA)
            else:
                # take a center crop of the video frame
                center_crop = (frame.shape[1] - frame.shape[0])//2
                frame = frame[:,center_crop:center_crop+frame.shape[0],:]

            frames.append(frame)
                
        else:
            break
        
        if len(frames) == input_frames: #replace with dimension 1 of model input
            count += 1
            if verbose:
                print("Processed {} out of {}".format(count, steps_total))
            example = np.array([frames])
            example = (example / 255) * 2 - 1
            prob = model.predict(example)
            #print(prob[0])
            # append probabilities and clear frames buffer by sampling interval
            probs.append(prob[0])
            
            if sampling_interval < 150:
                frames = frames[sampling_interval:]
            else:
                frames = []
                
    output = np.array(probs)
    return output
