import cv2
import numpy as np
from keras.models import load_model

# returns an array of probability for each class at timepoints sampled at the specified rate.
# e.g. event_detector(filename, model, 40) will sample the input file at 40 frame intervals
# and concatenate 

def videoscan(filename, model_file, sampling_interval):
        
    model = load_model(model_file) # Loads Keras model/weights for prediction
        
    cap = cv2.VideoCapture(filename)
    num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    
    steps_total = int(num_frames // sampling_interval)
    
    frames = []
    probs = []
    
    count = 0
                      
    while True:
        ret, frame = cap.read()
        
        if ret == True:
            # take a center crop of the video frame
            center_crop = (frame.shape[1] - frame.shape[0])//2
            frame = frame[:,center_crop:center_crop+frame.shape[0],:]
            frames.append(frame)
                
        else:
            break
        
        if len(frames) == 150:
            print('150!')
            count += 1
            print("Processed {} out of {}".format(count, steps_total))
            example = np.array([frames])
            example = (example / 255) * 2 - 1
            prob = model.predict(example)
            print('frames collected')
            print(prob[0])
            # append probabilities and clear frames buffer by sampling interval
            probs.append(prob[0])
            
            if sampling_interval < 150:
                frames = frames[sampling_interval:]
            else:
                frames = []
                
    output = np.array(probs)
    return output
    
    
