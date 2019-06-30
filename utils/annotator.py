## Annotator class: extracts annotation from inference data and has a video summarizer method

import pandas as pd
import numpy as np
import os

class Annotator:
    """A class to annotate videos using event inference output from videoscan function
    """

    def __init__(self, npy, classes_lst, thresh, rate=75):
        """ 
        Parameters
        ----------
        npy : numpy.array
            Array containing the output from videoscan function
            having shape (time steps x number of classes)
        classes_lst : list
            List containing class indices to be used for annotation
            e.g. [1,2] selects only goals and cards for annotation
        thresh : list
            List containing threshold values to incorporate an inferred
            event in the final annotation
            e.g. [0.9,0.95] selects only timestamps which have class
            probability of 0.9 or greater for class 1, and 0.95 or greater
            for class 2
        rate : int
            Sampling rate of inference, in number of frames
        """

        self.classes_lst = classes_lst
        self.thresh = thresh
        self.npy = npy
        self.rate = rate
        self.fps = 25.0
        self.events_sec = []

    def get_fps(self, source):
        """Function to extract frames per second from source video

        Parameters
        ----------
        source : str
            Path name of source video file
        """

        import cv2
        cap = cv2.VideoCapture(source)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        print("Source FPS : {}".format(self.fps))


    # Helper function for timestamp extraction
    def _extractor(self, vector, cls, npy):
        """Helper function for timestamp extraction from a specified class

        Parameters
        ----------
        vector : numpy.array
            Mask containing boolean values to identify indices above threshold
        cls : int
            Class index
        npy : numpy.array
            Original array used to retrieve probabilities

        Returns
        -------
            int : length of timestamp
            list : contains all timestamps with elements
                   [timestamp, class index, probability]
        """
        timestamp = []
        
        for i in range(vector.shape[0]):
            if vector[i]:
                sec = i / self.fps * self.rate + 3
                prob = float(npy[i, cls])
                timestamp.append([sec, cls, prob])

        return len(timestamp), timestamp

    # timestamp extractor (updates events_sec which contains timestamp of events for editing)
    def extract_timestamp(self, npy=None, classes_lst=None, thresh=None):
        """Timestamp extraction function
        Saves extracted timestamp into the attribute 'events_sec'

        Parameters
        ----------
        npy : numpy.array
        classes_lst = list
        thresh = list
        """

        events_sec = []
        
        if not npy:
            npy = self.npy
            
        if not classes_lst:
            classes_lst=self.classes_lst
            
        if not thresh:
            thresh=self.thresh
        
        for cls, t in zip(classes_lst, thresh):
            vec = npy[:,cls] > t
            events, timestamp = self._extractor(vec, cls, npy)
            self.events_sec = self.events_sec + timestamp

        self.events_sec = sorted(self.events_sec, key=lambda x: x[0])
        
    # save function (for exporting timestamps and class predictions to json)
    def save(self, output_dir=None):
        """Saves generated labels to json file

        Parameters
        ----------
        output_dir : str
            Target directory to save json file
        """

        import json
        
        if not output_dir:
            output_dir = '.'
        
        output_path = os.path.join(output_dir, 'Annotations.json')

        data = {'annotations': self.events_sec}
        
        with open(output_path, 'w') as file:
            json.dump(data, file, indent=4) 
            
        print('Annotations saved to ' + output_path)
        
    def load(self, input_path):
        import json
        
        with open(input_path) as file:
            data = json.load(file)
            
        self.events_sec = data['annotations']
        print('Annotations loaded from ' + input_path)
    
    
    def summarize(self, input_dir, output_path=None):
        """Function to generate a summary video from source file using
        information in 'events_sec' attribute.

        Parameters
        ----------
        input_dir : str
            Path to source video file
        output_path : str
            Target path to summary video file
        """
        from moviepy.editor import VideoFileClip, concatenate_videoclips
        
        source = VideoFileClip(input_dir)
        
        if not output_path:
            filename = os.path.basename(input_dir) + '_summary.mp4'
            output_path = os.path.join(os.path.dirname(input_dir), filename)
            

        summary_list = []

        start = 0
        end = 0

        for i, _, _ in self.events_sec:

            # If event detected over multiple windows, pop previous scene and lengthen end timestamp
            if i < end:
                summary_list.pop()
            else:
                start = max(i - 5, 0)

            end = i + 5
            sub = source.subclip(start,end)
            summary_list.append(sub)

        summary = concatenate_videoclips(summary_list)
        summary.write_videofile(output_path, codec='libx264', verbose=None)
        print('Summary saved to ' + output_path)
