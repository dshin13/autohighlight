from utils.event_detector import videoscan
from utils.clip_parser import scan_directories
from utils.timer import PrintTime

import numpy as np

import os
import re
import argparse

"""Video scanner to generate inferences from videos using a specified
classification model. All inferences are saved as serialized numpy
array files (.npy extension) in the same folder as the videos.

"""

def build_filter(pattern):
    """Function to return a filter function for scan_directories

    Parameters
    ----------
    pattern : str
        Regex pattern to match

    Returns
    -------
        function : a filter which returns true when a file matching
                   the pattern is present
    """

    def filter(files):

        for f in files:
            if re.match(pattern, f):
                return True

        return False

    return filter

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=str,
                        help="root directory to scan")
    parser.add_argument("-m", "--model", type=str,
                        help="model file path", default='./models/default.hdf5')
    parser.add_argument("-r", "--rate", type=int,
                        help="sampling rate", default=75)
    parser.add_argument("-e", "--ext", type=str,
                        help="video file extension", default='.mkv')
    args = parser.parse_args()

    if not args.source:
        raise Exception("Root directory for scanning not specified.")

    root = args.source

    pattern = '.*\{}$'.format(args.ext)

    filter_fcn = build_filter(pattern)

    dirs = scan_directories(root, filter_fcn)

    if not dirs:
        raise Exception("Root directory does not contain valid files.")

    model_dir = args.model

    sampling_rate = args.rate

    printer = PrintTime()
    printer.reset()

    for dir in dirs:
        # run videoscan on all video files in dirs
        vid_files = [f for f in os.listdir(dir) if re.match(pattern, f)]
        print('Processing directory: ' + dir)

        for v in vid_files:
            vid_path = os.path.join(dir, v)
            np_path = vid_path + '_pred.npy'
            out = videoscan(vid_path, model_dir, sampling_rate)
            np.save(np_path, out)
            print('Saved to path: ' + np_path)
            printer.time()

    print('All files processed.')
    printer.total()
