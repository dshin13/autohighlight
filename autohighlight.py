from utils.annotator import Annotator
from utils.event_detector import videoscan

import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=str,
                        help="source file path")
    parser.add_argument("output", type=str,
                        help="output file path")
    parser.add_argument("-m", "--model", type=str,
                        help="model file path", default="models/default.hdf5")
    parser.add_argument("-a", "--annotation", type=str,
                        help="annotation file path")
    parser.add_argument("-t", "--time", type=int,
                        help="highlight duration in seconds")
    parser.add_argument("-r", "--rate", type=int,
                        help="inference sampling rate", default=75)
    args = parser.parse_args()

    source_path = args.source
    output_path = args.output
    model_path = args.model
    rate = args.rate
    annotation = args.annotation

    if not args.annotation:
        out = videoscan(source_path, model_path, rate)
    else:
        import numpy as np
        out = np.load(annotation)

    # Set class labels of interest and thresholds for inclusion in a summary
    cls = [1, 2]
    thresh = [0.9, 0.95]

    anno = Annotator(out, cls, thresh)

    # Extract FPS information
    anno.get_fps(source_path)

    # Extract event timestamps using classes and thresholds applied above
    anno.extract_timestamp()

    # generate a summary video
    output_folder = os.path.dirname(output_path)

    if not os.path.exists(output_folder) and output_folder:
        os.mkdir(output_folder)
        print("Made directory : " + output_folder)

    anno.summarize(source_path, output_path)

    print("Saved output to path : " + output_path)
