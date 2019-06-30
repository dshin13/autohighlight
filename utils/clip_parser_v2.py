from moviepy.editor import *

import os
import json
import argparse

# Code assumes that each valid directory contains two video files and a label file

# Code to scan directories and return a list of directories containing labels and videos
def scan_directories(data_dir, file_filter):
    """Function to scan directories for folders containing videos and labels

    Parameters
    ----------
    data_dir : str
        Parent directory to scan for data
    file_filter : function
        Python function which returns a boolean flag based on the contents of a folder,
        presented as a list of strings

    Returns
    -------
        list : a list of directories containing files determined by filter
    """

    root = os.walk(data_dir)

    print('Scanning for files...')

    output = []

    for directory in root:

        files = directory[2]

        # Valid dataset contains video files of both halves and an accompanying label
        if file_filter(files):
            output.append(directory[0])

    print('Done')

    return output


def default_filter(files):
    """Function to filter folders based on content

    Parameters
    ----------
    files : list
        A list containing strings of filenames in directory

    Returns
    -------
        bool : a flag indicating whether the list contains '1.mkv', '2.mkv'
               and 'Labels.json'
    """

    if '1.mkv' in files and '2.mkv' in files and 'Labels.json' in files:
        return True

    return False


# helper code to parse timestap contained in 'Labels.json'
def parseTimestamp(gameTime):
    """Function to parse timestamp contained in 'Labels.json'

    Parameters
    ----------
    gameTime : str
        String containing unparsed information for each stamp in the
        format 'half-MM:SS'

    Returns
    -------
        (str, float) : parsed tuple containing half identifier and event
        timestamp in seconds
    """

    half, time = gameTime.split(' - ')
    
    min, sec = time.split(':')
    
    time_float = (float(min)*60 + float(sec)) // 1
    
    return (half, time_float)

def parseLabels(labels):
    """Function to parse labels from 'Labels.json'
    Uses the helper function parseTimestamp to perform string parsing

    Parameters
    ----------
    labels : dict
        Output of json file read from 'Labels.json'

    Returns
    -------
        list : contains timestamps, class labels and half identifier for every
        annotated event in 'Labels.json'
    """

    timestamp = [(parseTimestamp(e['gameTime']), e['label']) for e in labels['annotations']]
                
    return timestamp

def clip_video(vid_name, start_pos=0, duration=20, i=0, output_dir='./clips'):
    """Function to clip videos using MoviePy module
    Saves a clip of the source video based on the provided parameters

    Parameters
    ----------
    vid_name : str
        Path to the source video file
    start_pos : float
        Starting time of the clip, in seconds
    duration : float
        Duration of the clip, in seconds
    i : int
        Index to be used to generate filename for the clip
    """

    video = VideoFileClip(vid_name).subclip(start_pos, start_pos + duration)
    video.write_videofile(os.path.join(output_dir, str(i) + '.mkv'),
                          codec='libx264',
                          verbose=None)

def event_overlap(labels, half, timestamp, window):
    """Function to check whether a given timestamp overlaps with another event

    Function assumes a fixed and identical time window for all labels, defined by
    the window parameter. Called by generate_clips to help sample valid label-free
    counter-examples for each class example.

    Parameters
    ----------
    labels : list
        A list containing [(half, timestamp), class] as its elements
    half : str
        A string indicating which half of a match the label belongs to
        ('1' or '2')
    timestamp : float
        A float indicating timestamp of the event being evaluated, in seconds
    window : int
        Defines the span of every event in labels, in seconds

    Returns
    -------
        bool : a flag to indicate whether there is an overlap
    """

    for l, _ in labels:
        if l[0] == half:
            ceil = l[1] + window//2
            floor = l[1] - window//2
            if timestamp <= ceil and timestamp >= floor:
                return True
    return False
    
def generate_clips(input_dir, output_dir, duration=20, ext='.mkv'):
    """Function to generate example clips from full-length source and labels
    Creates subdirectories and generates examples of each class using index-based
    naming system. Each class is indexed separately.
    Background class (bg) is sampled 45 seconds upstream of a labeled event,
    provided there is no overlap with another labeled event (checked by
    event_overlap function).

    Parameters
    ----------
    input_dir : list
        A list containing directory names which contain both videos and labels;
        returned from scan_directories function
    output_dir : str
        Destination folder for the newly created samples, organized by class
    duration : float
        Duration of the clips to be generated, in seconds
    """
    
    i = [0,0,0,0]
    
    output_dirs = [os.path.join(output_dir, 'goals'),
                  os.path.join(output_dir, 'bg'),
                  os.path.join(output_dir, 'cards'),
                  os.path.join(output_dir, 'subs')]

    for dir in output_dirs:
        if not os.path.exists(dir):
            os.mkdir(dir)
            print('Made directory ' + dir)
    
    
    for path in input_dir:
        print(path)
        
        with open(os.path.join(path, 'Labels.json')) as f:
            file = json.load(f)
            labels = parseLabels(file)
            
            # for each item in label, crop a specified length and save to output_dir
            for timestamp, label in labels:
                half = timestamp[0]
                time = timestamp[1]
                vid_name = os.path.join(path, half + ext)

                if time - 5 > 0:

                    if label == 'soccer-ball':
                        # collect an instance of a goal
                        clip_video(vid_name, time - 5, duration, i[0], output_dirs[0])
                        i[0] += 1

                        # collect an instance of a non-goal, if it does not overlap with another event
                        if event_overlap(labels, half, time - 45, duration) == False and time - 45 > 0:
                            clip_video(vid_name, time - 45, duration, i[1], output_dirs[1])
                            i[1] += 1

                    elif 'card' in label:
                        # collect an instance of a carding event
                        clip_video(vid_name, time - 4, duration, i[2], output_dirs[2])
                        i[2] += 1

                        if event_overlap(labels, half, time - 45, duration) == False and time - 45 > 0:
                            clip_video(vid_name, time - 45, duration, i[1], output_dirs[1])
                            i[1] += 1


                    elif 'substitution' in label:
                        # collect an instance of a carding event
                        clip_video(vid_name, time - 4, duration, i[3], output_dirs[3])
                        i[3] += 1

                        if event_overlap(labels, half, time - 45, duration) == False and time - 45 > 0:
                            clip_video(vid_name, time - 45, duration, i[1], output_dirs[1])
                            i[1] += 1

            print('Saved clip from ' + path)
            
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir", type=str,
                        help="source directory for videos and labels")
    parser.add_argument("output_dir", type=str,
                        help="output directory for labeled video clips")
    args = parser.parse_args()

    source_dir = args.source_dir
    output_dir = args.output_dir

    directories = scan_directories(source_dir, default_filter)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        print('Made directory ' + output_dir)

    generate_clips(directories, output_dir, 10)
