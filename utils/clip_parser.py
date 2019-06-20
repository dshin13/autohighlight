from moviepy.editor import *
import os
import json

# Code assumes that each valid directory contains two video files and a label file

# Code to scan directories and return a list of directories containing labels and videos
def scan_directories(data_dir='./data'):
    
    root = os.walk(data_dir)

    print('Scanning for files...')

    output = []

    for directory in root:

        files = directory[2]

        if '1.mkv' in files and '2.mkv' in files and 'Labels.json' in files:
            output.append(directory[0])

    print('done')
    
    return output


# helper code to parse timestap contained in 'Labels.json'
def parseTimestamp(gameTime):
    
    half, time = gameTime.split(' - ')
    
    min, sec = time.split(':')
    
    time_float = (float(min)*60 + float(sec)) // 1
    
    return (half, time_float)

def parseLabels(labels):
       
    timestamp = [(parseTimestamp(e['gameTime']), e['label']) for e in labels['annotations']]
                
    return timestamp

def clip_video(vid_name, start_pos=0, duration=20, i=0, output_dir='./clips'):
    
    video = VideoFileClip(vid_name).subclip(start_pos, start_pos + duration)
    video.write_videofile(os.path.join(output_dir, str(i) + '.mkv'),
                          codec='libx264',
                          verbose=None)

def event_overlap(labels, half, timestamp, window):
    for l, _ in labels:
        if l[0] == half:
            ceil = l[1] + window//2
            floor = l[1] - window//2
            if timestamp <= ceil and timestamp >= floor:
                return True
    return False
    
def generate_clips(input_dir, output_dir, duration=20):
    
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
                vid_name = os.path.join(path, half + '.mkv')
                
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
    
    output_dir = './data/split_clips'
    directories = scan_directories('./data')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        print('Made directory ' + output_dir)
    generate_clips(directories, output_dir, 10)
