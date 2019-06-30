This folder contains various utility functions supporting this framework.

annotator.py : contains the Annotator class, used for generating timestamps from inferences.
event_detector.py: contains videoscan function, which generates inferences from videos.

video_generator.py : contains the VideoGenerator class, used as a file generator
                     and data augmentation tool for model training in Keras.

video_writer.py : a collection of functions used for creating video overlay. Can be
                  used to visualize intermediate layer activations over the original
                  video clip.

clip_parser.py : a collection of functions used for extracting labeled clips
                 from SoccerNet dataset.

timer.py : contains the PrintTime class, used to measure time at the end of each inference.
