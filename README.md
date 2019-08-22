# AutoHighlight

[![Build Status](https://api.travis-ci.org/dshin13/autohighlight.svg?branch=master)](https://travis-ci.org/dshin13/autohighlight)

AutoHighlight is an automatic video summarizer utilizing action recognition via 3D convolutional neural networks.

  - AutoHighlight utilizes Inflated 3D Convolution architecture from [DeepMind](I3D) to classify video snippets (architecture obtained from [this Keras implementation](I3D_keras)).
  - The example weight included in the repo is trained on association football (soccer) matches using the [SoccerNet dataset](SoccerNet-paper) (GitHub repo [here](SoccerNet)).

## Features

  - Train your own AI video summarizer using your own dataset!
  - Includes a pre-trained network that you can use for summarizing soccer matches!

## Dependencies

AutoHighlight was developed in Python 3.6.5 on AWS Deep Learning AMI (Ubuntu) Version 23.0.
It uses the following open source packages:

* [Keras] - 2.2.4
* [Tensorflow] - 1.13.1
* [OpenCV] - 3.4.2
* [Numpy] - 1.15.4
* [MoviePy] - 0.2.3.5
* [imageio] - 2.5.0
* [Scikit-learn] (for evaluation only) - 

AutoHighlight is open source with a [public repository](git-repo-url)
 on GitHub.

## Installation and usage

AutoHighlight requires Python 3 to run.

Install the dependencies and clone AutoHighlight from GitHub repo.

```sh
$ git clone https://github.com/dshin13/autohighlight.git
```

### Making inference on soccer videos

To run an inference on a video file, navigate to AutoHighlight directory and use the following command:

```sh
$ python autohighlight.py <videopath> <output>.mp4
```

To create videos from pre-existing annotation files, use the following command:

```sh
$ python autohighlight.py <videopath> <output>.mp4 -a <annotation path>
```

If you have video files to annotate, use the following command:

```sh
$ python videoscan.py <parent directory>
```

This will create filename_pred.npy for every video file (.mkv) in the same folder as the video.


### Using model visualizer tool for I3D activation

To create overlay video clips showing activations from specific I3D blocks, use the following command:

```sh
$ python activation_visualizer.py <videopath> -b <block index to probe (0-8)> 
```

This will create a file named overlay.mp4 in your current working directory.


### Building your own model: follow these steps!

To generate class-labeled video clips, use the following command from the home directory:

```sh
$ python utils/clip_parser.py <source directory> <target directory>
```

Please refer to clip_parser.py docstring to define an appropriate filter function first.

Afterwards, generate train/val/test split using the following command:

```sh
$ python train_test_split.py <source directory> <target directory>
```

To define a custom RGB-stream I3D classifier for N classes, use the following command:

```sh
$ python models/build_RGB_model.py <num_frames> <num_width> <num_height> N
```

The model can be trained by using the following command:

```sh
$ python train.py <training set directory> <validation set directory>
```

Model training parameters and optimizer definitions can be modified as necessary inside train.py.

To use your own model to run an inference on a video file, use the following command:

```sh
$ python autohighlight.py -s <videopath> -o <output> -m <your model>
```

### Todo

 - Write tests

License
----
Code and contents of this repository are released under MIT License.

Kinetics-pretrained weights were released by DeepMind under Apache 2.0 License.

   [git-repo-url]: <https://github.com/dshin13/autohighlight.git>
   [SoccerNet-paper]: <https://arxiv.org/abs/1804.04527>
   [SoccerNet]: <https://github.com/SilvioGiancola/SoccerNet-code>
   [I3D]: <https://arxiv.org/pdf/1705.07750.pdf>
   [I3D_keras]: <https://github.com/dlpbc/keras-kinetics-i3d>
   [keras]: <https://keras.io/>
   [tensorflow]: <https://www.tensorflow.org/>
   [opencv]: <https://opencv.org/>
   [numpy]: <https://www.numpy.org/>   
   [moviepy]: <https://zulko.github.io/moviepy/>   
   [imageio]: <https://imageio.github.io/>
