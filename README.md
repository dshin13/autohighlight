# AutoHighlight

[![Build Status](https://api.travis-ci.org/dshin13/autohighlight.svg?branch=master)](https://travis-ci.org/dshin13/autohighlight)

AutoHighlight is an automatic video summarizer utilizing action recognition via 3D convolutional neural networks.

  - AutoHighlight utilizes Inflated 3D Convolution architecture from [DeepMind](I3D) to classify video snippets (architecture obtained from [this Keras implementation](I3D_keras)).
  - The example weight included in the repo is trained on association football (soccer) matches using the [SoccerNet dataset](SoccerNet-paper) (GitHub repo [here](SoccerNet)).

# Features

  - Train your own AI video summarizer using your own dataset!
  - Includes a pre-trained network that you can use for summarizing soccer matches!

## Dependencies

AutoHighlight is developed in Python 3.6.5 and uses a number of open source packages:

* [Keras] - 2.2.4
* [Tensorflow] - 1.13.1
* [OpenCV] - 3.4.2
* [Numpy] - 1.15.4
* [MoviePy] - 0.2.3.5

AutoHighlight is open source with a [public repository](git-repo-url)
 on GitHub.

### Installation and usage

AutoHighlight requires Python 3 to run.

Install the dependencies and clone AutoHighlight from GitHub repo.

```sh
$ git clone https://github.com/dshin13/autohighlight.git
```

To run an inference on a video file, navigate to AutoHighlight directory and use the following command:

```sh
$ python autohighlight.py -s <videopath> -o <output>
```

If you have video files to annotate, use the following command:
```sh
$ python videoscan.py -s <parent directory>
```
This will create <filename>_pred.npy for every video file (.mkv) in the same folder as the video.

### Building your own model

To generate class-labeled video clips, use the following command from the home directory:
```sh
$ python utils/clip_parser.py <source directory> <target directory>
```
Please refer to clip_parser.py docstring to define an appropriate filter function first.

Afterwards, generate train/val/test split using the following command:
```sh
$ python train_test_split.py <source directory> <target directory>
```

To define a custom RGB-stream I3D classifier, use the following command:
```sh
$ python models/build_RGB_model.py <num_frames> <num_width> <num_height> <num_classes>
```


### Docker
TBD!

### Todos

 - Write tests
 - Write docstring
 - Docker image

License
----

MIT

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
