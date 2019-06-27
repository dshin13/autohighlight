import glob
import random
import os
import cv2
import numpy as np

# Based on generator template from https://github.com/jphdotam/keras_generator_example

class VideoGenerator:
    """ 
    A class used to generate videos for training, validation and testing in Keras.
    Implements data augmentation using temporal crop, horizontal flip and spatial crop.
    ...

    Usage
    -----
    

    """

    def __init__(self, train_dir, val_dir, dims, batch_size=2, shuffle=True, file_ext=".mkv"):
        """ 
        Parameters
        ----------
        train_dir : str
            Directory where training data is stored
        val_dir : str
            Directory where validation/test data is stored
        dims : tuple (int, int, int, int)
            Dimensions of input data
        batch_size : int
            Batch size (number of examples per yield)
        shuffle : bool
            If set to true, shuffles data
        file_ext : str
            File extension for data (set to .mkv by default)
        """

        self.train_dir = train_dir
        self.val_dir = val_dir
        self.frames, self.height, self.width, self.channels = dims
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.file_ext = file_ext

        self.filenames_train = self.get_filenames(train_dir)
        if self.val_dir:
            self.filenames_val = self.get_filenames(val_dir)

        self.classname_by_id = {i: cls for i, cls in
                                enumerate(sorted({os.path.basename(os.path.dirname(file)) for file in self.filenames_train}, reverse=True))}
        self.id_by_classname = {cls: i for i, cls in self.classname_by_id.items()}

        self.n_classes = len(self.classname_by_id)
        assert self.n_classes == len(
            self.id_by_classname), "Number of unique classes for training set isn't equal to validation set"

    def get_filenames(self, dir):
        """Returns a list of paths for filenames ending with self.file_ext

        Parameters
        ----------
        dir : str
            Parent directory to search for the files
        """

        filenames = glob.glob(os.path.join(dir, f"**/*{self.file_ext}"))
        return filenames
    
    # Extracts frames from clip using OpenCV
    def vid2npy(self, filename, num_frames=250, random_start=False, resize=False):
        """Returns a numpy array of size (num_frames, self.width, self.height, channels)

        Parameters
        ----------
        filename : str
            File path to video
        num_frames : int
            Number of frames to be sampled
            (relevant if temporal augmentation is being performed during training)
        random_start : bool
            If True, uses randomized starting frame for temporal augmentation

        Returns
        -------
            numpy.array : video file with all pixel values normalized to range [-1,1]
        """

        cap = cv2.VideoCapture(filename)
        frames =[]
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        
        if random_start:
            start_frame = random.randint(0,total_frames - num_frames - 1)
            cap.set(1, start_frame)
        
        for _ in range(num_frames):
            ret, frame = cap.read()
            
            if ret == True:
                if resize:
                    frame = cv2.resize(frame, (self.width,self.height), interpolation=cv2.INTER_AREA)
                frames.append(frame)
                
            else:
                break
        
        output = np.array(frames)
        output = (output / 255) * 2 - 1
        
        # zero-pad frames if not enough frames
        if output.shape[0] < num_frames:
            output = np.concatenate([output, np.zeros((num_frames - output.shape[0], output.shape[1], output.shape[2], output.shape[3]))])
                
        return output

    def generate(self, train_or_val, horizontal_flip=False, random_crop=False, random_start=False):
        """Generator function
        Passes parameters to the helper function _generate_data_from_batch_file_names

        Parameters
        ----------

        train_or_val : str
            Flag to specify whether generator is for training.
            'train' enables data shuffling and augmentation, while 'val' disables both.
        horizontal_flip : bool
            Flag to enable/disable horizontal flip during augmentation
        random_crop : bool
            Flag to enable spatial crop during augmentation
        random_start : bool
            Flag to enable temporal crop during augmentation

        Returns
        -------
            Generator : yields video frame stack (numpy array) and class labels
        """

        if train_or_val == 'train':
            dir = self.train_dir
        elif train_or_val == 'val':
            dir = self.val_dir
        else:
            raise ValueError

        while True:
            filenames = self.get_filenames(dir)
            if self.shuffle and train_or_val == 'train': # shuffle flag when run on training set
                random.shuffle(filenames)

            n_batches = int(len(filenames) / self.batch_size)

            for i in range(n_batches):
                # print(f"Slicing {i*self.batch_size}:{(i+1)*self.batch_size}")
                filenames_batch = filenames[i * self.batch_size:(i + 1) * self.batch_size]
                x, y = self._generate_data_from_batch_file_names(filenames_batch,
                                                                 train_or_val,
                                                                 horizontal_flip,
                                                                 random_crop,
                                                                 random_start)
                yield x, y

    def _generate_data_from_batch_file_names(self, filenames_batch, train_or_val, horizontal_flip, random_crop, random_start):
        """Generator helper function
        Retrieves frame stack from video source and performs data augmentation based on parameters
        passed from generate function. Returns a single batch of data and labels.

        Parameters
        ----------
            filenames_batch : list
                Contains file paths for video clips in the current batch
            train_or_val : str
                Flag to toggle processes useful during training, including shuffling and data augmentation
            horizontal_flip : bool
                Toggles data augmentation via horizontal flip
            random_crop : bool
                Toggles data augmentation via spatial crop
            random_start: bool
                Toggles data augmentation via temporal crop

        Returns
        -------
            numpy.array : Array of frame stacks of size (batch size x num_frames x H x W x channels)
            numpy.array : Array of one-hot encoded class labels (len(integers) x self.n_classes)
        """

        data = []
        labels = []

        for i, filename in enumerate(filenames_batch):
            # Data augmentation implements horizontal flip and random crop (temporal and spatial)
            if train_or_val == 'train':
                npy = self.vid2npy(filename, num_frames=self.frames, random_start=random_start)

                if horizontal_flip:
                    flip = random.random()
                    if flip > 0.5:
                        npy = np.fliplr(npy)
                if random_crop:
                    horizontal_crop = random.randint(0, npy.shape[2] - npy.shape[1])
                    npy = npy[:,:,horizontal_crop:horizontal_crop + npy.shape[1],:]
                    assert npy.shape[1] == npy.shape[2]

            # Center crop all validation clips
            elif train_or_val == 'val':
                npy = self.vid2npy(filename, num_frames=self.frames, random_start=False)

                horizontal_crop = (npy.shape[2] - npy.shape[1])//2
                npy = npy[:,:,horizontal_crop:horizontal_crop + npy.shape[1],:]

            if len(npy.shape) == 3:  # Add color channel to B&W images
                npy = np.expand_dims(npy, axis=-1)

            data.append(npy)
            label = os.path.basename(os.path.dirname(filename))
            labels.append(self.id_by_classname[label])

        x = np.stack(data)
        y = self.list_of_integers_to_2d_onehots(labels)
        return x, y



    def list_of_integers_to_2d_onehots(self, integers):
        """Function to generate one-hot encoding of class labels

        Parameters
        ----------
        integers : list
            List of class labels (e.g. [0, 1, 1, 0, 2, 1, 0, 1, 0])

        Returns
        -------
            numpy.array : an array of dimension (len(integers) x self.n_classes)
        """

        array = [[1 if integers[sample] == cls else 0 for cls in range(self.n_classes)] for sample in range(len(integers))]
        return np.array(array)
