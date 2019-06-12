import glob
import random
import os
import cv2
import numpy as np

# Using template code from https://github.com/jphdotam/keras_generator_example

class VideoGenerator:

    def __init__(self, train_dir, test_dir, dims, batch_size=2, shuffle=True, file_ext=".mkv"):
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.frames, self.width, self.height, self.channels = dims
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.file_ext = file_ext

        self.filenames_train = self.get_filenames(train_dir)
        if self.test_dir:
            self.filenames_test = self.get_filenames(test_dir)

        self.classname_by_id = {i: cls for i, cls in
                                enumerate({os.path.basename(os.path.dirname(file)) for file in self.filenames_train})}
        self.id_by_classname = {cls: i for i, cls in self.classname_by_id.items()}

        self.n_classes = len(self.classname_by_id)
        assert self.n_classes == len(
            self.id_by_classname), "Number of unique classes for training set isn't equal to testing set"

    def get_filenames(self, dir):
        filenames = glob.glob(os.path.join(dir, f"**/*{self.file_ext}"))
        return filenames
    
    # Extracts frames from clip using OpenCV
    def vid2npy(self, filename):
        cap = cv2.VideoCapture(filename)
        frames =[]
        while True:
            ret, frame = cap.read()
            if ret == True:
                frame = cv2.resize(frame, (self.width,self.height), interpolation=cv2.INTER_AREA)
                frames.append(frame)
            else:
                break
        
        output = np.array(frames)
        output = (output / 255) * 2 - 1
                
        return output

    def generate(self, train_or_test, rotation_range=None, heigt_shift_range=None, width_shift_range=None,
                 shear_range=None, zoom_range=None, horizontal_flip=None, vertical_flip=None, brightness_range=None):

        if train_or_test == 'train':
            dir = self.train_dir
        elif train_or_test == 'test':
            dir = self.test_dir
        else:
            raise ValueError

        while True:
            filenames = self.get_filenames(dir)
            if self.shuffle:
                random.shuffle(filenames)

            n_batches = int(len(filenames) / self.batch_size)

            for i in range(n_batches):
                # print(f"Slicing {i*self.batch_size}:{(i+1)*self.batch_size}")
                filenames_batch = filenames[i * self.batch_size:(i + 1) * self.batch_size]
                x, y = self.__generate_data_frome_batch_file_names(filenames_batch)
                yield x, y

    def __generate_data_frome_batch_file_names(self, filenames_batch):
        data = []
        labels = []

        for i, filename in enumerate(filenames_batch):
            npy = self.vid2npy(filename)
            try:
                npy = npy[npy.files[0]] # If an npz file we need to get the data out using the filename as a key
            except:
                pass

            if len(npy.shape) == 3:  # Add colour channel to B&W images
                npy = np.expand_dims(npy, axis=-1)

            data.append(npy)
            label = os.path.basename(os.path.dirname(filename))
            labels.append(self.id_by_classname[label])

        x = np.stack(data)
        y = self.list_of_integers_to_2d_onehots(labels)
        return x, y



    def list_of_integers_to_2d_onehots(self, integers):
        array = [[1 if integers[sample] == cls else 0 for cls in range(self.n_classes)] for sample in range(len(integers))]
        return np.array(array)