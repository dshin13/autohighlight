"""Train-test partition script

Expects source directory of the following structure:
(output of utils/clip_parser.py)

   source_dir/class1
             /class2
             /class3
             etc...

Output directory is structured as follows:

   output_dir/train/class1
                   /class2
                   /class3

             /val/class1
                 /class2
                 /class3

             /test/class1
                  /class2
                  /class3

Split is stratified by default.

"""

import random
import shutil
import os
import argparse

def train_test_split(src, dest, classes=None, split_ratio=[0.8, 0.1, 0.1]):
    """Function to create train/val/test split from parsed examples.
    Expects root directory to contain folders with class labels, each containing examples of
    corresponding class.

    Folders named 'train', 'val' and 'test' are created in the directory specified by 'root'.
    Classes will be organized into separate folders under each directory.

    Parameters
    ----------
    src : str
        Name of parent directory in which all examples are contained
    dest : str
        Name of directory to which train/val/test sets will be saved
    classes : list
        A list of strings corresponding to folder names under the parent
        directory, to be used as class labels
    split_ratio : list
        A list of floats, representing the fraction of train, validation
        and test sets to be split from the source files
    """

    # if no classes specified, infer classes from all folders in source directory
    if not classes:
         classes = os.listdir(src)

    # make directories
    train_dir = os.path.join(dest, 'train')
    val_dir = os.path.join(dest, 'val')
    test_dir = os.path.join(dest, 'test')

    if not os.path.exists(train_dir):
        print('Creating directory: ' + train_dir)
        os.mkdir(train_dir)

    if not os.path.exists(val_dir):
        print('Creating directory: ' + val_dir)
        os.mkdir(val_dir)

    if not os.path.exists(test_dir):
        print('Creating directory: ' + test_dir)
        os.mkdir(test_dir)

    for cls in classes:
        cls_dir = os.path.join(src, cls)
        print('Accessing files in ' + cls_dir)
        cls_list = os.listdir(cls_dir)
        assert len(cls_list) > 0
        
        random.shuffle(cls_list)
        split_train = round(len(cls_list) * split_ratio[0])
        split_train = max(0, split_train)
        
        split_test = len(cls_list) - round(len(cls_list) * split_ratio[2])
        split_test = min(split_test, len(cls_list) - 1)
        
        cls_train_dir = os.path.join(train_dir, cls)
        cls_val_dir = os.path.join(val_dir, cls)
        cls_test_dir = os.path.join(test_dir, cls)
        
        if not os.path.exists(cls_train_dir):
            print('Creating directory: ' + cls_train_dir)
            os.mkdir(cls_train_dir)

        if not os.path.exists(cls_val_dir):
            print('Creating directory: ' + cls_val_dir)
            os.mkdir(cls_val_dir)

        if not os.path.exists(cls_test_dir):
            print('Creating directory: ' + cls_test_dir)
            os.mkdir(cls_test_dir)
        
        cls_train_set = cls_list[:split_train]
        cls_val_set = cls_list[split_train:split_test]
        cls_test_set = cls_list[split_test:]
        
        for dir in cls_train_set:
            shutil.copy2(os.path.join(cls_dir, dir), cls_train_dir)

        for dir in cls_val_set:
            shutil.copy2(os.path.join(cls_dir, dir), cls_val_dir)
            
        for dir in cls_test_set:
            shutil.copy2(os.path.join(cls_dir, dir), cls_test_dir)
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir",
                        help="source directory for video clips")
    parser.add_argument("output_dir",
                        help="output directory for split files")
    parser.add_argument("-c" "--classes",
                        help="classes for extraction")
    parser.add_argument("-r", "--ratio", default='0.85,0.1,0.05',
                        help="dataset split ratio")
    args = parser.parse_args()

    source_dir = args.source_dir
    output_dir = args.output_dir

    if not os.path.exists(output_dir):
        print('Creating directory: ' + output_dir)
        os.mkdir(output_dir)

    if 'classes' in args.__dict__.keys():
        classes = args.classes.split(',')
    else:
        classes = None
    split_ratio = list(map(float, args.ratio.split(',')))

    # classes = ['goals', 'cards', 'subs', 'bg']
    # split_ratio = [0.85, 0.1, 0.05]

    train_test_split(src=source_dir,
                 dest=output_dir,
                 classes=classes,
                 split_ratio=split_ratio
                )
    
    print('Done')
