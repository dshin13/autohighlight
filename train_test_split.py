import random
import shutil
import os


def train_test_split(root='./data', classes=['goals', 'nongoals'], split_ratio=[0.8, 0.1, 0.1]):
    """Function to create train/val/test split from parsed examples.
    Expects root directory to contain folders with class labels, each containing examples of
    corresponding class.

    Folders named 'train', 'val' and 'test' are created in the directory specified by 'root'.
    Classes will be organized into separate folders under each directory.

    Parameters
    ----------
    root : str
        Name of parent directory in which all examples are contained
    classes : list
        A list of strings corresponding to folder names under the parent
        directory, to be used as class labels
    split_ratio : list
        A list of floats, representing the fraction of train, validation
        and test sets to be split from the source files
    """
    # make train and test directories
    current_dir = os.listdir(root)
    train_dir = os.path.join(root, 'train')
    val_dir = os.path.join(root, 'val')
    test_dir = os.path.join(root, 'test')    
    
    if 'train' not in current_dir:
        os.mkdir(train_dir)

    if 'val' not in current_dir:
        os.mkdir(val_dir)        
        
    if 'test' not in current_dir:
        os.mkdir(test_dir)
    
    train_dir_content = os.listdir(train_dir)
    val_dir_content = os.listdir(val_dir)
    test_dir_content = os.listdir(test_dir)
    
    for cls in classes:
        cls_dir = os.path.join(root, cls)
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
        
        if cls not in train_dir_content:
            os.mkdir(cls_train_dir)

        if cls not in val_dir_content:
            os.mkdir(cls_val_dir)            
            
        if cls not in test_dir_content:
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
    train_test_split(root='./data/split_clips',
                 classes=['goals', 'cards', 'subs', 'bg'],
                 split_ratio = [0.85, 0.1, 0.05]
                )
    
    print('Done')
