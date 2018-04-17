
# coding: utf-8

# # Using cifar 

# import tensorflow as tf

# ## Get familiar with [Cifar](https://www.cs.toronto.edu/~kriz/cifar.html) DataSet 
# - Import data.
# - The struct of cifar dataset.
# 
# 
# Cifar Dataset was collected by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton. The CIFAR-10 dataset consists of 60000 32x32 colour images(RGB) in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. (I'll use CIFAR-10 as my cnn dataset.）
# 
# 10 classes are:
# 
#     [b'airplane', b'automobile', b'bird', b'cat', b'deer', b'dog', b'frog', b'horse', b'ship', b'truck']
# every picture is 32 * 32 * 3(RGB three layers) = 3072 features.

MODEL_IMAGE_SIZE = 24
CIFAR_IMAGE_SIZE = 32
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='latin1')
    return data

def read_data(data_dir,eval_data):
    """Construct input for CIFAR(CIFAR-10 or CIFAR-100)evaluation.
        Read data from `data_dir` (train data or eval_data)
    
    Args:
        data_dir:  string, path to CIFAR data.
        eval_data: bool, indicating if one should use the train or eval data set.
        
    Returns:
        images: Images. [batch_size, 3, IMAGE_SIZE, IMAGE_SIZE] size.
        labels: Labels. [batch_size] size.
        batch_size : int. train or eval data size.
        names:  name of the classes.
    """
    import os 
    names = unpickle(os.path.join(data_dir,"batches.meta"))["label_names"]
    images,labels = [],[]
    batch_size = 0
    if not eval_data:
        files = ["data_batch_{}".format(i) for i in range(1,6)]
        filenames = [os.path.join(data_dir,file) for file in files]
    else:
        filenames = [os.path.join(data_dir,"test_batch")]
    for f in filenames:
        if not os.path.exists(f):
            raise ValueError('Failed to find file: ' + f)
        cifar_ith_data = unpickle(f)
        if batch_size > 0:
            images = np.vstack((images,cifar_ith_data["data"]))
            labels = np.hstack((labels,cifar_ith_data["labels"]))
            batch_size += cifar_ith_data["labels"].__len__()
        else:
            images = cifar_ith_data["data"]
            labels = cifar_ith_data["labels"]
            batch_size += cifar_ith_data["labels"].__len__()
    images = np.reshape(images,[-1,3,CIFAR_IMAGE_SIZE,CIFAR_IMAGE_SIZE])
    return images,labels,batch_size,names

def show_some_examples(names, data, labels,picture):
    plt.figure()
    rows, cols = 4, 4
    random_idxs = random.sample(range(len(data)), rows * cols)
    for i in range(rows * cols):
        plt.subplot(rows, cols, i + 1)
        j = random_idxs[i]
        plt.title(names[labels[j]])
        img = np.transpose(data[j],[1,2,0])
        plt.imshow(img, cmap='Greys_r')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(picture)