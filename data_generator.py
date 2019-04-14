
# %% 
from keras.utils import Sequence
import os
import glob
import numpy as np
import tensorflow as tf
import cv2
import math
import util

INPUT_DIR = '/bmrNAS/people/yuxinh/deep_bmask'

class ImageNetDataGenerator(Sequence):
    """
    Add Phase shifting to Image Net and generate data on the fly
    """
    def __init__(self, input_images, batch_size=128, dim=(256, 256), n_channels=1, n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.input_images = input_images
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.input_images))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.input_images) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        # Find list of IDs
        input_images_path = [self.input_images[k] for k in indexes]

        # Generate data
        X, y = self.__data_augmentation(input_images_path)

        return X, y

    def __data_augmentation(self, input_images_temp):
        'Returns augmented data with batch_size images' 
        # X : (n_samples, v_size, v_size, v_size, n_channels)
        # Initialization

        # Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.zeros(1)
        y = np.zeros(1)

        # Generate data
        for ii in range(len(input_images_temp)):
            # load image
            input_img = cv2.imread(input_images_temp[ii])
            input_img = input_img[2:402, 2:498, :]
            # iterate through three channels of RGB
            for jj in range(input_img.shape[2]):
                gt_image, skspace = subsample(addphase(input_img[:, :, jj], 4), combine_output = True, noise_level = 0.0)
                # Store class
                if X.any():
                    X = np.concatenate((X, skspace), axis=0)
                else:
                    X = skspace
                if y.any():
                    y = np.concatenate((y, gt_image), axis=0)
                else:
                    y = gt_image

        return X, y

def get_list_image_id():

    dpRoot = setup_path()
    dpDataSet = os.path.join(dpRoot, 'data')
    count = 0

    # %% subjects
    clsidx = sorted(glob.glob(os.path.join(dpDataSet, 'n*')))
    train_images_data_fp = []
    test_images_data_fp = []

    for idx in clsidx:
        subjects = sorted(glob.glob(os.path.join(idx, 'n*.JPEG')))
        for sj in subjects:
            if count % 10 == 0:
                test_images_data_fp.append(sj)
            else:
                train_images_data_fp.append(sj)
            count+=1

    print("# training: " , len(train_images_data_fp))
    print("# testing: " , len(test_images_data_fp))
    return train_images_data_fp, test_images_data_fp

def setup_path():
    """
    setting up the working directory to read data from.
    """
    # %% set up path
    dpRoot = INPUT_DIR
    
    if not os.path.exists(dpRoot):
        # try cwd if we don't have input dir
        dpRoot = os.getcwd()
    
    os.chdir(dpRoot)
    return dpRoot

def phase_simulation(nx,ny,noise_level = 0.0):
    const = tf.random.uniform(shape = (1,1), minval = 0, maxval = 4)
    px = 4 * tf.linspace(-0.5, 0.5, num = nx)
    py = 4 * tf.linspace(-0.5, 0.5, num = ny)
    p = tf.reshape(px, shape = (nx,1)) * tf.reshape(py, shape = (1,ny)) + const
    if noise_level > 0:
        p += noise_level * tf.random.normal(shape = (nx, ny), mean = 0, stddev = 1.0)
    return p

def addphase(image, nshot, noise_level = 0.0):
    "ground truth in image domain"
    nx, ny = image.shape[0], image.shape[1]
    gt_image = tf.zeros(shape = (nx,ny,nshot), dtype = tf.complex64)
    for i in range(nshot):
        gt_image[:,:,i] = image * tf.exp(1j * 2 * math.pi * phase_simulation(nx,ny, noise_level))
    return gt_image

def subsample(image, combine_output = True, noise_level = 0.0):
    "input as skspace in k-space"
    nx, ny, nshot = image.shape
    gt_image = image
    if noise_level > 0.0:
        image += noise_level * tf.random.normal(shape = (nx, ny, nshot), mean = 0, stddev = 1.0)
    kspace = util.fft2c(np.expand_dims(image, axis=0)) # nbatch-nx-ny-nshot/1
    skspace = tf.zeros(shape = (nx,ny,nshot), dtype = tf.complex64)
    for i in range(nshot):
        skspace[:,i:-1:nshot,i] = kspace[:,i:-1:nshot,i]
    if combine_output:
        skspace = tf.reduce_sum(skspace, axis = -1, keep_dims = True)
    return gt_image, skspace

# %%

