
# %% 
from keras.utils import Sequence
from keras.preprocessing import image
import os
import glob
import numpy as np
import tensorflow as tf
import math
import util

INPUT_DIR = '/bmrNAS/people/yuxinh/phase_net'

class ImageNetDataGenerator(Sequence):
    """
    Add Phase shifting to Image Net and generate data on the fly
    """
    def __init__(self, input_images, batch_size=128, dim=(256, 256), shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.input_images = input_images
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
        # print("__data_augmentation done generate data")

        return X, y

    def __data_augmentation(self, input_images_temp):
        'Returns augmented data with batch_size images' 
        # Initialization
        X = np.zeros(1)
        y = np.zeros(1)

        # Generate data
        for ii in range(len(input_images_temp)):
            # load image
            # input_img = cv2.imread(input_images_temp[ii])
            img = image.load_img(input_images_temp[ii], target_size=(256, 256))
            input_img = np.array(img)
            # iterate through three channels of RGB
            for jj in range(input_img.shape[2]):
                target = input_img[:, :, jj]
                target = target / 1.0 / np.max(np.max(target, axis=0),axis=0)
                gt_image, skspace = subsample(addphase(target, 4), \
                    combine_output = True, noise_level = 0.0)
                gt_image = np.expand_dims(gt_image, axis=0)
                skspace = np.expand_dims(skspace, axis=0)
                # Store class
                if X.any():
                    X = np.concatenate((X, skspace), axis=0)
                else:
                    X = skspace
                if y.any():
                    y = np.concatenate((y, gt_image), axis=0)
                else:
                    y = gt_image
        # X = np.expand_dims(X, axis=3)
        # y = np.expand_dims(y, axis=3)
        # print(X.shape, y.shape)

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

def load_test_images(test_images_data_fp, n):
    X = np.zeros(1)
    y = np.zeros(1)
    for ii in range(n):
        if ii < len(test_images_data_fp):
            img = image.load_img(test_images_data_fp[ii], target_size=(256, 256))
            input_img = np.array(img)
            # iterate through three channels of RGB
            for jj in range(input_img.shape[2]):
                target = input_img[:, :, jj]
                target = target / 1.0 / np.max(np.max(target, axis=0),axis=0)
                gt_image, skspace = subsample(addphase(target, 4), \
                    combine_output = True, noise_level = 0.0)
                gt_image = np.expand_dims(gt_image, axis=0)
                skspace = np.expand_dims(skspace, axis=0)
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
    ph = np.random.uniform(low = 0, high = 2)
    px = np.linspace(-2, 2, num = nx)
    py = np.linspace(-2, 2, num = ny)
    p = px.reshape(nx,1) * py.reshape(1,ny) + ph
    if noise_level > 0:
        p += noise_level * np.random.normal(loc = 0, scale = 1.0, size = (nx, ny))
    return p

def addphase(image, nshot, noise_level = 0.0):
    "ground truth in image domain"
    nx, ny = image.shape[0], image.shape[1]
    gt_image = np.zeros(shape = (nx,ny,nshot), dtype = 'complex64')
    for i in range(nshot):
        phase = phase_simulation(nx,ny, noise_level)
        gt_image[:,:,i] = image * np.exp(1j * 2 * math.pi * phase)
    return gt_image

def subsample(image, combine_output = True, noise_level = 0.0):
    "input as skspace in k-space"
    nx, ny, nshot = image.shape
    gt_image = image
    if noise_level > 0.0:
        image += noise_level * np.random.normal(loc = 0, scale = 1.0, size = (nx, ny, nshot))
    # convert np image to tf for fft2c
    kspace = np_fft2c(image) # nbatch-nx-ny-nshot/1
    # converting back to np
    skspace = np.zeros(shape = (nx,ny,nshot), dtype = 'complex64')
    for i in range(nshot):
        skspace[:,i:-1:nshot,i] = kspace[:,i:-1:nshot,i]
    if combine_output:
        skspace = np.sum(skspace, axis = -1, keepdims=True)
    return gt_image, skspace

def np_fftshift(im, axis=0):

    split0, split1 = np.split(im, 2, axis=axis)
    output = np.concatenate((split1, split0), axis=axis)
    return output

def np_fft2c(im, do_orthonorm=True):
    im_out = im
    length = len(im.shape)

    im_out = np_fftshift(im_out, axis = 1)
    im_out = np_fftshift(im_out, axis = 0 if length <= 3 else 2)

    if do_orthonorm:
        fftscale = np.sqrt(1.0 * im_out.shape[-1] * im_out.shape[-2])
    else:
        fftscale = 1.0

    if length <= 3:
        im_out = np.fft.fft2(im_out, axes = (0,1)) / fftscale
    else:
        im_out = np.fft.fft2(im_out, axes = (1,2)) / fftscale

    im_out = np_fftshift(im_out, axis = 1)
    im_out = np_fftshift(im_out, axis = 0 if length <= 3 else 2)
    return im_out

def np_ifft2c(im, do_orthonorm=True):
    im_out = im
    length = len(im.shape)

    im_out = np_fftshift(im_out, axis = 1)
    im_out = np_fftshift(im_out, axis = 0 if length <= 3 else 2)

    if do_orthonorm:
        fftscale = np.sqrt(1.0 * im_out.shape[-1] * im_out.shape[-2])
    else:
        fftscale = 1.0

    if length <= 3:
        im_out = np.fft.ifft2(im_out, axes = (0,1)) * fftscale
    else:
        im_out = np.fft.ifft2(im_out, axes = (1,2)) * fftscale

    im_out = np_fftshift(im_out, axis = 1)
    im_out = np_fftshift(im_out, axis = 0 if length <= 3 else 2)
    return im_out