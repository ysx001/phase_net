# %%
# """Common functions for setup."""
import tensorflow as tf
import numpy as np
import scipy.signal


def complex_to_channels(image, name="complex2channels"):
    "Convert data from complex to channels."
    #  (!) image: batch-nx-ny-1 
    with tf.name_scope(name):
        image_out = tf.stack([tf.real(image), tf.imag(image)], axis=-1)
        # tf.shape: returns tensor
        # image.shape: returns actual values
        shape_out = tf.concat([tf.shape(image)[:-1], [image.shape[-1]*2]],
                              axis=0)
        image_out = tf.reshape(image_out, shape_out)
    return image_out


def channels_to_complex(image, name="channels2complex"):
    "Convert data from channels to complex."
    with tf.name_scope(name):
        image_out = tf.reshape(image, [-1, 2])
        image_out = tf.complex(image_out[:, 0], image_out[:, 1])
        shape_out = tf.concat([tf.shape(image)[:-1], [image.shape[-1] // 2]],
                              axis=0)
        image_out = tf.reshape(image_out, shape_out)
    return image_out


def fftshift(im, axis=0, name="fftshift"):
    """Perform fft shift.

    This function assumes that the axis to perform fftshift is divisible by 2.
    """
    with tf.name_scope(name):
        split0, split1 = tf.split(im, 2, axis=axis)
        output = tf.concat((split1, split0), axis=axis)
    return output


def ifftc(im, name="ifftc", do_orthonorm=True):
    "Centered iFFT on second to last dimension."
    with tf.name_scope(name):
        im_out = im
        if do_orthonorm:
            fftscale = tf.sqrt(1.0 * im_out.get_shape().as_list()[-2])
        else:
            fftscale = 1.0
        fftscale = tf.cast(fftscale, dtype=tf.complex64)
        if len(im.get_shape()) == 4:
            im_out = tf.transpose(im_out, [0, 3, 1, 2])
            im_out = fftshift(im_out, axis=3)
        else:
            im_out = tf.transpose(im_out, [2, 0, 1])
            im_out = fftshift(im_out, axis=2)
        with tf.device('/gpu:0'):
            # FFT is only supported on the GPU
            im_out = tf.ifft(im_out) * fftscale
        if len(im.get_shape()) == 4:
            im_out = fftshift(im_out, axis=3)
            im_out = tf.transpose(im_out, [0, 2, 3, 1])
        else:
            im_out = fftshift(im_out, axis=2)
            im_out = tf.transpose(im_out, [1, 2, 0])

    return im_out


def fftc(im, name="fftc", do_orthonorm=True):
    "Centered FFT on second to last dimension."
    with tf.name_scope(name):
        im_out = im
        if do_orthonorm:
            fftscale = tf.sqrt(1.0 * im_out.get_shape().as_list()[-2])
        else:
            fftscale = 1.0
        fftscale = tf.cast(fftscale, dtype=tf.complex64)
        if len(im.get_shape()) == 4:
            im_out = tf.transpose(im_out, [0, 3, 1, 2])
            im_out = fftshift(im_out, axis=3)
        else:
            im_out = tf.transpose(im_out, [2, 0, 1])
            im_out = fftshift(im_out, axis=2)
        with tf.device('/gpu:0'):
            im_out = tf.fft(im_out) / fftscale
        if len(im.get_shape()) == 4:
            im_out = fftshift(im_out, axis=3)
            im_out = tf.transpose(im_out, [0, 2, 3, 1])
        else:
            im_out = fftshift(im_out, axis=2)
            im_out = tf.transpose(im_out, [1, 2, 0])

    return im_out

def identity(im, name="identity", do_orthonorm=True):
    "Centered iFFT2."
    with tf.name_scope(name):
        print("in identity")
        print(im.dtype)
        print(name)
        return im

def ifft2c(im, name="ifft2c", do_orthonorm=True):
    "Centered iFFT2."
    with tf.name_scope(name):
        print(name)
        im_out = im
        print(im.dtype)
        print(im_out.dtype)

        if len(im.get_shape()) == 5:
            im_out = tf.transpose(im_out, [0, 3, 4, 1, 2])
            im_out = fftshift(im_out, axis=4)
            im_out = fftshift(im_out, axis=3)
        elif len(im.get_shape()) == 4:
            im_out = tf.transpose(im_out, [0, 3, 1, 2])
            im_out = fftshift(im_out, axis=3)
            im_out = fftshift(im_out, axis=2)
        else:
            im_out = tf.transpose(im_out, [2, 0, 1]) 
            im_out = fftshift(im_out, axis=2)
            im_out = fftshift(im_out, axis=1)

        if do_orthonorm:
            fftscale = tf.sqrt(1.0 * im_out.get_shape().as_list()[-1] * im_out.get_shape().as_list()[-2])
        else:
            fftscale = 1.0
        fftscale = tf.cast(fftscale, dtype=tf.complex64)


        with tf.device('/gpu:0'):
            # FFT is only supported on the GPU
            im_out = tf.ifft2d(im_out) * fftscale

        if len(im.get_shape()) == 5:
            im_out = fftshift(im_out, axis=4)
            im_out = fftshift(im_out, axis=3)
            im_out = tf.transpose(im_out, [0, 3, 4, 1, 2])
        elif len(im.get_shape()) == 4:
            im_out = fftshift(im_out, axis=3)
            im_out = fftshift(im_out, axis=2)
            im_out = tf.transpose(im_out, [0, 2, 3, 1])
        else:
            im_out = fftshift(im_out, axis=2)
            im_out = fftshift(im_out, axis=1)
            im_out = tf.transpose(im_out, [1, 2, 0])

    return im_out

def fft2c(im, name="fft2c", do_orthonorm=True):
    "Centered FFT2."
    with tf.name_scope(name):
        im_out = im

        if len(im.get_shape()) == 5:
            im_out = tf.transpose(im_out, [0, 3, 4, 1, 2])
            im_out = fftshift(im_out, axis=4)
            im_out = fftshift(im_out, axis=3)
        elif len(im.get_shape()) == 4:
            im_out = tf.transpose(im_out, [0, 3, 1, 2])
            im_out = fftshift(im_out, axis=3)
            im_out = fftshift(im_out, axis=2)
        else:
            im_out = tf.transpose(im_out, [2, 0, 1])
            im_out = fftshift(im_out, axis=2)
            im_out = fftshift(im_out, axis=1) 

        if do_orthonorm: # Todo: and im_out.shape[-1] != None
            fftscale = tf.sqrt(1.0 * im_out.get_shape().as_list()[-1]
                               * im_out.get_shape().as_list()[-2])
        else:
            fftscale = 1.0
        fftscale = tf.cast(fftscale, dtype=tf.complex64)

        with tf.device('/gpu:0'):
            im_out = tf.fft2d(im_out) / fftscale

        if len(im.get_shape()) == 5:
            im_out = fftshift(im_out, axis=4)
            im_out = fftshift(im_out, axis=3)
            im_out = tf.transpose(im_out, [0, 3, 4, 1, 2])
        elif len(im.get_shape()) == 4:
            im_out = fftshift(im_out, axis=3)
            im_out = fftshift(im_out, axis=2)
            im_out = tf.transpose(im_out, [0, 2, 3, 1])
        else:
            im_out = fftshift(im_out, axis=2)
            im_out = fftshift(im_out, axis=1)
            im_out = tf.transpose(im_out, [1, 2, 0])

    return im_out


def sumofsq(image_in, keep_dims=False, axis=-1, name="sumofsq"):
    "Compute square root of sum of squares."
    with tf.variable_scope(name):
        image_out = tf.square(tf.abs(image_in))
        image_out = tf.reduce_sum(image_out, keep_dims=keep_dims,
                                  axis=axis)
        image_out = tf.sqrt(image_out)
    return image_out


def replace_kspace(image_orig, image_cur, name="replace_kspace"):
    "Replace k-space with known values."
    with tf.variable_scope(name):
        mask_x = kspace_mask(image_orig)
        image_out = tf.add(tf.multiply(mask_x, image_orig),
                           tf.multiply((1 - mask_x), image_cur))

    return image_out


def kspace_mask(ksp, name="kspace_mask", dtype=None):
    "Find k-space mask."
    with tf.variable_scope(name):
        mask_x = tf.not_equal(ksp, 0)
        if dtype is not None:
            mask_x = tf.cast(mask_x, dtype=dtype)
    return mask_x


def kspace_threshhold(image_orig, threshhold=1e-8, name="kspace_threshhold"):
    """Find k-space mask based on threshhold.

    Anything less the specified threshhold is set to 0.
    Anything above the specified threshhold is set to 1.
    """
    with tf.variable_scope(name):
        mask_x = tf.greater(tf.abs(image_orig), threshhold)
        mask_x = tf.cast(mask_x, dtype=tf.float32)
    return mask_x