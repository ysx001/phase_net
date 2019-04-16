# %%
from keras.models import Model
from keras.layers import Input, Conv2D, concatenate, MaxPooling2D, UpSampling2D, BatchNormalization, Activation, Flatten, Dense, Reshape, Conv2DTranspose, Dropout, ZeroPadding2D, RepeatVector, Lambda
from keras.utils import plot_model
from keras.regularizers import l2
from keras.layers.merge import add
from keras import losses
import keras.backend as K
import tensorflow as tf
import util

# Custom layers import
# from ourlayers import (CropLayer2D, NdSoftmax, DePool2D)

ROW_AXIS = 1
COL_AXIS = 2
CHANNEL_AXIS = 3

############################################################
#  Utility Functions
############################################################

def conv_bn_relu(inputs, n_filters, kernel_size, bn_layer=False):
    conv = Conv2D(filters = n_filters, kernel_size=kernel_size, padding='same', kernel_initializer="he_normal")(inputs)
    if bn_layer:
        conv = BatchNormalization(axis=3)(conv)
    act = Activation('relu')(conv)
    return act

def shortcut(input, residual):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    stride_width = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
    stride_height = int(round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))
    equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Conv2D(filters=residual_shape[CHANNEL_AXIS],
                          kernel_size=(1, 1),
                          strides=(stride_width, stride_height),
                          padding="valid",
                          kernel_initializer="he_normal",
                          kernel_regularizer=l2(0.0001))(input)

    return add([shortcut, residual])

def build_down_conv_block(input_block, depth, kernel_size=3, repeat = 3, bn_layer = False, residual = True, last_layer = False):
    # residual block (using proposed method: https://arxiv.org/abs/1603.05027)
    # first block of downsampling
    cbr = conv_bn_relu(input_block, depth, kernel_size, bn_layer) # conv-bn-relu
    for ii in range(repeat - 1):
        cbr = conv_bn_relu(cbr, depth, kernel_size, bn_layer) # conv-bn-relu

    if residual:
        res = Conv2D(depth, kernel_size=kernel_size, padding='same')(cbr) # residual => full activation
        skip = shortcut(input_block, res) # save for skip connection
    else:
        skip = cbr
    
    if last_layer:
        next_input = cbr
    else:
        # we do max pooling only when it's not the last layer
        next_input = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(skip) # input for next iteration

    return skip, next_input

# residual block (using proposed method: https://arxiv.org/abs/1603.05027)
def build_up_conv_block(input_block, skip_block, depth, bn_layer = False, kernel_size=3, repeat = 3, residual = True):
    # up-conv 2x2
    up = UpSampling2D(size=(2, 2))(input_block)
    up = Conv2D(depth, kernel_size=kernel_size, padding='same')(up)
    # up = conv_bn_relu(up, depth, kernel_size, bn_layer)
    
    # concate
    inputs = concatenate([up, skip_block], axis=3)

    # repeat convs
    cbr = conv_bn_relu(inputs, depth, kernel_size, bn_layer) # conv-bn-relu
    for ii in range(repeat - 1):
        cbr = conv_bn_relu(cbr, depth, kernel_size, bn_layer) # conv-bn-relu
    
    if residual:
        # do a conv for res input
        conv_inputs = Conv2D(filters = depth, kernel_size=kernel_size, padding='same', kernel_initializer="he_normal")(inputs)
        res = Conv2D(depth, kernel_size=kernel_size, padding='same')(cbr) # residual => full activation
        return shortcut(conv_inputs, res)
    
    return cbr

def build_basic_unet(input_shape = (256, 256, 1), depths = [16, 32, 64, 128], n_shot = 4, kernel_size = 3, bn_layer=False, name = "basic_unet"):
    # Build encoder
    encoder_sc_depth_dict = {}
    inputs = Input(shape=input_shape, name='encoder_input')
    next_encoder_input = Conv2D(filters = depths[0], kernel_size=kernel_size, padding='same', kernel_initializer="he_normal")(inputs)

    # encoder
    for ii in range(len(depths)):
        isLast = (ii == (len(depths) - 1))
        encoder_sc_depth_dict[depths[ii]], next_encoder_input = build_down_conv_block(next_encoder_input, depths[ii], residual = True, last_layer = isLast, bn_layer=bn_layer)

    # next_decoder_input = Dropout(0.5)(next_encoder_input)
    next_decoder_input = next_encoder_input

    # decoder
    for jj in range(len(depths)-2, -1, -1): # doing backward iteration, stupid python
        depth = depths[jj]
        next_decoder_input = build_up_conv_block(next_decoder_input, encoder_sc_depth_dict[depth], depth, bn_layer=bn_layer)

    decoded = Conv2D(filters = n_shot * 2, kernel_size=kernel_size, padding='same', kernel_initializer="he_normal")(next_decoder_input)

    unet = Model(inputs, decoded, name=name)
        
    return unet

def build_unet_k_domain(input_shape = (256, 256, 1), depths = [16, 32, 64, 128], n_shot = 4, kernel_size = 3, ifft_flag = False, bn_layer=False):
    # convert complex to channels
    inputs = Input(shape=input_shape, name='encoder_input', dtype='complex64')
    channel_inputs = Lambda(util.complex_to_channels, name="complex_to_channels")(inputs)

    # basic unet
    basic_unet_input_shape = K.int_shape(channel_inputs)[1:]
    basic_unet = build_basic_unet(input_shape = basic_unet_input_shape, depths = depths, n_shot = n_shot, kernel_size = kernel_size, bn_layer=bn_layer)
    decoded = basic_unet(channel_inputs)

    # convert channels back to complex and inverse fft to image domain
    ifft_input = Lambda(util.channels_to_complex, name="channels_to_complex", dtype='complex64')(decoded)
    # explictly pass in output_shape to avoid this being run twice to infer output shape
    output = Lambda(util.ifft2c, output_shape=(input_shape[0], input_shape[1], n_shot), name="iff2c_layer", dtype='complex64')(ifft_input)

    k_unet = Model(inputs, output, name='k_unet')

    return k_unet

def build_unet_i_domain(input_shape = (256, 256, 1), depths = [16, 32, 64, 128], n_shot = 4, kernel_size = 3, ifft_flag = False, bn_layer=False):
    # ifft2c to image domain and convert complex to channels
    inputs = Input(shape=input_shape, name='encoder_input', dtype='complex64')
    channel_inputs = Lambda(util.ifft2c, output_shape=input_shape, name="iff2c_layer", dtype='complex64')(inputs)
    channel_inputs = Lambda(util.complex_to_channels, name="complex_to_channels")(channel_inputs)

    basic_unet_input_shape = K.int_shape(channel_inputs)[1:]
    basic_unet = build_basic_unet(input_shape = basic_unet_input_shape, depths = depths, n_shot = n_shot, kernel_size = kernel_size, bn_layer=bn_layer)
    decoded = basic_unet(channel_inputs)

    # inverse fft
    output = Lambda(util.channels_to_complex, name="channels_to_complex", dtype='complex64')(decoded)
    i_unet = Model(inputs, output, name='i_unet')

    return i_unet

def build_unet_k_i_domain(input_shape = (256, 256, 1), depths = [16, 32, 64, 128], n_shot = 4, kernel_size = 3, ifft_flag = False, bn_layer=False):
    # convert complex to channels
    inputs = Input(shape=input_shape, name='encoder_input', dtype='complex64')
    channel_inputs = Lambda(util.complex_to_channels, name="complex_to_channels_1")(inputs)

    # basic unet i in k-domain
    basic_unet_input_shape = K.int_shape(channel_inputs)[1:]
    basic_unet = build_basic_unet(input_shape = basic_unet_input_shape, depths = depths, n_shot = n_shot, kernel_size = kernel_size, bn_layer=bn_layer, name="k_basic_unet")
    decoded = basic_unet(channel_inputs)

    # inverse fft
    ifft_input = Lambda(util.channels_to_complex, name="channels_to_complex_1", dtype='complex64')(decoded)
    complex_image = Lambda(util.ifft2c, output_shape=(input_shape[0], input_shape[1], n_shot), name="iff2c_layer", dtype='complex64')(ifft_input)
    channel_inputs = Lambda(util.complex_to_channels, name="complex_to_channels_2")(complex_image)

    # basic unet 2 in i-domain
    basic_unet_input_shape = K.int_shape(channel_inputs)[1:]
    basic_unet = build_basic_unet(input_shape = basic_unet_input_shape, depths = depths, n_shot = n_shot, kernel_size = kernel_size, bn_layer=bn_layer, name="i_basic_unet")
    decoded = basic_unet(channel_inputs)

    # channel back to complex
    output = Lambda(util.channels_to_complex, name="channels_to_complex", dtype='complex64')(decoded)
    ki_unet = Model(inputs, output, name='ki_unet')

    return ki_unet