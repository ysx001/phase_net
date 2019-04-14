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

def conv_bn_relu(inputs, n_filters, kernel_size):
    conv = Conv2D(filters = n_filters, kernel_size=kernel_size, padding='same', kernel_initializer="he_normal")(inputs)
    bn = BatchNormalization(axis=3)(conv)
    act = Activation('relu')(bn)
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
        print("in additional_conv")
        shortcut = Conv2D(filters=residual_shape[CHANNEL_AXIS],
                          kernel_size=(1, 1),
                          strides=(stride_width, stride_height),
                          padding="valid",
                          kernel_initializer="he_normal",
                          kernel_regularizer=l2(0.0001))(input)

    return add([shortcut, residual])

def build_down_conv_block(input_block, depth, kernel_size=3, repeat = 3, residual = True, last_layer = False):
    # residual block (using proposed method: https://arxiv.org/abs/1603.05027)
    # first block of downsampling
    cbr = conv_bn_relu(input_block, depth, kernel_size) # conv-bn-relu
    for i in range(repeat - 1):
        cbr = conv_bn_relu(cbr, depth, kernel_size) # conv-bn-relu

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
def build_up_conv_block(input_block, skip_block, depth, kernel_size=3, repeat = 3, residual = True):
    # up-conv 2x2
    up = UpSampling2D(size=(2, 2))(input_block)
    up = conv_bn_relu(up, depth, kernel_size) # conv-bn-relu
    
    # concate
    inputs = concatenate([up, skip_block], axis=3)

    # repeat convs
    cbr = conv_bn_relu(inputs, depth, kernel_size) # conv-bn-relu
    for i in range(repeat - 1):
        cbr = conv_bn_relu(cbr, depth, kernel_size) # conv-bn-relu
    
    if residual:
        # do a conv for res input
        conv_inputs = Conv2D(filters = depth, kernel_size=kernel_size, padding='same', kernel_initializer="he_normal")(inputs)
        res = Conv2D(depth, kernel_size=kernel_size, padding='same')(cbr) # residual => full activation
        return shortcut(conv_inputs, res)
    
    return cbr

def build_unet(input_shape = (256, 304, 1), depths = [16, 32, 64, 128], num_ch = 8, kernel_size = 3, l2_reg = 0., ifft_flag = False):
    # Build encoder
    encoder_sc_depth_dict = {}
    inputs = Input(shape=input_shape, name='encoder_input')
    next_encoder_input = Conv2D(filters = depths[0], kernel_size=kernel_size, padding='same', kernel_initializer="he_normal")(inputs)

    # encoder
    for ii in range(len(depths)):
        isLast = (ii == (len(depths) - 1))
        encoder_sc_depth_dict[depths[ii]], next_encoder_input = build_down_conv_block(next_encoder_input, depths[ii], residual = True, last_layer = isLast)

    next_decoder_input = Dropout(0.5)(next_encoder_input)

    # decoder
    for jj in range(len(depths)-2, -1, -1): # doing backward iteration, stupid python
        depth = depths[jj]
        next_decoder_input = build_up_conv_block(next_decoder_input, encoder_sc_depth_dict[depth], depth)

    decoded = Conv2D(filters = num_ch, kernel_size=kernel_size, padding='same', kernel_initializer="he_normal")(next_decoder_input)
    decoded = BatchNormalization()(decoded)
    decoded = Activation('relu')(decoded)

    if ifft_flag:
        # inverse fft
        ifft_input = Lambda(util.channels_to_complex)(decoded)
        # explictly pass in output_shape to avoid this being run twice to infer output shape
        output = Lambda(util.ifft2c, output_shape=(None, input_shape[0], input_shape[1], input_shape[2]))(ifft_input)
        autoencoder = Model(inputs, output, name='autoencoder')
    else:
        autoencoder = Model(inputs, decoded, name='autoencoder')
        
    return autoencoder

# %%
img_w = 400
img_h = 496
num_ch = 1
input_shape = (img_w, img_h, num_ch)

unet = build_unet(input_shape, num_ch = 8, ifft_flag=True)
unet.summary()