# %%
import data_generator
import util
import os

############################################################
#  Utility Functions
############################################################
def setup_output_path(dpRoot):
        fpOutput = os.path.join(dpRoot, network_name) 

        if not os.path.exists(fpOutput):
                os.makedirs(fpOutput)

        fpBase = os.path.join(fpOutput, 'network')
        if not os.path.exists(fpBase):
                os.makedirs(fpBase)

        return fpOutput, fpBase


# %% 
import keras.backend as K
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from keras import losses
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import load_model
import tensorflow as tf
from time import time
import unet_builder


img_w = 256
img_h = 256
num_ch = 1
n_shot = 4
batch_size = 8
multi = 2
depths = [i*multi for i in [16, 32, 64, 128]]
train = False
domain = 0

num_epoch = 500

# %%
if (domain == 0):
        network_name = 'unet_imagenet_ki_net_l1'
        unet = unet_builder.build_unet_k_i_domain(\
                        input_shape=(img_w, img_h, num_ch), \
                        depths=depths, \
                        n_shot = n_shot, \
                        bn_layer=False)
elif (domain == 1):
        network_name = 'unet_imagenet_kspace_net_l1'
        unet = unet_builder.build_unet_k_domain(\
                        input_shape=(img_w, img_h, num_ch), \
                        depths=depths, \
                        n_shot = n_shot, \
                        bn_layer=False)
else:
        network_name = 'unet_imagenet_image_net_l1'
        unet = unet_builder.build_unet_i_domain(\
                        input_shape=(img_w, img_h, num_ch), \
                        depths=depths, \
                        n_shot = n_shot, \
                        bn_layer=False)
unet.summary()

# %% set up adam
adam_opt = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
unet.compile(loss = losses.mean_absolute_error, optimizer = adam_opt)

# %% generate data
train_images_data_fp, test_images_data_fp = data_generator.get_list_image_id()

# Parameters
params = {'dim': (img_w, img_h),
        'batch_size': batch_size,
        'shuffle': True}

imagenet_training_generator = data_generator.ImageNetDataGenerator(train_images_data_fp, **params)
imagenet_validation_generator = data_generator.ImageNetDataGenerator(test_images_data_fp, **params)

# %% save checkpoints data


dpRoot = data_generator.setup_path()
fpOutput, fpBase = setup_output_path(dpRoot)

K.tensorflow_backend._get_available_gpus()

sess = tf.Session(config=tf.ConfigProto(
      allow_soft_placement=True, log_device_placement=True))
K.set_session(sess)
fnCp = network_name + str(num_epoch)
fpCp = os.path.join(fpBase, fnCp + '.h5') 


if train:
        callbacks = [
                EarlyStopping(patience=10, verbose=1),\
                ModelCheckpoint(fpCp, monitor='val_loss', save_best_only = True),\
                ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.0001, verbose=1),\
                TensorBoard(log_dir="logs/{}".format(time()))]
        # Bug in keras, force initialize all variable for GPU mode
        K.get_session().run(tf.initialize_all_variables())

        # %%                        # steps_per_epoch=1,
        history = unet.fit_generator(generator=imagenet_training_generator,
                        validation_data=imagenet_validation_generator,
                        workers=3,
                        use_multiprocessing=True,
                        epochs = num_epoch,  
                        callbacks = callbacks)


        # save loss scores
        import scipy.io as sio
        sio.savemat(os.path.join(fpBase, fnCp + '.mat') , {'loss_train':history.history['loss'], 'loss_val':history.history['val_loss']})

else:
        # %% load previous trained model
        unet.load_weights(fpCp)
        unet.summary()


# %%
test_image, ground_truth = data_generator.load_test_images(test_images_data_fp, 1)
unet_pred = unet.predict(test_image) 

# fpPred = os.path.join(fpOutput, 'pred')
# if not os.path.exists(fpPred):
#         os.makedirs(fpPred)

# %% save
import scipy.io as sio
sio.savemat(os.path.join(fpBase, network_name + 'res.mat') , {'unet_pred':unet_pred, 'test_image':test_image, 'ground_truth':ground_truth})    

# from matplotlib import pyplot as plt
# import numpy as np
# for ii in np.arange(0, 256):
#     res = np.abs(flair_test[ii, :, :, 0] - flair_pred[ii, :, :, 0])
#     concate = np.concatenate((flair_test[ ii, :, :, 0], flair_pred[ ii, :, :, 0], lesion_test[ ii, :, :, 0], res), axis=0)
#     # plt.imshow(np.abs(res[700, :, :, 0]), clim=(-2., 2.), cmap='gray')
#     plt.imsave(os.path.join(fpPred, 'concate' + str(ii) + '.png'), concate, cmap='gray')
#     plt.imsave(os.path.join(fpPred, 'res' + str(ii) + '.png'), res, cmap='gray')
#     plt.imsave(os.path.join(fpPred, 'flair_pred' + str(ii) + '.png'), flair_pred[ ii, :, :, 0], cmap='gray')

