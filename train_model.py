# %% 
import keras.backend as K
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras import losses
import keras.backend as K
from keras.layers import Input, Lambda
import tensorflow as tf
import unet_builder
import data_generator
import util

img_w = 400
img_h = 496
num_ch = 1
input_shape = (img_w, img_h, num_ch)

# %%
unet = unet_builder.get_unet(input_shape, num_ch = 8, ifft_flag=True)
unet.summary()

# %% set up adam
adam_opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
unet.compile(loss = losses.mean_absolute_error, optimizer = adam_opt)

# %%
train_images_data_fp, test_images_data_fp = = data_generator.get_list_image_id()

# Parameters
params = {'dim': (img_w, img_h),
          'batch_size': 16,
          'n_classes': 6,
          'n_channels': 1,
          'shuffle': True}

training_generator = data_generator.ImageNetDataGenerator(train_images_data_fp, **params)
validation_generator = data_generator.ImageNetDataGenerator(test_images_data_fp, **params)

unet.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=True,
                    workers=6)

# %% tuning network with new training data
sz_batch = 16
num_epoch = 5

# new_network_name = 'unetsegres_nomask_l1_double_lesion'
fpOutput = os.path.join(dpRoot, network_name) 

if not os.path.exists(fpOutput):
    os.makedirs(fpOutput)

fpBase = os.path.join(fpOutput, 'network')
if not os.path.exists(fpBase):
        os.makedirs(fpBase)

fpPred = os.path.join(fpOutput, 'pred')
if not os.path.exists(fpPred):
        os.makedirs(fpPred)

# save checkpoints data
fnCp = network_name + str(num_epoch)
fpCp = os.path.join(fpBase, fnCp + '.h5') 
callbacks = [EarlyStopping(patience=10, verbose=1),\
    ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1),\
        ModelCheckpoint(fpCp, monitor='val_loss', save_best_only = True)]

K.tensorflow_backend._get_available_gpus()

sess = tf.Session(config=tf.ConfigProto(
      allow_soft_placement=True, log_device_placement=True))
K.set_session(sess)

# Bug in keras, force initialize all variable for GPU mode
K.get_session().run(tf.initialize_all_variables())

history = autoencoder.fit(x = t2w_block_train, 
                          y = lmask_block_train, 
                          validation_data = (t2w_block_test, lmask_block_test),
                          batch_size = sz_batch, 
                          epochs = num_epoch,  
                          shuffle = True, 
                          callbacks = callbacks, 
                          verbose=1)

# save loss scores
import scipy.io as sio
sio.savemat(os.path.join(fpBase, fnCp + '.mat') , {'loss_train':history.history['loss'], 'loss_val':history.history['val_loss']})

# %% Prediction
lmask_block_pred = autoencoder.predict(t2w_block_test[0:256, :, :, :])

# %% save
import scipy.io as sio
sio.savemat(os.path.join(dpRoot, network_name, network_name + 'res.mat') , {'lmask_block_pred':lmask_block_pred, 't2w_block_test':t2w_block_test, 'lmask_block_test':lmask_block_test})    

from matplotlib import pyplot as plt
import numpy as np
for ii in np.arange(0, 256):
    res = np.abs(lmask_block_test[ii, :, :, 0] - lmask_block_pred[ii, :, :, 0])
    concate = np.concatenate((t2w_block_test[ ii, :, :, 0], lmask_block_test[ ii, :, :, 0], lmask_block_pred[ ii, :, :, 0], res), axis=0)
    # plt.imshow(np.abs(res[700, :, :, 0]), clim=(-2., 2.), cmap='gray')
    plt.imsave(os.path.join(fpPred, 'concate' + str(ii) + '.png'), concate, cmap='gray')
