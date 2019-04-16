
#%%
from scipy.io import loadmat
import os
import matplotlib.pyplot as plt

network = "unet_imagenet_phase_l1"
# %% load loss
import numpy as np
fpLoss = os.path.join(network, 'network', network + '250.mat') 
loss_mat = loadmat(fpLoss)
loss_train = loss_mat['loss_train']
print(loss_train.shape)
loss_val = loss_mat['loss_val']
print(loss_val.shape)

# summarize history for loss
plt.figure(figsize=(8, 8))
plt.title(network + ' - model loss')
plt.plot(loss_train[0], label="training loss")
plt.plot(loss_val[0], label="val loss")
plt.plot( np.argmin(loss_val[0]), np.min(loss_val[0]), marker="x", color="r", label="best model")
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# %% load result matrix
import numpy as np
fpRes = os.path.join(network, 'network', network + 'res.mat') 
res_mat = loadmat(fpRes)
ground_truth = res_mat['ground_truth']
test_image = res_mat['test_image']
unet_pred = res_mat['unet_pred']


# print(loss_train.shape)
# print(loss_val.shape)