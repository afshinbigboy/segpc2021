# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 18:15:43 2019

@author: Reza Azad
"""
from __future__ import division
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import model_unet as M
import numpy as np
from keras.callbacks import ModelCheckpoint, TensorBoard,ReduceLROnPlateau
from keras import callbacks
import pickle
import cv2
import matplotlib.pyplot as plt
import utilz as U   
from skimage.color import rgb2lab

patch_h     = 512
patch_w     = 512


def resize(x):
    if x.ndim == 4:
      x2 = np.zeros((x.shape[0], patch_h, patch_w, 3))
    else:
      x2 = np.zeros((x.shape[0], patch_h, patch_w))
    for idx in range(len(x)):
        x2[idx] = cv2.resize(x[idx], (patch_h, patch_w), interpolation=cv2.INTER_NEAREST)   
    return x2
      
ADD = '/reza/datafast1/DNS/segpc/'
tr_data    = np.load(ADD+'data/data_train.npy') /255.
te_data    = np.load(ADD+'data/data_test.npy')  /255.


tr_mask_n    = np.load(ADD+'data/mask_train_n.npy') / 255.
tr_mask_c    = np.load(ADD+'data/mask_train_c.npy') / 255.

te_mask_n    = np.load(ADD+'data/mask_test_n.npy')/ 255.
te_mask_c    = np.load(ADD+'data/mask_test_c.npy')/ 255.

Test_names   = np.load(ADD+'data/Test_names.npy')

print(np.unique(tr_mask_n))

tr_data   = resize(tr_data)
te_data   = resize(te_data)

tr_mask_n = resize(tr_mask_n)
te_mask_n = resize(te_mask_n)

tr_mask_c = resize(tr_mask_c)
te_mask_c = resize(te_mask_c)

tr_mask_c    = np.expand_dims(tr_mask_c, axis=3)
te_mask_c    = np.expand_dims(te_mask_c, axis=3)

tr_mask_n    = np.expand_dims(tr_mask_n, axis=3)
te_mask_n    = np.expand_dims(te_mask_n, axis=3) 

Backtr = np.ones(tr_mask_c.shape)
Backte = np.ones(te_mask_c.shape)

Backtr -=  tr_mask_n
Backtr -=  tr_mask_c

Backte -=  te_mask_n
Backte -=  te_mask_c


Jmask_tr = np.concatenate((Backtr, tr_mask_n, tr_mask_c), axis = 3)
Jmask_te = np.concatenate((Backte, te_mask_n, te_mask_c), axis = 3)


       
####################################  Load Data #####################################

model = M.unet_dns(pretrained_weights = None, input_size = (patch_h, patch_w, 3), LR = 1e-4, OUt_ch = 3)
model.summary()

print('Training')
batch_size = 6
nb_epoch   = 20
  

fig,ax = plt.subplots(5,3,figsize=[15,15])

print(te_data.shape, Jmask_te.shape)

Flat_train = True
if Flat_train:
    for ids in range(nb_epoch):
       model.fit(tr_data, Jmask_tr,
              batch_size=batch_size,
              epochs=1,
              shuffle=True,
              verbose=1,
              validation_data=(te_data, Jmask_te))
       predictions= model.predict(te_data[0:5], batch_size=5, verbose=1)

       for idx in range(5):
           ax[idx, 0].imshow(np.uint8(te_data[idx]*255.))
           ax[idx, 1].imshow(np.squeeze(Jmask_te[idx]*255.))
           ax[idx, 2].imshow(np.squeeze(predictions[idx]*255.))
       if ids % 5==0:
          plt.savefig('../../represent/epoch_'+str(ids+1)+' sample_results_simpleunet.png') 
    
model.save_weights('weight_sus_n.hdf5')


