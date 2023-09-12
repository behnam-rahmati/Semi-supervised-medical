import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from keras import backend as K
import tensorflow as tf
#!/usr/bin/env python2.7
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
print(tf.executing_eagerly())
from keras import optimizers
from keras.models import Model
from keras.layers import Dropout, Lambda
from keras.layers import Input, average
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose
from keras.layers import ZeroPadding2D, Cropping2D
from keras import backend as K
from keras.losses import categorical_crossentropy as CCE
import sys
import math
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import distance_transform_edt
from keras.losses import binary_crossentropy


# def hausdorff_distance(y_true, y_pred, threshold=0.5):
    # """
    # Computes the Hausdorff distance between a binary or continuous segmentation prediction and the ground truth.

    # Args:
        # y_pred (ndarray): Prediction segmentation mask, shape (num_batches, height, width, 1)
        # y_true (ndarray): Ground truth segmentation mask, shape (num_batches, height, width, 1)
        # threshold (float): Threshold value to convert y_pred to binary mask

    # Returns:
        # Hausdorff distance (float)
    # """
    # y_true = K.cast(np.expand_dims (y_true[:, :, :, 0] , axis = 3), 'float32')
    
    # # Threshold the prediction to obtain a binary mask
    # y_pred_binary = (y_pred >= threshold).astype(np.float32)

    # # Create binary masks for tissue and background classes
    # tissue_pred = y_pred_binary
    # tissue_gt = (y_true == 1).astype(np.float32)
    # bg_pred = 1 - tissue_pred
    # bg_gt = (y_true == 0).astype(np.float32)

    # # Compute the distance from each tissue point in y_pred to the closest tissue point in y_true
    # dist_tissue_pred_to_gt = distance_transform_edt(bg_gt[..., 0])
    
    # dist_tissue_pred_to_gt *= tissue_pred[..., 0]
    # # Compute the distance from each tissue point in y_true to the closest tissue point in y_pred
    # dist_tissue_gt_to_pred = distance_transform_edt(bg_pred[..., 0])
    # dist_tissue_gt_to_pred *= tissue_gt[..., 0]

    # # Compute the Hausdorff distance as the maximum distance from a tissue point in y_pred to the closest tissue point in y_true,
    # # and vice versa
    # hausdorff_dist = np.max(np.array([np.max(dist_tissue_pred_to_gt), np.max(dist_tissue_gt_to_pred)]))

    # return hausdorff_dist  
    
def mean_teacher_loss(y_true, y_pred):
    
    y_true_student = K.expand_dims(K.cast(y_true[:, :, :, 0], 'float32'), axis=-1)
    y_pred_teacher = K.expand_dims(K.cast(y_true[:, :, :, 1], 'float32'), axis=-1)
    y_pred_student = y_pred  # corrected variable name
    # print(y_true)
    # print(y_pred)
    cross_entropy_loss = K.mean (binary_crossentropy(y_true_student, y_pred_student))

    consistency_loss = 10 * tf.reduce_mean(tf.square(y_pred_student - y_pred_teacher))
    # print(    K.print_tensor(cross_entropy_loss, message='cross_entropy_loss = '))
    # print(    K.print_tensor(consistency_loss, message='consistency_loss = '))
    
    alpha = 0.5
    combined_loss = alpha * cross_entropy_loss + (1 - alpha) * consistency_loss

    return combined_loss
    
# def hausdorff_distance(y_true, y_pred, threshold=0.5):
    # """
    # Computes the Hausdorff distance between a binary or continuous segmentation prediction and the ground truth.

    # Args:
        # y_pred (ndarray): Prediction segmentation mask, shape (num_batches, height, width, 1)
        # y_true (ndarray): Ground truth segmentation mask, shape (num_batches, height, width, 1)
        # threshold (float): Threshold value to convert y_pred to binary mask

    # Returns:
        # Hausdorff distance (float)
    # """
    # # print(y_true.shape)
    # # print(y_pred.shape)
    
    # # y_true = np.expand_dims(y_true[..., 0], axis=3).astype('float32')

    # # Threshold the prediction to obtain a binary mask
    # y_pred_binary = (y_pred >= threshold).numpy().astype('float32')

    # # Compute the Hausdorff distance between the two masks
    # distances = []
    # for i in range(len(y_pred_binary)):
        # pred_mask = y_pred_binary[i, ..., 0]
        # true_mask = y_true[i, ..., 0]
        # distance = max(directed_hausdorff(pred_mask, true_mask)[0],
                       # directed_hausdorff(true_mask, pred_mask)[0])
        # distances.append(distance)

    # hausdorff_dist = np.max(distances)

    # return hausdorff_dist
def hausdorff_distance(y_true_binary, y_pred, threshold=0.5):
    """
    Computes the Hausdorff distance between a binary or continuous segmentation prediction and the ground truth.

    Args:
        y_pred (ndarray): Prediction segmentation mask, shape (num_batches, height, width, 1)
        y_true (ndarray): Ground truth segmentation mask, shape (num_batches, height, width, 1)
        threshold (float): Threshold value to convert y_pred to binary mask

    Returns:
        Hausdorff distance (float)
    """
    # Threshold the prediction and convert to binary mask
    y_pred_binary = tf.cast(y_pred >= threshold, dtype=tf.float32).numpy()
    diff = np.abs(y_pred_binary - y_true_binary)

    # Compute the distance between each foreground pixel in y_pred and the nearest foreground pixel in y_true
    # Compute the average distance of foreground pixels from the true mask
    dist_pred = np.mean(np.sqrt(np.sum(diff * y_true_binary, axis=(1, 2, 3))))

    # Compute the average distance of foreground pixels from the predicted mask
    dist_true = np.mean(np.sqrt(np.sum(diff * y_pred_binary, axis=(1, 2, 3))))

    # Compute the alternative Hausdorff distance as the maximum of the two distances
    hausdorff_dist = max(dist_pred, dist_true)

    return hausdorff_dist

    return hausdorff_dist
    
def AAAI_Loss(y_true, y_pred): 
    # print(y_true)
    # print(y_pred.shape)
    ignore_mask = K.expand_dims(y_true[:, :, :, 1] , axis=-1)
    ignore_mask = K.cast(ignore_mask, 'float32')
    # # print( K.sum(ignore_mask))
    if (K.sum(ignore_mask) == 0):
        flag_gt = 1
    else:
        flag_gt = 0
    y_pred = K.cast(y_pred, 'float32')
    y_true = K.cast(y_true[:, :, :, 0], 'float32')
    y_true = K.expand_dims(y_true, axis=-1)
    
    # print(y_pred.shape, "pred")
    # print(y_true.shape, "true")
    x = y_pred[:,1:,:,:] - y_pred[:,:-1,:,:]    
    y = y_pred[:,:,1:,:] - y_pred[:,:,:-1,:] 
  
    delta_x = x[:,1:,:-2,:]**2
    # print(x.shape)
    
    delta_y = y[:,:-2,1:,:]**2
    # print(delta_x.shape)
    
    delta_u = K.abs(delta_x + delta_y) 
    lenth = K.mean(K.sqrt(delta_u + 0.00000001) ) # equ.(11) in the paper

    """
    region term
    """

    C_1 = np.ones((1,128, 128,1))
    C_2 = np.zeros((1,128, 128,1))
    region_in = K.abs(K.mean( y_pred * ((y_true - C_1)**2)  )) # equ.(12) in the paper
    region_out = K.abs(K.mean( (1-y_pred) * ((y_true - C_2)**2)  )) # equ.(12) in the paper
    
    

    lambdaP = 1 # lambda parameter could be various.
    mu = 1# mu parameter could be various.
    loss_ce = K.mean((K.binary_crossentropy(y_true, y_pred)), axis = (1,2,3))
    all_c = lenth+ region_in + region_out
    lenth_c = all_c/(lenth+ 0.0000001)
    regi_c =  all_c/(region_in+ 0.0000001)
    rego_c = all_c/(region_out+ 0.0000001)
    # print(region_in, "region in")
    # print(region_out, "region out")
    # print(lenth, "length")
    # print(loss_ce, "cross entropy")
    # # Standardize the terms
    # lenth = (lenth - K.mean(lenth)) / K.std(lenth)
    # region_in = (region_in - K.mean(region_in)) / K.std(region_in)
    # region_out = (region_out - K.mean(region_out)) / K.std(region_out)
    # loss_ce = (loss_ce - K.mean(loss_ce)) / K.std(loss_ce)
    if(flag_gt):
        loss = loss_ce
    else:
        loss =  lenth*0  + lambdaP * (  region_in +  region_out) +loss_ce
    # return lenth  +  (  region_in +  region_out) + loss_ce
    
    return  loss

    
def AAAI2_Loss(legth_coeff):
    def loss(y_true, y_pred): 
        # print(y_true)
        # print(y_pred.shape)
        ignore_mask = K.expand_dims(y_true[:, :, :, 1] , axis=-1)
        ignore_mask = K.cast(ignore_mask, 'float32')
        # # print( K.sum(ignore_mask))
        if (K.sum(ignore_mask) == 0):
            flag_gt = 1
        else:
            flag_gt = 0
        # print((K.sum(ignore_mask)))
        y_pred = K.cast(y_pred, 'float32')
        y_true = K.cast(y_true[:, :, :, 0], 'float32')
        y_true = K.expand_dims(y_true, axis=-1)
       
        # print(y_pred.shape, "pred")
        # print(y_true.shape, "true")
        x = y_pred[:,1:,:,:] - y_pred[:,:-1,:,:]    
        y = y_pred[:,:,1:,:] - y_pred[:,:,:-1,:] 

        delta_x = x[:,1:,:-2,:]**2
        # print(x.shape)
        
        delta_y = y[:,:-2,1:,:]**2
        # print(delta_x.shape)
        
        delta_u = K.abs(delta_x + delta_y) * ignore_mask [:,1:127, 1:127, :]
        lenth = K.mean(K.sqrt(delta_u + 0.00000001) ) # equ.(11) in the paper

        """
        region term
        """

        C_1 = np.ones((1,128, 128,1))
        C_2 = np.zeros((1,128, 128,1))
        region_in = K.abs(K.mean( ignore_mask * y_pred * ((y_true - C_1)**2)  )) # equ.(12) in the paper
        region_out = K.abs(K.mean( ignore_mask* (1-y_pred) * ((y_true - C_2)**2)  )) # equ.(12) in the paper
        
        

        lambdaP = 1 # lambda parameter could be various.
        mu = 1# mu parameter could be various.
        # print(1 - ignore_mask)
        loss_ce = K.mean((K.binary_crossentropy(y_true, y_pred)* (1-ignore_mask)), axis = (1,2,3))
        # all_c = lenth + region_in + region_out + loss_ce 
        # lenth_c = all_c/(lenth + 0.0000001)
        # regi_c =  all_c/(region_in + 0.0000001)
        # rego_c = all_c/(region_out + 0.0000001)
        # ce_c = all_c/ (loss_ce  + 0.0000001)
        # print(region_in, "region in")
        # print(region_out, "region out")
        # print(lenth, "length")
        # print(loss_ce, "cross entropy")
        # print(flag_gt)
        # # Standardize the terms
        # lenth = (lenth - K.mean(lenth)) / K.std(lenth)
        # region_in = (region_in - K.mean(region_in)) / K.std(region_in)
        # region_out = (region_out - K.mean(region_out)) / K.std(region_out)
        # loss_ce = (loss_ce - K.mean(loss_ce)) / K.std(loss_ce)

        # return lenth  +  (  region_in +  region_out) + loss_ce
        if(flag_gt):
            loss = loss_ce
        else:
            # loss = loss_ce
            loss =  0.5 *lenth  +  ( region_in +  region_out * 10) + loss_ce 
        # print(loss, "loss")
        # print( lenth, "lenth")
        # print( region_in, "regin")
        # print(region_out, "rego")
        # print(loss_ce, "ce")
        # print(loss, "loss")
        
        return  loss
    return loss
    
def AC_Loss(y_true, y_pred): 

	"""
	lenth term
	"""

	x = y_pred[:,:,1:,:] - y_pred[:,:,:-1,:] # horizontal and vertical directions 
	y = y_pred[:,:,:,1:] - y_pred[:,:,:,:-1]

	delta_x = x[:,:,1:,:-2]**2
	delta_y = y[:,:,:-2,1:]**2
	delta_u = K.abs(delta_x + delta_y) 

	lenth = K.mean(K.sqrt(delta_u + 0.00000001)) # equ.(11) in the paper

	"""
	region term
	"""

	C_1 = np.ones((128, 128))
	C_2 = np.zeros((128, 128))

	region_in = K.abs(K.mean( y_pred[:,0,:,:] * ((y_true[:,0,:,:] - C_1)**2) ) ) # equ.(12) in the paper
	region_out = K.abs(K.mean( (1-y_pred[:,0,:,:]) * ((y_true[:,0,:,:] - C_2)**2) )) # equ.(12) in the paper

	lambdaP = 1 # lambda parameter could be various.
	mu = 1 # mu parameter could be various.
	
	return lenth + lambdaP * (mu * region_in + region_out) 
    
crop_size = 128
def total_variation_loss(x):
    a = tf.square(
        x[:, : crop_size - 1, : crop_size - 1, :] - x[:, 1:, : crop_size - 1, :]
    )
    b = tf.square(
        x[:, : crop_size - 1, : crop_size - 1, :] - x[:, : crop_size - 1, 1:, :]
    )
    return tf.reduce_sum(tf.pow(a + b, 1.25))
def custom_loss(ignore_matrix):
    def loss(ytrue, ypred):
        # print("hi")
        ignore_mat = K.cast(ignore_matrix, 'float32')
        # print("hii")

        ytrue = K.cast(ytrue, 'float32')
        # print("hiii")
        # print(np.array(ytrue).shape)
        ypred = K.cast(ypred, 'float32')
        # print("hiiiii")
        # x = K.binary_crossentropy(ytrue, ypred)
        # print(np.array(x).shape)
        # print(x)
        loss_ce = K.mean((K.binary_crossentropy(ytrue, ypred) * (1 - ignore_mat)), axis = (1,2,3))
        # print(np.array(loss_ce))
        # ypred_dx = ypred[:, :-1, :] - ypred[:, 1:, :]
        # ypred_dy = ypred[:, :, :-1] - ypred[:, :, 1:]
        # print('hi')
        y_true = np.array(ytrue).flatten()
        y_pred = np.array(ypred).flatten()
        epsilon = K.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        ignore_mat = np.array(ignore_mat).flatten()
        loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        loss = loss * (1 - ignore_mat)
        loss = np.sum(loss) / np.sum(1 - ignore_mat)
        tv = 0
        # for i in range(ypred.shape[1]-1):
            # for j in range(ypred.shape[2]-1):
                # tv += np.abs(ypred[0, i+1, j, 0] - ypred[0, i, j, 0])
                # tv += np.abs(ypred[0, i, j+1, 0] - ypred[0, i, j, 0])
        loss_tv = tv
        # loss_tv = K.mean(K.abs(ypred_dx)) + K.mean(K.abs(ypred_dy))
        # print(loss_ce)
        # print(loss_tv)
        # print(np.array(loss_tv).shape)
        return loss_ce+ loss_tv
    return loss
        
    
    
    
    
def dice_coef(y_true, y_pred, smooth=0.0):
    '''Average dice coefficient per batch.'''
    axes = (1,2,3)
    intersection = K.sum(y_true * y_pred, axis=axes)
    summation = K.sum(y_true, axis=axes) + K.sum(y_pred, axis=axes)
    
    return K.mean((2.0 * intersection + smooth) / (summation + smooth), axis=0)

def mvn(tensor):
    '''Performs per-channel spatial mean-variance normalization.'''
    epsilon = 1e-6
    mean = K.mean(tensor, axis=(1,2), keepdims=True)
    std = K.std(tensor, axis=(1,2), keepdims=True)
    mvn = (tensor - mean) / (std + epsilon)
    
    return mvn
    
def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred, smooth=10.0)

    
def dice_coef(y_true, y_pred, smooth=0.0):
    # print(np.array(y_true).shape)
    '''Average dice coefficient per batch.'''
    axes = (1,2,3)
    intersection = K.sum(y_true * y_pred, axis=axes)
    summation = K.sum(y_true, axis=axes) + K.sum(y_pred, axis=axes)   
    return K.mean((2.0 * intersection + smooth) / (summation + smooth), axis=0)



def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred, smooth=10.0)

def jaccard_coef(y_true, y_pred, smooth=0.0):
    '''Average jaccard coefficient per batch.'''
    axes = (1,2,3)
    intersection = K.sum(y_true * y_pred, axis=axes)
    union = K.sum(y_true, axis=axes) + K.sum(y_pred, axis=axes) - intersection
    return K.mean( (intersection + smooth) / (union + smooth), axis=0)
    
    
def unet(input_size, pretrained_weights = None ):
    data = Input(shape=input_size, dtype='float', name='data')
    mvn0 = Lambda(mvn, name='mvn0')(data)
    #pad = ZeroPadding2D(padding=5, name='pad')(mvn0)

    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(mvn0)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    mvn1 = Lambda(mvn, name='mvn1')(conv1)

    pool1 = MaxPooling2D(pool_size=(2, 2))(mvn1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    mvn2 = Lambda(mvn, name='mvn2')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(mvn2)
    mvn3 = Lambda(mvn, name='mvn3')(pool2)

    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(mvn3)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    mvn4 = Lambda(mvn, name='mvn4')(conv3)

    pool3 = MaxPooling2D(pool_size=(2, 2))(mvn4)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    #drop4 = Dropout(0.5)(conv4)
    mvn5 = Lambda(mvn, name='mvn5')(conv4)

    pool4 = MaxPooling2D(pool_size=(2, 2))(mvn5)
    mvn6 = Lambda(mvn, name='mvn6')(pool4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(mvn6)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    #drop5 = Dropout(0.5)(conv5)
    mvn7 = Lambda(mvn, name='mvn7')(conv5)
    
    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(mvn7))
    merge6 = concatenate([conv4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    mvn8 = Lambda(mvn, name='mvn8')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(mvn8))
    merge7 = concatenate([conv3,up7], axis = 3)    
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    mvn9 = Lambda(mvn, name='mvn9')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(mvn9))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    mvn10 = Lambda(mvn, name='mvn10')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(mvn10))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)
    model = Model(data, conv10)
    print(model.summary())
    sgd = tf.keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
    # model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=[ dice_coef])
    
    #model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model

def fcn_model(input_shape, num_classes, weights = None):
    ''' "Skip" FCN architecture similar to Long et al., 2015
    https://arxiv.org/abs/1411.4038
    '''
    if num_classes == 2:
        num_classes = 1
        loss = dice_coef_loss
        # loss = weighted
        activation = 'sigmoid'
    else:

        loss = 'categorical_crossentropy'
        activation = 'softmax'

    kwargs = dict(
        kernel_size=3,
        strides=1,
        activation='relu',
        padding='same',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        trainable=True,
    )

    data = Input(shape=input_shape, dtype='float', name='data')
    mvn0 = Lambda(mvn, name='mvn0')(data)
    pad = ZeroPadding2D(padding=5, name='pad')(mvn0)

    conv1 = Conv2D(filters=64, name='conv1', **kwargs)(pad)
    mvn1 = Lambda(mvn, name='mvn1')(conv1)

    conv2 = Conv2D(filters=64, name='conv2', **kwargs)(mvn1)
    mvn2 = Lambda(mvn, name='mvn2')(conv2)

    conv3 = Conv2D(filters=64, name='conv3', **kwargs)(mvn2)
    mvn3 = Lambda(mvn, name='mvn3')(conv3)
    pool1 = MaxPooling2D(pool_size=3, strides=2,
                    padding='valid', name='pool1')(mvn3)
    print('**********pool1*****************')                
    print(pool1.shape)

    conv4 = Conv2D(filters=128, name='conv4', **kwargs)(pool1)
    mvn4 = Lambda(mvn, name='mvn4')(conv4)

    conv5 = Conv2D(filters=128, name='conv5', **kwargs)(mvn4)
    mvn5 = Lambda(mvn, name='mvn5')(conv5)

    conv6 = Conv2D(filters=128, name='conv6', **kwargs)(mvn5)
    mvn6 = Lambda(mvn, name='mvn6')(conv6)

    conv7 = Conv2D(filters=128, name='conv7', **kwargs)(mvn6)
    mvn7 = Lambda(mvn, name='mvn7')(conv7)
    pool2 = MaxPooling2D(pool_size=3, strides=2,
                    padding='valid', name='pool2')(mvn7)

    print('**********pool2*****************')                
    print(pool2.shape)
    conv8 = Conv2D(filters=256, name='conv8', **kwargs)(pool2)
    mvn8 = Lambda(mvn, name='mvn8')(conv8)

    conv9 = Conv2D(filters=256, name='conv9', **kwargs)(mvn8)
    mvn9 = Lambda(mvn, name='mvn9')(conv9)

    conv10 = Conv2D(filters=256, name='conv10', **kwargs)(mvn9)
    mvn10 = Lambda(mvn, name='mvn10')(conv10)

    conv11 = Conv2D(filters=256, name='conv11', **kwargs)(mvn10)
    mvn11 = Lambda(mvn, name='mvn11')(conv11)
    pool3 = MaxPooling2D(pool_size=3, strides=2,
                    padding='valid', name='pool3')(mvn11)
    drop1 = Dropout(rate=0.5, name='drop1')(pool3)
    print('**********pool3*****************')                
    print(pool3.shape)

    conv12 = Conv2D(filters=512, name='conv12', **kwargs)(drop1)
    mvn12 = Lambda(mvn, name='mvn12')(conv12)

    conv13 = Conv2D(filters=512, name='conv13', **kwargs)(mvn12)
    mvn13 = Lambda(mvn, name='mvn13')(conv13)

    conv14 = Conv2D(filters=512, name='conv14', **kwargs)(mvn13)
    mvn14 = Lambda(mvn, name='mvn14')(conv14)

    conv15 = Conv2D(filters=512, name='conv15', **kwargs)(mvn14)
    mvn15 = Lambda(mvn, name='mvn15')(conv15)
    drop2 = Dropout(rate=0.5, name='drop2')(mvn15)


    score_conv15 = Conv2D(filters=num_classes, kernel_size=1,
                        strides=1, activation=None, padding='valid',
                        kernel_initializer='glorot_uniform', use_bias=True,
                        name='score_conv15')(drop2)
    upsample1 = Conv2DTranspose(filters=num_classes, kernel_size=3,
                        strides=2, activation=None, padding='valid',
                        kernel_initializer='glorot_uniform', use_bias=False,
                        name='upsample1')(score_conv15)
    score_conv11 = Conv2D(filters=num_classes, kernel_size=1,
                        strides=1, activation=None, padding='valid',
                        kernel_initializer='glorot_uniform', use_bias=True,
                        name='score_conv11')(mvn11)
    crop1 = Lambda(crop, name='crop1')([upsample1, score_conv11])
    fuse_scores1 = average([crop1, upsample1], name='fuse_scores1')

    upsample2 = Conv2DTranspose(filters=num_classes, kernel_size=3,
                        strides=2, activation=None, padding='valid',
                        kernel_initializer='glorot_uniform', use_bias=False,
                        name='upsample2')(fuse_scores1)
    score_conv7 = Conv2D(filters=num_classes, kernel_size=1,
                        strides=1, activation=None, padding='valid',
                        kernel_initializer='glorot_uniform', use_bias=True,
                        name='score_conv7')(mvn7)
    crop2 = Lambda(crop, name='crop2')([upsample2, score_conv7])
    fuse_scores2 = average([crop2, upsample2], name='fuse_scores2')

    upsample3 = Conv2DTranspose(filters=num_classes, kernel_size=3,
                        strides=2, activation=None, padding='valid',
                        kernel_initializer='glorot_uniform', use_bias=False,
                        name='upsample3')(fuse_scores2)
    print('**********upsample3*****************')                
    print(upsample3.shape)                        
    crop3 = Lambda(crop, name='crop3')([data, upsample3])
    predictions = Conv2D(filters=num_classes, kernel_size=1,
                        strides=1, activation=activation, padding='valid',
                        kernel_initializer='glorot_uniform', use_bias=True,
                        name='predictions')(crop3)

    model = Model(inputs=data, outputs=predictions)
    # model.save_weights('my_initial_weights.h5')

    if weights is not None:
        model.load_weights(weights)
    sgd = tf.keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss=loss,
                  metrics=['accuracy', dice_coef, jaccard_coef], run_eagerly = True)

    return model
    
def crop(tensors):
    '''
    List of 2 tensors, the second tensor having larger spatial dimensions.
    '''
    h_dims, w_dims = [], []
    for t in tensors:
        b, h, w, d = K.int_shape(t)
        
        h_dims.append(h)
        w_dims.append(w)
    crop_h, crop_w = (h_dims[1] - h_dims[0]), (w_dims[1] - w_dims[0])
    rem_h = crop_h % 2
    rem_w = crop_w % 2
    crop_h_dims = (crop_h / 2, crop_h / 2 + rem_h)
    lst1=list(crop_h_dims)
    lst1[0]=int(lst1[0])
    lst1[1]=int(lst1[1])
    crop_h_dims=tuple(lst1)
    crop_w_dims = (crop_w / 2, crop_w / 2 + rem_w)
    lst2=list(crop_w_dims)
    lst2[0]=int(lst2[0])
    lst2[1]=int(lst2[1])
    crop_w_dims=tuple(lst2)
    cropped = Cropping2D(cropping=(crop_h_dims, crop_w_dims))(tensors[1])
    
    return cropped
