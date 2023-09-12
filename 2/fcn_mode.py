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
from keras.losses import categorical_crossentropy 
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

    C_1 = np.ones((1,100, 100,1))
    C_2 = np.zeros((1,100, 100,1))
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
        loss = 0.5* lenth_c * lenth  + lambdaP * ( regi_c * region_in +  rego_c * region_out) 
    # return lenth  +  (  region_in +  region_out) + loss_ce
    
    return  loss

    
def AAAI2_Loss(y_true, y_pred): 
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
    
    delta_u = K.abs(delta_x + delta_y) * ignore_mask [:,1:99, 1:99, :]
    lenth = K.mean(K.sqrt(delta_u + 0.00000001) ) # equ.(11) in the paper

    """
    region term
    """

    C_1 = np.ones((1,100, 100,1))
    C_2 = np.zeros((1,100, 100,1))
    region_in = K.abs(K.mean( ignore_mask * y_pred * ((y_true - C_1)**2)  )) # equ.(12) in the paper
    region_out = K.abs(K.mean( ignore_mask* (1-y_pred) * ((y_true - C_2)**2)  )) # equ.(12) in the paper
    
    

    lambdaP = 1 # lambda parameter could be various.
    mu = 1# mu parameter could be various.
    # print(1 - ignore_mask)
    loss_ce = K.mean((K.binary_crossentropy(y_true, y_pred)* (1-ignore_mask)), axis = (1,2,3))
    all_c = lenth + region_in + region_out + loss_ce 
    lenth_c = all_c/(lenth + 0.0000001)
    regi_c =  all_c/(region_in + 0.0000001)
    rego_c = all_c/(region_out + 0.0000001)
    ce_c = all_c/ (loss_ce  + 0.0000001)
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
        loss = 0.5 * lenth_c * lenth  + lambdaP * ( regi_c * region_in +  rego_c * region_out) + ce_c * loss_ce
    # print(loss, "loss")
    return  loss



def AAAID_Loss(y_true, y_pred): 
    # print(y_true)
    # print(y_pred.shape)
    ignore_mask = K.expand_dims(y_true[:, :, :, 4] , axis=-1)
    ignore_mask = K.cast(ignore_mask, 'float32')
    # # print( K.sum(ignore_mask))
    if (K.sum(ignore_mask) == 0):
        flag_gt = 1
    else:
        flag_gt = 0
    y_pred = K.cast(y_pred, 'float32')
    y_true = K.cast(y_true[:, :, :, 0:4], 'float32')
    # y_true = K.expand_dims(y_true, axis=-1)
    
    # print(y_pred.shape, "pred")
    # print(y_true.shape, "true")
    x = y_pred[:,1:,:,:] - y_pred[:,:-1,:,:]    
    y = y_pred[:,:,1:,:] - y_pred[:,:,:-1,:] 

    delta_x = x[:,1:,:-2,:]**2
    # print(x.shape)
    
    delta_y = y[:,:-2,1:,:]**2
    # print(delta_x.shape)
    
    delta_u = K.abs(delta_x + delta_y) * ignore_mask [:,1:149, 1:149, :]
    lenth = K.mean(K.sqrt(delta_u + 0.00000001) ) # equ.(11) in the paper

    """
    region term
    """
    # print("ytrueeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
    # print(y_true.shape)
    # print(ignore_mask.shape)
    
    C_1 = np.ones((1,150, 150,4))
    C_2 = np.zeros((1,150, 150,4))
    region_in = K.abs(K.mean( ignore_mask * y_pred * ((y_true - C_1)**2)  )) # equ.(12) in the paper
    region_out = K.abs(K.mean( ignore_mask* (1-y_pred) * ((y_true - C_2)**2)  )) # equ.(12) in the paper
    
    

    lambdaP = 1 # lambda parameter could be various.
    mu = 1# mu parameter could be various.
    # print(1 - ignore_mask)
    loss_ce = K.mean((K.categorical_crossentropy(y_true, y_pred)* (1-ignore_mask)), axis = (1,2,3))
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
        loss = 0.5* lenth  + lambdaP * (  region_in +  5 * region_out) + loss_ce
    # print(loss, "loss")
    return  loss

    
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

	C_1 = np.ones((256, 256))
	C_2 = np.zeros((256, 256))

	region_in = K.abs(K.mean( y_pred[:,0,:,:] * ((y_true[:,0,:,:] - C_1)**2) ) ) # equ.(12) in the paper
	region_out = K.abs(K.mean( (1-y_pred[:,0,:,:]) * ((y_true[:,0,:,:] - C_2)**2) )) # equ.(12) in the paper

	lambdaP = 1 # lambda parameter could be various.
	mu = 1 # mu parameter could be various.
	
	return lenth + lambdaP * (mu * region_in + region_out) 
    
crop_size = 100
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
        
    
def mvn(tensor):
    '''Performs per-channel spatial mean-variance normalization.'''
    epsilon = 1e-6
    mean = K.mean(tensor, axis=(1,2), keepdims=True)
    std = K.std(tensor, axis=(1,2), keepdims=True)
    mvn = (tensor - mean) / (std + epsilon)
    
    return mvn


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
# def SSL (y_true , y_pred):
    # y_true_np = y_true.numpy()
    # y_pred_np = y_pred.numpy()    
    # index_reliable = np.ones ((1,100, 100,1), dtype = bool)
    # # index_reliable [(y_true_np == -1)] = False 
    # t = y_true[index_reliable]
    # p = y_pred[index_reliable]
    # t = tf.convert_to_tensor (t)
    # p = tf.convert_to_tensor (p)    
    # res= CCE(t,p)
    # #print (res)
    # return res/len (p)  
def weighted_CE (y_true , y_pred):
    y_true_np = y_true.numpy()
    y_pred_np = y_pred.numpy()    
    #y_pred_m is the y_pred with 2 classes (LV, BG)
    # y_pred_m = np.zeros (shape = (1,100,100,2), dtype = np.float32)
    # y_pred_m [:,:,:,0] = y_pred_np[:,:,:,0]  
    # y_pred_m [:,:,:,1] = 1 - y_pred_np[:,:,:,0]
    # y_pred_m = tf.convert_to_tensor(y_pred_m)
    # scale preds so that the class probas of each sample sum to 1
    # y_pred_m /= tf.reduce_sum(y_pred_m,
                            # reduction_indices=len(y_pred.get_shape()) - 1,
                            # keep_dims=True)
    # manual computation of crossentropy
    epsilon = K.epsilon()
    y_pred_m = tf.clip_by_value(y_pred_np, epsilon, 1. - epsilon)
    axes = (0,1,2)
    c_1 =  - K.sum( y_true_np[:,:,:,0] * tf.math.log(y_pred_np[:,:,:,0])) / K.sum(y_true_np [:,:,:,0])
    print(c_1)
    c_2 =  - K.sum( y_true_np[:,:,:,1] * tf.math.log(y_pred_np[:,:,:,1])) / K.sum(y_true_np [:,:,:,1])
    print(c_2)
    return c_1 + c_2
import tensorflow as tf

def masked_cross_entropy(y_true, y_pred):
    # Ignore pixels of the mask having value -1
    mask = tf.not_equal(y_true, -1)

    # Perform cross entropy only on the remaining pixels
    loss = tf.compat.v1.losses.sparse_categorical_crossentropy(y_true[mask], y_pred[mask])

    return loss
    
                               
def weak_loss (epoch):                               
    def weak_annotation (y_true , y_pred ):

        y_true_np = y_true.numpy()
        # print("this is loss")
        # print(y_pred.shape)
        y_true_1 = np.zeros((1,100,100,1), dtype = 'float32')
        y_true_2 = np.zeros((1,100,100,1), dtype = 'float32')
        y_true_1 [y_true_np == 1] = 1
        y_true_2 [y_true_np == 0] = 1
        if(K.sum((y_true_np==-1).astype(int))>0):
            SUP = 0
            # sample_weight = 0.2
        else:
            SUP = 1
            # sample_weight = 1
        # if (epoch<10) : 
            # sample_weight = SUP
        # else:
        sample_weight = (epoch/100 * (1-SUP)) + SUP
        epsilon = K.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        binary_entropy = -(K.sum((y_true_1*tf.math.log(y_pred)))/K.sum(y_true_1)) - (K.sum(y_true_2 * (tf.math.log(1- y_pred))) /K.sum(y_true_2))
        # print (binary_entropy)
        return binary_entropy * sample_weight
    return weak_annotation
# def weighted(logits, labels) :
    # '''
    # Weighted cross entropy loss, with a weight per class
    # :param logits: Network output before softmax
    # :param labels: Ground truth masks
    # :param class_weights: A list of the weights for each class
    # :return: weighted cross entropy loss
    # '''
    # class_weights = [0.1, 0.3, 0.3, 0.3]
    # n_class = len(class_weights)

    # flat_logits = tf.reshape(logits, [-1, n_class])
    # flat_labels = tf.reshape(labels, [-1, n_class])

    # class_weights = tf.constant(np.array(class_weights, dtype=np.float32))

    # weight_map = tf.multiply(flat_labels, class_weights)
    # weight_map = tf.reduce_sum(weight_map, axis=1)

    # #loss_map = tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits, labels=flat_labels)
    # loss_map = tf.nn.softmax_cross_entropy_with_logits_v2(logits=flat_logits, labels=flat_labels)
    # weighted_loss = tf.multiply(loss_map, weight_map)

    # loss = tf.reduce_mean(weighted_loss)

    # return loss


def dice1(y_true, y_pred, smooth=1e-7):

    intersection = tf.reduce_sum(y_true[..., 0] * y_pred[..., 0])
    union = tf.reduce_sum(y_true[..., 0] + y_pred[..., 0])

    dice = (2. * intersection + smooth) / (union + smooth)

    return dice

def dice2(y_true, y_pred, smooth=1e-7):

    intersection = tf.reduce_sum(y_true[..., 1] * y_pred[..., 1])
    union = tf.reduce_sum(y_true[..., 1] + y_pred[..., 1])

    dice = (2. * intersection + smooth) / (union + smooth)

    return dice

def dice3(y_true, y_pred, smooth=1e-7):

    intersection = tf.reduce_sum(y_true[..., 2] * y_pred[..., 2])
    union = tf.reduce_sum(y_true[..., 2] + y_pred[..., 2])

    dice = (2. * intersection + smooth) / (union + smooth)

    return dice

def dice4(y_true, y_pred, smooth=1e-7):

    intersection = tf.reduce_sum(y_true[..., 3] * y_pred[..., 3])
    union = tf.reduce_sum(y_true[..., 3] + y_pred[..., 3])

    dice = (2. * intersection + smooth) / (union + smooth)

    return dice


def jac2(y_true, y_pred, smooth=0.0):
    '''Average jaccard coefficient per batch.'''
    axes = (1,2)
    intersection = K.sum(y_true[..., 1] * y_pred[..., 1], axis=axes)
    union = K.sum(y_true[..., 1], axis=axes) + K.sum(y_pred[..., 1], axis=axes) - intersection
    return K.mean( (intersection + smooth) / (union + smooth), axis=0)

def jac3(y_true, y_pred, smooth=0.0):
    '''Average jaccard coefficient per batch.'''
    axes = (1,2)
    intersection = K.sum(y_true[..., 2] * y_pred[..., 2], axis=axes)
    union = K.sum(y_true[..., 2], axis=axes) + K.sum(y_pred[..., 2], axis=axes) - intersection
    return K.mean( (intersection + smooth) / (union + smooth), axis=0)

def jac4(y_true, y_pred, smooth=0.0):
    '''Average jaccard coefficient per batch.'''
    axes = (1,2)
    intersection = K.sum(y_true[..., 3] * y_pred[..., 3], axis=axes)
    union = K.sum(y_true[..., 3], axis=axes) + K.sum(y_pred[..., 3], axis=axes) - intersection
    return K.mean( (intersection + smooth) / (union + smooth), axis=0)    

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


def fcn_model(input_shape, num_classes, weights = None):
    ''' "Skip" FCN architecture similar to Long et al., 2015
    https://arxiv.org/abs/1411.4038
    '''
    if num_classes == 2:
        num_classes = 1
        loss = dice_coef_loss
        #loss = weighted
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
    print('**********upsampleeeeieieie3*****************')                
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


if __name__ == '__main__':
    model = fcn_model((100, 100, 1), 2, weights=None)