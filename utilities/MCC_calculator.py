from train_sunnybrook import *
import os, fnmatch, sys
import random
import time
import os
import cv2
import numpy as np
from fcn_mode import *
from chan_vese_1 import chanvese
import matplotlib.pyplot as plt
from skimage import exposure, measure
import tensorflow as tf
print("num gpu available:",len(tf.config.experimental.list_physical_devices('GPU')))
from sklearn.metrics import matthews_corrcoef

# import torch
# from new14.py import*
# Check if CUDA is available
def dice_index(pred, gt):
    """Calculate Dice coefficient."""
    intersection = np.sum(pred * gt)
    return (2. * intersection) / (np.sum(pred) + np.sum(gt))
    
    
from keras import backend as K
gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    print("Code is running on GPU")
else:
    print("Code is running on CPU")
        
seed = 1234
np.random.seed(seed)
#SAX_SERIES = get_SAX_SERIES()

ROOT_PATH = "C:\\Users\\Motamed-Lab\\Desktop\\AAAI\\data"
# VAL_CONTOUR_PATH = "C:\\Users\\Motamed-Lab\\Desktop\\AAAI\\data\\valGT"
# VAL_IMG_PATH = os.path.join(ROOT_PATH,
                   # 'challenge_validation')
# ONLINE_CONTOUR_PATH = "C:\\Users\\Motamed-Lab\\Desktop\\AAAI\\data\\online"
# ONLINE_IMG_PATH = os.path.join(ROOT_PATH,
                   # 'challenge_online')

TRAIN_CONTOUR_PATH = "C:\\Users\\Motamed-Lab\\Desktop\\AAAI\\redct"
TRAIN_IMG_PATH = os.path.join(ROOT_PATH,
                        'challenge_training')
TEST_CONTOUR_PATH = "C:\\Users\\Motamed-Lab\\Desktop\\AAAI\\backup\\test"
TEST_IMG_PATH = os.path.join(ROOT_PATH,
                        'challenge_test') 


ALL_CONTOUR_PATH = "C:\\Users\\Motamed-Lab\\Desktop\\AAAI\\backup2\\all"
ALL_IMG_PATH = os.path.join(ROOT_PATH,
                        'img_all')                        
                        

                        

                        

contour_type = sys.argv[1]
# os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[2]

crop_size = 100
input_shape = (crop_size, crop_size, 1)
num_classes = 2

all_ctrs = list(map_all_contours(ALL_CONTOUR_PATH, contour_type, shuffle=True))   
all_images, all_masks = export_all_contours(all_ctrs, ALL_IMG_PATH, crop_size=crop_size)   
                                     
indices_test = random.sample(range(len(all_ctrs)-1), 50)

indices_all = [i2 for i2 in range(len(all_ctrs)) if i2 not in indices_test]
  
img_dev = all_images[indices_test]
mask_dev = all_masks[indices_test]

img_all = all_images[indices_all]
mask_all = all_masks[indices_all]
for itr in range (0,5):

# # # # # # # # # # # preparing the labeled and unlabeled train data # # # # # # # # # #

    for num_ctrs in [15,22,30,50,100,200,300]:
        print("num_ctrs=", num_ctrs, "itr=", itr)

        save_folder = 'AAAI\\for_revision\\'+ str(num_ctrs)+'\\weights\\'+str(itr)
            
        
        suponly_weights = os.path.join(save_folder,'model_' + contour_type + '.h5')    
        model1 = fcn_model(input_shape, num_classes, weights = suponly_weights)  
        pred_masks_all = model1.predict(img_all, batch_size = 32, verbose = 1) 
        preds_all = np.copy(pred_masks_all)
        pred_masks_all[pred_masks_all>=0.5] = 1
        pred_masks_all[pred_masks_all<0.5] = 0
        
        pred_masks_dev = model1.predict(img_dev, batch_size = 32, verbose = 1) 
        pred_masks_dev[pred_masks_dev>=0.5] = 1
        pred_masks_dev[pred_masks_dev<0.5] = 0
        masks_deformables = []

        # Assuming pred_masks_all, img_all, masks_all are lists or numpy arrays containing your data.
        # Initialize accumulators for averaging
        tp_sum = 0
        tn_sum = 0
        fp_sum = 0
        fn_sum = 0
        mcc_sum = 0

        # Assuming pred_masks_all, img_all, masks_all are lists or numpy arrays containing your data.
        for idx, pred_mask in enumerate(pred_masks_all):
            gt_mask = mask_all[idx]
            
            # Flatten the masks to make them 1D for easier computation
            pred_mask_flat = pred_mask.flatten()
            gt_mask_flat = gt_mask.flatten()
            
            # Calculate TP, TN, FP, FN
            tp = np.sum((pred_mask_flat == 1) & (gt_mask_flat == 1))
            tn = np.sum((pred_mask_flat == 0) & (gt_mask_flat == 0))
            fp = np.sum((pred_mask_flat == 1) & (gt_mask_flat == 0))
            fn = np.sum((pred_mask_flat == 0) & (gt_mask_flat == 1))
            
            # Total number of pixels
            total_pixels = tp + tn + fp + fn
            
            # Convert counts to percentages
            tp_percentage = (tp / total_pixels) * 100
            tn_percentage = (tn / total_pixels) * 100
            fp_percentage = (fp / total_pixels) * 100
            fn_percentage = (fn / total_pixels) * 100
            
            # Accumulate results for averaging
            tp_sum += tp_percentage
            tn_sum += tn_percentage
            fp_sum += fp_percentage
            fn_sum += fn_percentage
            
            # Calculate MCC for the current prediction
            mcc = matthews_corrcoef(gt_mask_flat, pred_mask_flat)
            mcc_sum += mcc
            

        # Calculate averages
        num_images = len(pred_masks_all)
        tp_avg = tp_sum / num_images
        tn_avg = tn_sum / num_images
        fp_avg = fp_sum / num_images
        fn_avg = fn_sum / num_images
        mcc_avg = mcc_sum / num_images

        # Print the averaged results
        print("\nAveraged Results:")
        print(f"Average TP: {tp_avg:.2f}%")
        print(f"Average TN: {tn_avg:.2f}%")
        print(f"Average FP: {fp_avg:.2f}%")
        print(f"Average FN: {fn_avg:.2f}%")
        print(f"Average MCC: {mcc_avg:.4f}")

        # Save the results in a text file
        with open("results.txt", "w") as f:
            f.write(f"Average TP: {tp_avg:.2f}%\n")
            f.write(f"Average TN: {tn_avg:.2f}%\n")
            f.write(f"Average FP: {fp_avg:.2f}%\n")
            f.write(f"Average FN: {fn_avg:.2f}%\n")
            f.write(f"Average MCC: {mcc_avg:.4f}\n")

        print("Averaged results saved in results.txt")
                    