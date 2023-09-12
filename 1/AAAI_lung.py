from train_lung import *
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
from skimage.transform import resize

print("num gpu available:",len(tf.config.experimental.list_physical_devices('GPU')))
import torch
from U_net import *
# from new14.py import*
# Check if CUDA is available
from keras import backend as K
gpus = tf.config.experimental.list_physical_devices('GPU')

seed = 1234
np.random.seed(seed)
crop_size = 128
input_shape = (crop_size, crop_size, 1)
num_classes = 2


import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(threshold=np.inf)


def normalize_image(image):
    # Normalize pixel values to range [0, 1]
    min_val = np.min(image)
    max_val = np.max(image)
    return (image - min_val) / (max_val - min_val)
    
    
# Path to the .nii file
all_images = []
all_masks = []
for number in range (1,11, 1):
    case_number = f"{number:03}"
    # Path to the .nii file for image data
    image_file_path = "C://Users//Motamed-Lab//Desktop//Lung//COVID-19-CT-Seg_20cases//coronacases_" +  case_number + ".nii//coronacases_org_" + case_number + ".nii"

    # Path to the .nii file for ground truth data
    ground_truth_file_path = "C://Users//Motamed-Lab//Desktop//Lung//Lung_Mask//coronacases_" + case_number + ".nii//coronacases_" + case_number + ".nii"

    # Load the .nii files
    image_nii_data = nib.load(image_file_path)
    image_data = image_nii_data.get_fdata()

    print(image_data.shape)
    ground_truth_nii_data = nib.load(ground_truth_file_path)
    ground_truth_data = ground_truth_nii_data.get_fdata()

    # Iterate over the desired slices
    start_slice = 30
    end_slice = image_data.shape[2]
    skip_slices = 5

    for slice_index in range(start_slice, end_slice, skip_slices):
        # Extract the data for the current slice
        image_slice_data = image_data[:, :, slice_index]
        image_slice_data = normalize_image(image_slice_data)
        ground_truth_slice_data = ground_truth_data[:, :, slice_index]
        all_images.append(image_slice_data)        
        all_masks.append(ground_truth_slice_data)
        # Plot the image data
        plt.subplot(1, 2, 1)
        plt.imshow(image_slice_data, cmap='gray')
        plt.axis('off')
        plt.title('Image - Slice: {}'.format(slice_index))

        # Plot the ground truth data
        plt.subplot(1, 2, 2)
        plt.imshow(ground_truth_slice_data, cmap='gray')
        plt.axis('off')
        plt.title('Ground Truth - Slice: {}'.format(slice_index))
        # print((ground_truth_slice_data))

        # plt.show()
all_images = np.array(all_images)
all_masks = np.array(all_masks)
# all_masks[all_masks==1]=0
all_masks[all_masks==2]=1
all_images = resize(all_images, (all_images.shape[0], crop_size, crop_size),
                        mode='reflect', preserve_range=True)
                        
all_masks = resize(all_masks, (all_masks.shape[0], crop_size, crop_size),
                        mode='reflect', preserve_range=True)
                        
print("***********************************")
print(all_images.shape)
print(all_masks.shape)



    
num_samples = all_images.shape[0]
perm_index = np.random.permutation(num_samples)

# Shuffle the images and masks using the permutation index
all_images = all_images[perm_index]
all_masks = all_masks[perm_index]

# for i in range (1, all_images.shape[0]):
    # plt.subplot(1, 2, 1)
    # plt.imshow(all_images[i] , cmap='gray')
    # plt.subplot(1, 2, 2)
    # plt.imshow(all_masks[i] , cmap='gray')
    # plt.title(i)
    
    # plt.show()
    
print(all_images.shape)
print("shape of all images hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh")
for itr in range (0,2):
    contour_type = 'l'
# # # # # # # # # # # preparing the labeled and unlabeled train data # # # # # # # # # #

    for num_ctrs in [300]:


        img_all = all_images[:-50]
        mask_all = all_masks [:-50]

   
        img_dev = all_images [-50:]
        mask_dev = all_masks [-50:]

        train_indices = []
        indices_selected = random.sample(range(img_all.shape[0]-1), num_ctrs)
        

        not_indices = [i2 for i2 in range(img_all.shape[0]-1) if i2 not in indices_selected]
        unlabeled_indices = not_indices

        
        train_indices = indices_selected
     

        img_train = img_all[train_indices]
        mask_train = mask_all[train_indices]
                
        img_train = np.expand_dims(img_train, axis = -1)
        img_dev = np.expand_dims(img_dev, axis = -1)
        mask_train = np.expand_dims(mask_train, axis = -1)
        mask_dev = np.expand_dims(mask_dev, axis = -1)

        print(img_train.shape)
        print(img_dev.shape)        
        print("this is image train shape apfohwefpowiehfpwoeifhweofihrepvouhWEFPOHOFUH")     
                                               
        ###########################################################################        
        
        save_folder = 'AAAI_lung2\\suponly\\'+ str(num_ctrs)+'\\weights\\'+str(itr)
        restmp, epoch_tmp = train_module("l", save_folder, img_train, mask_train, img_dev, mask_dev, loss = 'CE', lr_input = 0.01 )

        # # train_module(contour_type, save_folder, img_train, mask_train, img_dev, mask_dev, loss = 'CE' )

        save_results ('AAAI_lung2\\suponly\\'+ str(num_ctrs)+'\\train_ctrs','ctrs'+str(itr)+'.txt', indices_selected)    
        # print(restmp)
        # print(epoch_tmp)
        save_results ('AAAI_lung2\\suponly\\'+ str(num_ctrs)+'\\results','indices'+str(itr)+'.txt', [restmp])
        

        
        
        
        suponly_weights = os.path.join(save_folder,'model_' + contour_type + '.h5')    
        # model1 = unet(input_shape, pretrained_weights= suponly_weights)
        model1 = fcn_model(input_shape, num_classes, weights = suponly_weights)  
        
        pred_masks_all = model1.predict(img_all, batch_size = 8, verbose = 1) 
        preds_all = np.copy(pred_masks_all)
        pred_masks_all[pred_masks_all>=0.5] = 1
        pred_masks_all[pred_masks_all<0.5] = 0
        
        pred_masks_dev = model1.predict(img_dev, batch_size = 8, verbose = 1) 
        pred_masks_dev[pred_masks_dev>=0.5] = 1
        pred_masks_dev[pred_masks_dev<0.5] = 0
        # masks_deformables = []
        # for idx, init_mask in enumerate (pred_masks_all):
            # img = img_all[idx]
            # res1,res2,res3 = chanvese(I = img.squeeze(), init_mask = init_mask[:,:,0], max_its = 100, display = False, alpha= 1 , shape_w = 0.5) 
            # mask_deformable = np.expand_dims(res1, axis=2)            
            # mask_deformable = mask_deformable.astype('uint8')
            # masks_deformables.append(mask_deformable)
        # masks_deformables =  np.array(masks_deformables)     
                
        # # ##########################################################################################################################
        # # # # # ## # # # # # # # # # # # # apply pseudo-labeling # # # # # # # # # # # # # # # # # 

        mask_all  = np.expand_dims(mask_all, axis = -1)       
        img_PL = np.copy(img_all) 
        mask_PL = np.copy(mask_all)
        # mask_deformable_ = np.copy(mask_all)
        mask_PL[unlabeled_indices] = pred_masks_all [unlabeled_indices]
        # mask_deformable_  [unlabeled_indices] = masks_deformables [unlabeled_indices]
        preds_all [indices_selected] = mask_all [indices_selected]
        # print(preds_all)

        # for i in range (1, img_all.shape[0]):
            # plt.subplot(1, 2, 1)
            # plt.imshow(img_all[i] , cmap='gray')
            # plt.subplot(1, 2, 2)
            # plt.imshow(pred_masks_all[i] , cmap='gray')
            # plt.title(i)
            
            # plt.show()
    
    
        

                
        save_folder = 'AAAI_lung2\\only deformable\\'+ str(num_ctrs) + '\\nofilter\\weights\\'+str(itr)              
        # # restmp, epoch_tmp = train_module(contour_type, save_folder, img_all, mask_deformable_, img_dev, mask_dev, loss = 'CE' )
            
        # # mean_teacher(img_train, mask_train, img_PL, img_dev, mask_dev, 40, alpha = 0.99)
        

        
        # # save_results ('AAAI_lung\\PL\\'+ str(num_ctrs)+'\\nofilter\\resuls','indices'+str(itr)+'.txt', [restmp])  


        # ######################################## our method ######################################

            
        for thresh1 in [ 10, 0.000001,  0.001, 0.01, 0.1]:
            reliable_mask = np.ones_like(preds_all, dtype='uint16') 
            # # reliable_tmp = np.ones_like(preds_all, dtype='uint16')
            # print("this is thresholdAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
            # print(thresh1)
            reliable_tmp = np.logical_or(preds_all < thresh1, preds_all > 1 - thresh1)
            reliable_mask[reliable_tmp] = 0
                                 
            # # ########################################### deformable co-training ##########################################
                            
            # final_deformable = np.copy(mask_deformable_)
            # final_deformable[reliable_mask == 0] = mask_PL [reliable_mask ==0]

            # save_folder = 'AAAI_lung\\co-training deformable\\'+ str(num_ctrs) + '\\nofilter\\weights\\'+str(itr)+ "\\" + str(thresh1)                

            # save_folder = 'AAAI_lung\\AC\\'+ str(num_ctrs) + '\\nofilter\\weights\\'+str(itr)+ "\\" + str(thresh1)

            mask_AC = np.concatenate((mask_PL, reliable_mask), axis=3)

            # restmp, epoch_tmp = train_module(contour_type, save_folder, img_all, mask_AC, img_dev, mask_dev, loss = 'AC' )


            save_folder = 'AAAI_lung2\\AC2\\'+ str(num_ctrs) + '\\nofilter\\weights\\'+str(itr)+ "\\" + str(thresh1)

            restmp, epoch_tmp = train_module(contour_type, save_folder, img_all, mask_AC, img_dev, mask_dev, loss = 'AC2', lr_input = 0.01 )            
            

        # #########################################PL############################################################
    
        save_folder = 'AAAI_lung2\\PL\\'+ str(num_ctrs) + '\\nofilter\\weights\\'+str(itr)
        
        restmp, epoch_tmp = train_module(contour_type, save_folder, img_all, mask_PL, img_dev, mask_dev, loss = 'CE', lr_input = 0.01 )
            
        # ########################################AC############################################################
    
        # save_folder = 'AAAI_lung\\AC\\'+ str(num_ctrs) + '\\nofilter\\weights\\'+str(itr)
        
        # restmp, epoch_tmp = train_module(contour_type, save_folder, img_all, mask_AC, img_dev, mask_dev, loss = 'AC', lr_input = 0.01 )
            

            
    


    
                  
        

    