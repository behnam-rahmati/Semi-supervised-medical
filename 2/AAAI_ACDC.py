# from train_lung import *
import os, fnmatch, sys
import random
import time
import os
import cv2
import numpy as np
from fcn_mode import*
print("g")
from ACDC_util import *
from chan_vese_1 import chanvese
import matplotlib.pyplot as plt
from skimage import exposure, measure
import tensorflow as tf
from skimage.transform import resize
from matplotlib.colors import ListedColormap
from train_lung import *
print("num gpu available:",len(tf.config.experimental.list_physical_devices('GPU')))
import torch
from helpers import*
# from U_net import *
# from new14.py import*
# Check if CUDA is available
from keras import backend as K
gpus = tf.config.experimental.list_physical_devices('GPU')
seed = 1234
np.random.seed(seed)
crop_size = 150
input_shape = (crop_size, crop_size, 1)
num_classes = 4


import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(threshold=np.inf)


def calculate_entropy(probabilities):
    # Avoid log(0) by setting very small values to a small positive number
    epsilon = 1e-7
    log_probs = np.log(probabilities + epsilon)
    entropy = -np.sum(probabilities * log_probs, axis=-1)
    return entropy   

def normalize_image(image):
    # Normalize pixel values to range [0, 1]
    min_val = np.min(image)
    max_val = np.max(image)
    return (image - min_val) / (max_val - min_val)
    


base_directory = r"C:\Users\Motamed-Lab\Desktop\ACDC\Resources"
gt_files, corresponding_files = find_gt_and_corresponding_files(base_directory)

# if gt_files:
    # print("Found the following GT files:")
    # for file in gt_files:
        # print(file)
# else:
    # print("No GT files found.")

# if corresponding_files:
    # print("\nFound corresponding files:")
    # for folder, files in corresponding_files.items():
        # print(f"Folder: {folder}")
        # for file in files:
            # print(file)
        # print()
# else:
    # print("\nNo corresponding files found.")

        
# Path to the .nii file
g= (75/256, 245/256, 66/256)
rgb_color = (39/255, 153/255, 230/255)

colors =  ['black', rgb_color, g, 'red']


# Define the corresponding values (0, 1, 2, 3)
values = [0, 1, 2, 3]

# Create the custom color map using ListedColormap
cmap_1 = ListedColormap(colors)


all_images = []
all_masks = []
crop_size = 150
for path, (folder, files) in zip(gt_files, corresponding_files.items()):
    # print(path)
    # print(files)

    image_file_path = files[0]

    # Path to the .nii file for ground truth data
    ground_truth_file_path = path

    # # Load the .nii files
    image_nii_data = nib.load(image_file_path)
    image_data = image_nii_data.get_fdata()

    # print(image_data.shape)
    ground_truth_nii_data = nib.load(ground_truth_file_path)
    ground_truth_data = ground_truth_nii_data.get_fdata()
    # print("shape     ................  shape")
    # print(ground_truth_data.shape)
    # plt.imshow(ground_truth_data[:,:,2], cmap='gray')
    # plt.show()
    # Iterate over the desired slices

    image_data = image_data.astype(int)  # You can use np.uint16, np.int32, etc., depending on your data type
    for slice_index in range(image_data.shape[2]):
        # Extract the data for the current slice
        # image_data = resize(image_data, ( crop_size, crop_size,image_data.shape[2]),
                                # mode='reflect', preserve_range=True)
                                
        image_slice_data = image_data[:, :, slice_index]
        image_slice_data = center_crop(image_slice_data, crop_size = 150)
        image_slice_data = normalize_image(image_slice_data)
        
        # ground_truth_data = np.round(resize(ground_truth_data, ( crop_size, crop_size,ground_truth_data.shape[2]),
                                        # mode='reflect', preserve_range=True))

        # ground_truth_data = ground_truth_data.astype(int)  # You can use np.uint16, np.int32, etc., depending on your data type
                                        
        ground_truth_slice_data = ground_truth_data[:, :, slice_index]
        ground_truth_slice_data = center_crop(ground_truth_slice_data, crop_size = 150)

        # print((ground_truth_slice_data))
        # ground_truth_slice_data = np.array(ground_truth_slice_data, dtype=np.uint8)
        # print(ground_truth_slice_data)
        all_images.append(image_slice_data)        
        all_masks.append(ground_truth_slice_data)
        # Plot the image data
        plt.subplot(1, 2, 1)
        plt.imshow(image_slice_data, cmap='gray')
        plt.axis('off')
        plt.title('Image - Slice: {}'.format(slice_index))

        # Plot the ground truth data
        plt.subplot(1, 2, 2)
        plt.imshow(ground_truth_slice_data, cmap= cmap_1)
        plt.axis('off')
        plt.title('Ground Truth - Slice: {}'.format(slice_index))
        # print((ground_truth_slice_data))

        # plt.show()
all_images = np.array(all_images)
all_masks = np.array(all_masks, dtype=np.uint8)
# print(all_masks==3)
# all_masks[all_masks==1]=0
# all_masks[all_masks==2]=1
all_images = resize(all_images, (all_images.shape[0], crop_size, crop_size),
                        mode='reflect', preserve_range=True)
                        
all_masks = resize(all_masks, (all_masks.shape[0], crop_size, crop_size),
                        mode='reflect', preserve_range=True)

all_masks = np.array(all_masks, dtype=np.uint8)
# all_masks[all_masks!=1]= 0                       
print("***********************************")

all_images = np.expand_dims(all_images, axis = 3)
all_masks_one_hot = tf.one_hot(all_masks, depth=num_classes)
all_masks = all_masks_one_hot.numpy()
all_images = all_images.squeeze()
all_masks = all_masks.squeeze()
print(all_images.shape)
print(all_masks.shape)



# for slice_index in range(all_images.shape[2]):
    # # Extract the data for the current slice

    # plt.subplot(1, 2, 1)
    # plt.imshow(all_images[slice_index].squeeze(), cmap='gray')
    # plt.axis('off')
    # plt.title('Image - Slice: {}'.format(slice_index))

    # # Plot the ground truth data
    # plt.subplot(1, 2, 2)
    # plt.imshow(all_masks[slice_index].squeeze(), cmap='gray')
    # plt.axis('off')
    # plt.title('Ground Truth - Slice: {}'.format(slice_index))
    # # print((ground_truth_slice_data))

    # plt.show()

        
num_samples = all_images.shape[0]
perm_index = np.random.permutation(num_samples)

# Shuffle the images and masks using the permutation index
all_images = all_images[perm_index]
all_masks = all_masks[perm_index]

    

for itr in range (0,1):
    contour_type = 'A'
# # # # # # # # # # # preparing the labeled and unlabeled train data # # # # # # # # # #

    for num_ctrs in [30, 40, 50, 75, 100]:


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
        # mask_train = np.expand_dims(mask_train, axis = -1)
        # mask_dev = np.expand_dims(mask_dev, axis = -1)

        print(img_train.shape)
        print(mask_train.shape)  
        print(img_dev.shape)
        print("aaaaaaaaa")
        print(img_dev.shape)
        print(mask_dev.shape)
        print("this is image train shape apfohwefpowiehfpwoeifhweofihrepvouhWEFPOHOFUH")     
                                               
        ###########################################################################        
        
        save_folder = 'AAAI_ACDC\\suponly\\'+ str(num_ctrs)+'\\weights\\'+str(itr)
        # print(mask_train)
        restmp, epoch_tmp = train_module("A", save_folder, img_train, mask_train, img_dev, mask_dev, loss = 'CE', lr_input = 0.05 )
        # # train_module(contour_type, save_folder, img_train, mask_train, img_dev, mask_dev, loss = 'CE' )
        # print(abc)
        # save_results ('AAAI_lung\\suponly\\'+ str(num_ctrs)+'\\train_ctrs','ctrs'+str(itr)+'.txt', indices_selected)    
        # print(restmp)
        # print(epoch_tmp)
        # save_results ('AAAI_lung\\suponly\\'+ str(num_ctrs)+'\\results','indices'+str(itr)+'.txt', [restmp])
        

        
        
        
        suponly_weights = os.path.join(save_folder,'model_' + contour_type + '.h5')    
        # model1 = unet(input_shape, pretrained_weights= suponly_weights)
        model1 = fcn_model(input_shape, num_classes, weights = suponly_weights)  
        
        pred_masks_all = model1.predict(img_all, batch_size = 8, verbose = 1) 
        # print("a")
        # for j in range(100):
            # print(pred_masks_all[j,64,64,:])
        preds_all = np.copy(pred_masks_all)
        pred_masks_all[pred_masks_all>=0.5] = 1
        pred_masks_all[pred_masks_all<0.5] = 0
        print(pred_masks_all.shape)
        print("ARRRRRRRRRRRRRRRRRRR")
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

        # mask_all  = np.expand_dims(mask_all, axis = -1)       
        img_PL = np.copy(img_all) 
        mask_PL = np.copy(mask_all)
        # mask_deformable_ = np.copy(mask_all)
        mask_PL[unlabeled_indices] = pred_masks_all [unlabeled_indices]
        # mask_deformable_  [unlabeled_indices] = masks_deformables [unlabeled_indices]
        preds_all [indices_selected] = mask_all [indices_selected]
        # print(preds_all)
        single_mask = np.argmax(pred_masks_all, axis=-1)
        gt_mask = np.argmax(mask_all, axis=-1)

    
    
        

                
        save_folder = 'AAAI_lung\\only deformable\\'+ str(num_ctrs) + '\\nofilter\\weights\\'+str(itr)              
        # # restmp, epoch_tmp = train_module(contour_type, save_folder, img_all, mask_deformable_, img_dev, mask_dev, loss = 'CE' )
            
        # # mean_teacher(img_train, mask_train, img_PL, img_dev, mask_dev, 40, alpha = 0.99)
        

        
        # # save_results ('AAAI_lung\\PL\\'+ str(num_ctrs)+'\\nofilter\\resuls','indices'+str(itr)+'.txt', [restmp])  


        # ######################################## our method ######################################

            
        for thresh_entropy in [ 0.001]:
        
        
            self_entropy = calculate_entropy(preds_all)

            reliable_mask = np.ones_like(preds_all[..., 0], dtype='uint16')  # Initialize the mask with ones
            reliable_tmp = self_entropy < thresh_entropy
            reliable_mask[reliable_tmp] = 0
            reliable_mask = np.expand_dims(reliable_mask, axis=-1)            
            # print(reliable_mask.shape)
            # print("rrrrrrrr")


            # reliable_mask2 = np.ones_like(preds_all[..., 1], dtype='uint16') 
            # reliable_tmp2 = np.ones_like(preds_all[..., 1], dtype='uint16')
            # # print("this is thresholdAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
            # # print(thresh1)
            # reliable_tmp2 = np.logical_or(preds_all[..., 1] < thresh1, preds_all[..., 1] > 1 - thresh1)
            # reliable_mask2[reliable_tmp2] = 0


            
            # for i in range (1, img_all.shape[0]):
                # plt.subplot(3, 2, 1)
                # plt.imshow(img_all[i] , cmap='gray')
                # plt.subplot(3, 2, 2)
                # plt.imshow(single_mask[i] , cmap=cmap_1)
                # plt.subplot(3, 2, 3)
                # plt.imshow(reliable_mask[i].squeeze() , cmap='gray')
                # plt.subplot(3, 2, 4)
                # plt.imshow(gt_mask[i] , cmap = cmap_1)
                # # plt.subplot(3, 2, 5)
                # # plt.imshow(reliable_mask2[i].squeeze() , cmap = 'gray')
                # plt.title(i)
                
                # plt.show()
            # # reliable_mask = np.ones_like(preds_all, dtype='uint16') 
            # # # # reliable_tmp = np.ones_like(preds_all, dtype='uint16')
            # # # print("this is thresholdAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
            # # # print(thresh1)
            # # reliable_tmp = np.logical_or(preds_all < thresh1, preds_all > 1 - thresh1)
            # # reliable_mask[reliable_tmp] = 0
                                 
            # # # ########################################### deformable co-training ##########################################
                            
            # # final_deformable = np.copy(mask_deformable_)
            # # final_deformable[reliable_mask == 0] = mask_PL [reliable_mask ==0]

            # # save_folder = 'AAAI_lung\\co-training deformable\\'+ str(num_ctrs) + '\\nofilter\\weights\\'+str(itr)+ "\\" + str(thresh1)                

            # # save_folder = 'AAAI_lung\\AC\\'+ str(num_ctrs) + '\\nofilter\\weights\\'+str(itr)+ "\\" + str(thresh1)

            mask_AC = np.concatenate((mask_PL, reliable_mask), axis=3)

            # # restmp, epoch_tmp = train_module(contour_type, save_folder, img_all, mask_AC, img_dev, mask_dev, loss = 'AC' )

            for lr_value in [0.003]:
                print(lr_value)
                print(mask_AC.shape)
                save_folder = 'AAAI_lung\\AC3\\'+ str(num_ctrs) + '\\nofilter\\weights\\'+str(itr)+ "\\" + str(thresh_entropy) + "\\" + str(lr_value)

                # restmp, epoch_tmp = train_module(contour_type, save_folder, img_all, mask_AC, img_dev, mask_dev, loss = 'AC3', lr_input = lr_value )            

        # # #########################################PL############################################################
    
            save_folder = 'AAAI_ACDC\\PLA\\'+ str(num_ctrs) + '\\nofilter\\weights\\'+str(itr)+'\\' +str(lr_value)
            
            restmp, epoch_tmp = train_module(contour_type, save_folder, img_all, mask_PL, img_dev, mask_dev, loss = 'CE', lr_input = lr_value )
            
        # ########################################AC############################################################
    
        # save_folder = 'AAAI_lung\\AC\\'+ str(num_ctrs) + '\\nofilter\\weights\\'+str(itr)
        
        # restmp, epoch_tmp = train_module(contour_type, save_folder, img_all, mask_AC, img_dev, mask_dev, loss = 'AC', lr_input = 0.01 )
            

            
    


    
                  
        

    