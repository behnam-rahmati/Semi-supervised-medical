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
import torch
# from new14.py import*
# Check if CUDA is available
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
# # # # # test_ctrs = list(map_all_contours(TEST_CONTOUR_PATH, contour_type, shuffle=True))
# # # # # # print(len(test_ctrs))
# # # # # # print("aaa")
# # # # # indices_test = [random.randint(0, len(test_ctrs)-1) for i in range(50)]
# # # # # save_results('test_ctrs', 'indices.txt',indices_test)


# # # # # test_ctrs_selected = get_elements(test_ctrs,indices_test)
# # # # # train_ctrs = list(map_all_contours(TRAIN_CONTOUR_PATH, contour_type, shuffle=True))
# # # # # # print(len(train_ctrs))
# # # # # # print("bbb")


# # # # # img_all, mask_all = export_all_contours(train_ctrs,
                                            # # # # # TRAIN_IMG_PATH,
                                            # # # # # crop_size=crop_size)  
# # # # # print(img_all.shape)
# # # # # print("AAAAAAAAAFEFEFEFEFFDGFGBBFDGFDFBRFAVBNDCVNREVFGFNBNRTASHWED%YHRTUU#RYEUTRYTEEJ%YRZU%#BUJTUUYTGHJT")

                                       
# # # # # img_dev, mask_dev = export_all_contours(test_ctrs_selected,
                                        # # # # # TEST_IMG_PATH,
                                        # # # # # crop_size=crop_size)
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

    for num_ctrs in [20]:
        train_indices = []
        train_indices = random.sample(range(len(indices_all)-1), num_ctrs)
        indices_selected = train_indices
                
        img_train = img_all[train_indices]
        mask_train = mask_all[train_indices]
        
        unlabeled_indices = [i2 for i2 in range(len(indices_all)) if i2 not in train_indices]
        
       
        # print(len(unlabeled_indices), "len unlabeled indices")
        # print(len(indices_selected), "len labeled indices")

        
     
        # train_ctrs_selected = get_elements(train_ctrs,train_indices)
        # img_train, mask_train = export_all_contours(train_ctrs_selected,
                                            # TRAIN_IMG_PATH,
                                            # crop_size=crop_size) 
                                       
        # print(np.max(img_train[0]))                          
        # # # # # # # # # # # # # # # # # supervised only  # # # # # # # # # # # # # # # # #                                    
        # print("hihihihihibyebyebyebyebye")
        # print(img_train.shape)
        # print(mask_train.shape)
        
        # print(img_dev.shape)
        # print(mask_dev.shape)
        ###########################################################################
        
        
        save_folder = 'AAAI\\suponly\\'+ str(num_ctrs)+'\\weights\\'+str(itr)
        restmp, epoch_tmp = train_module(contour_type, save_folder, img_train, mask_train, img_dev, mask_dev, loss = 'CE' )

        # train_module(contour_type, save_folder, img_train, mask_train, img_dev, mask_dev, loss = 'CE' )

        save_results ('AAAI\\suponly\\'+ str(num_ctrs)+'\\train_ctrs','ctrs'+str(itr)+'.txt', indices_selected)    
        # print(restmp)
        # print(epoch_tmp)
        save_results ('AAAI\\suponly\\'+ str(num_ctrs)+'\\results','indices'+str(itr)+'.txt', [restmp])
        

        
        
        
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
        for idx, init_mask in enumerate (pred_masks_all):
            img = img_all[idx]
            res1,res2,res3 = chanvese(I = img.squeeze(), init_mask = init_mask[:,:,0], max_its = 100, display = False, alpha= 1 , shape_w = 0.5) 
            mask_deformable = np.expand_dims(res1, axis=2)            
            mask_deformable = mask_deformable.astype('uint8')
            masks_deformables.append(mask_deformable)
        masks_deformables =  np.array(masks_deformables)     
                
        # ##########################################################################################################################
        # # # # ## # # # # # # # # # # # # apply pseudo-labeling # # # # # # # # # # # # # # # # # 

        
        img_PL = np.copy(img_all) 
        mask_PL = np.copy(mask_all)
        mask_deformable_ = np.copy(mask_all)
        mask_PL[unlabeled_indices] = pred_masks_all [unlabeled_indices]
        mask_deformable_  [unlabeled_indices] = masks_deformables [unlabeled_indices]
        preds_all [indices_selected] = mask_all [indices_selected]
        # pred_PL = preds_all [unlabeled_indices]
        # mask_PL [mask_PL>=0.5] = 1
        # mask_PL [mask_PL<0.5] = 0
        #################################################################################
        # print(img_train.shape)
        # print(mask_train.shape)
        # print(img_PL.shape)
        # print(img_dev.shape)
        # print(mask_dev.shape)
        # print("all the shapes are shown here VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV")

        # ########################################### only deformable ###############################################

        


        # Plot the reliable_mask





                
        save_folder = 'AAAI\\only deformable\\'+ str(num_ctrs) + '\\nofilter\\weights\\'+str(itr)              
        # restmp, epoch_tmp = train_module(contour_type, save_folder, img_all, mask_deformable_, img_dev, mask_dev, loss = 'CE' )
            
        # mean_teacher(img_train, mask_train, img_PL, img_dev, mask_dev, 40, alpha = 0.99)
        
        #########################################PL############################################################
        
        save_folder = 'AAAI\\PL\\'+ str(num_ctrs) + '\\nofilter\\weights\\'+str(itr)
        
        restmp, epoch_tmp = train_module(contour_type, save_folder, img_all, mask_PL, img_dev, mask_dev, loss = 'CE' )
        
        # save_results ('AAAI\\PL\\'+ str(num_ctrs)+'\\nofilter\\resuls','indices'+str(itr)+'.txt', [restmp])  


        ######################################## our method ######################################
       
        for thresh1 in [  0.0001, 0.001, 0.01, 0.1]:
            reliable_mask = np.ones_like(preds_all, dtype='uint16') 
            # reliable_tmp = np.ones_like(preds_all, dtype='uint16')
            print("this is thresholdAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
            print(thresh1)
            reliable_tmp = np.logical_or(preds_all < thresh1, preds_all > 1 - thresh1)
            reliable_mask[reliable_tmp] = 0

            # print(reliable_tmp.shape)
            # for kk in range (4):

                
                # # print(preds_all[20])

                
                
                
                # # Plot the reliable_mask
                # plt.subplot(1, 3, 1)
                # plt.imshow(reliable_mask[kk].squeeze(), cmap='gray')
                # plt.title('reliable_mask')

                # # Plot the additional_data1
                # plt.subplot(1, 3, 2)
                # plt.imshow(img_all [kk], cmap='gray')
                # plt.title('additional_data1')

                # # Plot the additional_data2
                # plt.subplot(1, 3, 3)
                # plt.imshow(mask_all[kk], cmap='gray')
                # plt.title('additional_data2')

                # plt.show()

            
                                  
            # ########################################### deformable co-training ##########################################
                            
            final_deformable = np.copy(mask_deformable_)
            final_deformable[reliable_mask == 0] = mask_PL [reliable_mask ==0]
            

                       
            
            
            save_folder = 'AAAI\\co-training deformable\\'+ str(num_ctrs) + '\\nofilter\\weights\\'+str(itr)+ "\\" + str(thresh1)                
            # restmp, epoch_tmp = train_module(contour_type, save_folder, img_all, final_deformable, img_dev, mask_dev, loss = 'CE' )


        
            # sum_result = K.sum(reliable_mask, axis=(1, 2))
            # print(sum_result[kk])             
            print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACCCCCCCCCCCCCCCCCCCCCCCCCCCCAAAAAAAAAAAAAAAAAAAAAAAAAADDDDDDDDDDDDDDDDDDDDDDDDDDDDDOOOOOOOOOOOOOOOOOOOOOOOO")
            # print(reliable_mask.shape)

            save_folder = 'AAAI\\AC\\'+ str(num_ctrs) + '\\nofilter\\weights\\'+str(itr)+ "\\" + str(thresh1)

            mask_AC = np.concatenate((mask_PL, reliable_mask), axis=3)


            # # mask_dev_AC = np.concatenate((mask_dev, np.zeros_like(mask_dev)), axis=3)

            # # print("this is mask ac size")
            # # print(mask_AC.shape)
            restmp, epoch_tmp = train_module(contour_type, save_folder, img_all, mask_AC, img_dev, mask_dev, loss = 'AC' )

            # # print(restmp)
            # # save_results ('AAAI\\AC\\'+ str(num_ctrs)+'\\nofilter\\resuls','indices'+str(itr)+'.txt', [restmp])  
            # ######################################## AC2 deformable ######################################
            # print(reliable_mask.shape)

            save_folder = 'AAAI\\AC2\\'+ str(num_ctrs) + '\\nofilter\\weights\\'+str(itr)+ "\\" + str(thresh1)
            # # # mask_AC = pred_masks_all [unlabeled_indices]
            # # # mask_AC [mask_AC>=0.5] = 1
            # # # mask_AC [mask_AC<0.5] = 0
            
            # # mask_AC = np.concatenate((mask_PL, reliable_mask), axis=3)


            # # # mask_dev_AC = np.concatenate((mask_dev, np.zeros_like(mask_dev)), axis=3)

            # # # print("this is mask ac size")
            # # # print(mask_AC.shape)
            restmp, epoch_tmp = train_module(contour_type, save_folder, img_all, mask_AC, img_dev, mask_dev, loss = 'AC2' )            
            

            
      
    # # # save_results ('PL2\\'+ str(num_ctrs)+'\\filter0.9999\\results','indices'+str(itr)+'.txt', [PLtmp])    
    

    
                  
        

    