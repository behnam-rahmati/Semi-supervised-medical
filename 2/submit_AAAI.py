# 1- CCLISD
#crop size
#load model
#load weights
#load data
#create predictions
#imshow
# # # # # # # # # 2- CCLISD
# # # # # # # # from matplotlib.colors import ListedColormap

# # # # # # # # from train_lung import *
# # # # # # # # import os, fnmatch, sys
# # # # # # # # import random
# # # # # # # # import time
# # # # # # # # import os
# # # # # # # # import cv2
# # # # # # # # import numpy as np
# # # # # # # # from fcn_mode import *
# # # # # # # # from chan_vese_1 import chanvese
# # # # # # # # import matplotlib.pyplot as plt
# # # # # # # # from skimage import exposure, measure
# # # # # # # # import tensorflow as tf
# # # # # # # # from skimage.transform import resize

# # # # # # # # print("num gpu available:",len(tf.config.experimental.list_physical_devices('GPU')))
# # # # # # # # import torch
# # # # # # # # from U_net import *
# # # # # # # # # from new14.py import*
# # # # # # # # # Check if CUDA is available
# # # # # # # # from keras import backend as K
# # # # # # # # gpus = tf.config.experimental.list_physical_devices('GPU')

# # # # # # # # seed = 1234
# # # # # # # # np.random.seed(seed)
# # # # # # # # crop_size = 128
# # # # # # # # input_shape = (crop_size, crop_size, 1)
# # # # # # # # num_classes = 2

# # # # # # # # import nibabel as nib
# # # # # # # # import matplotlib.pyplot as plt
# # # # # # # # import numpy as np
# # # # # # # # np.set_printoptions(threshold=np.inf)

# # # # # # # # def normalize_image(image):
    # # # # # # # # # Normalize pixel values to range [0, 1]
    # # # # # # # # min_val = np.min(image)
    # # # # # # # # max_val = np.max(image)
    # # # # # # # # return (image - min_val) / (max_val - min_val)
    
    
# # # # # # # # # Path to the .nii file
# # # # # # # # all_images = []
# # # # # # # # all_masks = []
# # # # # # # # for number in range (1,11, 1):
    # # # # # # # # case_number = f"{number:03}"
    # # # # # # # # # Path to the .nii file for image data
    # # # # # # # # image_file_path = "C://Users//Motamed-Lab//Desktop//Lung//COVID-19-CT-Seg_20cases//coronacases_" +  case_number + ".nii//coronacases_org_" + case_number + ".nii"

    # # # # # # # # # Path to the .nii file for ground truth data
    # # # # # # # # ground_truth_file_path = "C://Users//Motamed-Lab//Desktop//Lung//Lung_Mask//coronacases_" + case_number + ".nii//coronacases_" + case_number + ".nii"

    # # # # # # # # # Load the .nii files
    # # # # # # # # image_nii_data = nib.load(image_file_path)
    # # # # # # # # image_data = image_nii_data.get_fdata()

    # # # # # # # # print(image_data.shape)
    # # # # # # # # ground_truth_nii_data = nib.load(ground_truth_file_path)
    # # # # # # # # ground_truth_data = ground_truth_nii_data.get_fdata()

    # # # # # # # # # Iterate over the desired slices
    # # # # # # # # start_slice = 30
    # # # # # # # # end_slice = image_data.shape[2]
    # # # # # # # # skip_slices = 10

    # # # # # # # # for slice_index in range(start_slice, end_slice, skip_slices):
        # # # # # # # # # Extract the data for the current slice
        # # # # # # # # image_slice_data = image_data[:, :, slice_index]
        # # # # # # # # image_slice_data = normalize_image(image_slice_data)
        # # # # # # # # ground_truth_slice_data = ground_truth_data[:, :, slice_index]
        # # # # # # # # all_images.append(image_slice_data)        
        # # # # # # # # all_masks.append(ground_truth_slice_data)
        # # # # # # # # # Plot the image data
        # # # # # # # # plt.subplot(1, 2, 1)
        # # # # # # # # plt.imshow(image_slice_data, cmap='gray')
        # # # # # # # # plt.axis('off')
        # # # # # # # # plt.title('Image - Slice: {}'.format(slice_index))

        # # # # # # # # # Plot the ground truth data
        # # # # # # # # plt.subplot(1, 2, 2)
        # # # # # # # # plt.imshow(ground_truth_slice_data, cmap='gray')
        # # # # # # # # plt.axis('off')
        # # # # # # # # plt.title('FS - Slice: {}'.format(slice_index))
        # # # # # # # # # print((ground_truth_slice_data))

        # # # # # # # # # plt.show()
# # # # # # # # all_images = np.array(all_images)
# # # # # # # # all_masks = np.array(all_masks)
# # # # # # # # # all_masks[all_masks==1]=0
# # # # # # # # all_masks[all_masks==2]=1
# # # # # # # # all_images = resize(all_images, (all_images.shape[0], crop_size, crop_size),
                        # # # # # # # # mode='reflect', preserve_range=True)
                        
# # # # # # # # all_masks = resize(all_masks, (all_masks.shape[0], crop_size, crop_size),
                        # # # # # # # # mode='reflect', preserve_range=True)
                        
# # # # # # # # print("***********************************")
# # # # # # # # print(all_images.shape)
# # # # # # # # print(all_masks.shape)


# # # # # # # # img_dev = all_images 
# # # # # # # # mask_dev = all_masks 

# # # # # # # # img_dev = np.expand_dims(img_dev, axis = -1)
# # # # # # # # mask_dev = np.expand_dims(mask_dev, axis = -1)

# # # # # # # # # print(img_train.shape)
# # # # # # # # print(img_dev.shape)        
# # # # # # # # print("this is image train shape apfohwefpowiehfpwoeifhweofihrepvouhWEFPOHOFUH")     
                                       
# # # # # # # # ###########################################################################        
        
# # # # # # # # # suponly_weights = os.path.join(save_folder,'model_' + contour_type + '.h5')  
# # # # # # # # # def weights
# # # # # # # # our_weights = "C://Users//Motamed-Lab//Desktop//AAAI//AAAI_lung2//AC2//20//nofilter//weights//0//0.1//model_l.h5"
# # # # # # # # pl_weights = "C://Users//Motamed-Lab//Desktop//AAAI//AAAI_lung2//PL//20//nofilter//weights//0//model_l.h5"
# # # # # # # # sup_weights = "C://Users//Motamed-Lab//Desktop//AAAI//AAAI_lung2//suponly//20//weights//0//model_l.h5"

# # # # # # # # # model1 = unet(input_shape, pretrained_weights= suponly_weights)
# # # # # # # # model1 = fcn_model(input_shape, num_classes, weights = our_weights)  

# # # # # # # # model2 = fcn_model(input_shape, num_classes, weights = pl_weights)  

# # # # # # # # model3 = fcn_model(input_shape, num_classes, weights = sup_weights)  

# # # # # # # # pred_masks_dev = model1.predict(img_dev, batch_size = 8, verbose = 1) 
# # # # # # # # pred_masks_dev[pred_masks_dev>=0.5] = 1
# # # # # # # # pred_masks_dev[pred_masks_dev<0.5] = 0

# # # # # # # # pred_masks_dev2 = model2.predict(img_dev, batch_size = 8, verbose = 1) 
# # # # # # # # pred_masks_dev2[pred_masks_dev2>=0.5] = 1
# # # # # # # # pred_masks_dev2[pred_masks_dev2<0.5] = 0


# # # # # # # # pred_masks_dev3 = model3.predict(img_dev, batch_size = 8, verbose = 1) 
# # # # # # # # pred_masks_dev3[pred_masks_dev3>=0.5] = 1
# # # # # # # # pred_masks_dev3[pred_masks_dev3<0.5] = 0


# # # # # # # # masks_deformables = []
# # # # # # # # for idx in [97,99,90,36,94,93]:
    # # # # # # # # init_mask = pred_masks_dev3[idx]
    # # # # # # # # img = img_dev[idx]
    # # # # # # # # res1,res2,res3 = chanvese(I = img.squeeze(), init_mask = init_mask[:,:,0], max_its = 200, display = False, alpha= 0.3 , shape_w = 0) 
    # # # # # # # # mask_deformable = np.expand_dims(res1, axis=2)            
    # # # # # # # # mask_deformable = mask_deformable.astype('uint8')
    # # # # # # # # masks_deformables.append(mask_deformable)
# # # # # # # # masks_deformables =  np.array(masks_deformables)   
# # # # # # # # i2 = 0
# # # # # # # # for i in [97,99,90,36,94,93]:
    # # # # # # # # # y_coordinates = np.arange(mask_dev[i].shape[1])
    # # # # # # # # # # print(y_coordinates)
    # # # # # # # # # # Create a copy of the mask
    # # # # # # # # modified_mask1 = np.copy(pred_masks_dev[i])
    # # # # # # # # modified_mask2 = np.copy(pred_masks_dev2[i])
    # # # # # # # # modified_mask3 = np.copy(pred_masks_dev3[i])
    # # # # # # # # modified_mask4 = np.copy(mask_dev[i])
    # # # # # # # # modified_mask5 = np.copy(masks_deformables[i2])

    # # # # # # # # # modified_mask5 = np.copy(pred_masks_dev[i])

    # # # # # # # # # # Broadcast the condition to match the shape of the mask
    # # # # # # # # # y_coordinates_expanded = y_coordinates[np.newaxis, :]
    # # # # # # # # # print(y_coordinates_expanded)

    # # # # # # # # # condition = (mask_dev[i] == 1) & (y_coordinates_expanded < 65)
    # # # # # # # # # # print(condition.shape)

    # # # # # # # # # # Set the corresponding pixels to 2
    # # # # # # # # # modified_mask[condition] = 2
    # # # # # # # # boolean_mask = np.zeros((128, 128), dtype=bool)
    # # # # # # # # boolean_mask[67:, :] = True
    # # # # # # # # condition21 = (pred_masks_dev[i] == 1).squeeze()    # Pixels with value 0
    # # # # # # # # condition22 = (pred_masks_dev2[i] == 1).squeeze()    # Pixels with value 0
    # # # # # # # # condition23 = (pred_masks_dev3[i] == 1).squeeze()    # Pixels with value 0
    # # # # # # # # condition24 = (mask_dev[i] == 1).squeeze()    # Pixels with value 0
    # # # # # # # # condition25 = (masks_deformables[i2] == 1).squeeze()    # Pixels with value 0

    # # # # # # # # # modified_mask[(modified_mask == 1) & (np.arange(modified_mask.shape[0]) < 65)] = 2
    # # # # # # # # # Combine the masks using element-wise AND
    # # # # # # # # combined_condition1 = np.logical_and(boolean_mask, condition21)
    # # # # # # # # combined_condition2 = np.logical_and(boolean_mask, condition22)
    # # # # # # # # combined_condition3 = np.logical_and(boolean_mask, condition23)
    # # # # # # # # combined_condition4 = np.logical_and(boolean_mask, condition24)
    # # # # # # # # combined_condition5 = np.logical_and(boolean_mask, condition25)

    # # # # # # # # # Set pixels that satisfy the combined condition to 2
    # # # # # # # # modified_mask1[combined_condition1] = 2
    # # # # # # # # modified_mask2[combined_condition2] = 2
    # # # # # # # # modified_mask3[combined_condition3] = 2
    # # # # # # # # modified_mask4[combined_condition4] = 2
    # # # # # # # # modified_mask5[combined_condition5] = 2

    # # # # # # # # # rgb_color = (255/255, 255/255, 0/255)
    # # # # # # # # # rgb_color2 = (255/255, 128/255, 0/255)

    # # # # # # # # # colors =  [ 'white', rgb_color2, rgb_color]
    # # # # # # # # # values = [0,1,2]
    # # # # # # # # # cmap_2 = ListedColormap(colors)
    # # # # # # # # # Visualize the results
    # # # # # # # # plt.subplot(3, 2, 1)
    # # # # # # # # plt.imshow(img_dev[i], cmap='gray')
    # # # # # # # # plt.subplot(3, 2, 2)
    # # # # # # # # plt.imshow(modified_mask1, cmap='gray')
    # # # # # # # # plt.subplot(3, 2, 3)
    # # # # # # # # plt.imshow(modified_mask2, cmap='gray')
    # # # # # # # # plt.subplot(3, 2, 4)
    # # # # # # # # plt.imshow(modified_mask3, cmap='gray')
    # # # # # # # # plt.subplot(3, 2, 5)
    # # # # # # # # plt.imshow(modified_mask4, cmap='gray')
    # # # # # # # # plt.subplot(3, 2, 6)
    # # # # # # # # plt.imshow(modified_mask5, cmap='gray')
    # # # # # # # # plt.title(i)
    
    # # # # # # # # plt.show()
    # # # # # # # # i2 = i2+1

# # # # # 2- SCD
# # # from train_sunnybrook import *
# # # import os, fnmatch, sys
# # # import random
# # # import time
# # # import os
# # # import cv2
# # # import numpy as np
# # # from fcn_mode import *
# # # from chan_vese_1 import chanvese
# # # import matplotlib.pyplot as plt
# # # from skimage import exposure, measure
# # # import tensorflow as tf
# # # print("num gpu available:",len(tf.config.experimental.list_physical_devices('GPU')))
# # # import torch
# # # from matplotlib.colors import ListedColormap


# # # rgb_color = (255/255, 0/255, 255/255)


# # # colors =  [rgb_color,'blue']



# # # # Define the corresponding values (0, 1, 2, 3)
# # # values = [0, 1]

# # # # Create the custom color map using ListedColormap
# # # cmap_1 = ListedColormap(colors)
# # # colors =  [ 'white', rgb_color]
# # # cmap_2 = ListedColormap(colors)

# # # rgb_color = (0/255, 204/255, 0/255)
# # # colors =  [ 'white', rgb_color]
# # # values = [0, 1]
# # # cmap_4 = ListedColormap(colors)

# # # # Create the custom color map using ListedColormap
# # # # colors =  [ 'white', 'blue']
# # # # cmap_2 = ListedColormap(colors)


# # # # Define the corresponding values (0, 1, 2, 3)
# # # values = [0, 1]

# # # # Create the custom color map using ListedColormap

# # # colors =  [ 'white', 'blue']


# # # # Define the corresponding values (0, 1, 2, 3)
# # # values = [0, 1]

# # # # Create the custom color map using ListedColormap
# # # cmap_3 = ListedColormap(colors)


# # # # from new14.py import*
# # # # Check if CUDA is available
# # # from keras import backend as K
# # # gpus = tf.config.experimental.list_physical_devices('GPU')

# # # if gpus:
    # # # print("Code is running on GPU")
# # # else:
    # # # print("Code is running on CPU")
        
# # # seed = 1234
# # # np.random.seed(seed)
# # # #SAX_SERIES = get_SAX_SERIES()

# # # ROOT_PATH = "C:\\Users\\Motamed-Lab\\Desktop\\AAAI\\data"
# # # # VAL_CONTOUR_PATH = "C:\\Users\\Motamed-Lab\\Desktop\\AAAI\\data\\valGT"
# # # # VAL_IMG_PATH = os.path.join(ROOT_PATH,
                   # # # # 'challenge_validation')
# # # # ONLINE_CONTOUR_PATH = "C:\\Users\\Motamed-Lab\\Desktop\\AAAI\\data\\online"
# # # # ONLINE_IMG_PATH = os.path.join(ROOT_PATH,
                   # # # # 'challenge_online')

# # # TRAIN_CONTOUR_PATH = "C:\\Users\\Motamed-Lab\\Desktop\\AAAI\\redct"
# # # TRAIN_IMG_PATH = os.path.join(ROOT_PATH,
                        # # # 'challenge_training')
# # # TEST_CONTOUR_PATH = "C:\\Users\\Motamed-Lab\\Desktop\\AAAI\\backup\\test"
# # # TEST_IMG_PATH = os.path.join(ROOT_PATH,
                        # # # 'challenge_test') 


# # # ALL_CONTOUR_PATH = "C:\\Users\\Motamed-Lab\\Desktop\\AAAI\\backup2\\all"
# # # ALL_IMG_PATH = os.path.join(ROOT_PATH,
                        # # # 'img_all')                        
                        

                        

                        

# # # contour_type = sys.argv[1]
# # # # os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[2]

# # # crop_size = 100
# # # input_shape = (crop_size, crop_size, 1)
# # # num_classes = 2

# # # all_ctrs = list(map_all_contours(ALL_CONTOUR_PATH, contour_type, shuffle=True))   
# # # all_images, all_masks = export_all_contours(all_ctrs, ALL_IMG_PATH, crop_size=crop_size)   
                                     
# # # indices_test = random.sample(range(len(all_ctrs)-1), 200)

# # # indices_all = [i2 for i2 in range(len(all_ctrs)) if i2 not in indices_test]
  
# # # indices = [  90]

# # # img_dev = all_images[500:700]
# # # mask_dev = all_masks[500:700]



# # # # # # # # # # # # # # preparing the labeled and unlabeled train data # # # # # # # # # #

  

        
# # # our_weights = "C:/Users/Motamed-Lab/Desktop/AAAI/AAAI/AC2/40/nofilter/weights/0/0.1/model_i.h5"       
# # # pl_weights = "C:/Users/Motamed-Lab/Desktop/AAAI\AAAI/PL/40/nofilter/weights/0/model_i.h5"      
# # # sup_weights =  "C:/Users/Motamed-Lab/Desktop/AAAI/AAAI/suponly/40/weights/0/model_i.h5"

# # # model1 = fcn_model(input_shape, num_classes, weights = our_weights)  

# # # model2 = fcn_model(input_shape, num_classes, weights = pl_weights)  
# # # model3 = fcn_model(input_shape, num_classes, weights = sup_weights)  


# # # pred_masks_dev = model1.predict(img_dev, batch_size = 32, verbose = 1) 

# # # pred_masks_dev[pred_masks_dev>=0.5] = 1
# # # pred_masks_dev[pred_masks_dev<0.5] = 0

# # # pred_masks_dev2 = model2.predict(img_dev, batch_size = 32, verbose = 1) 
# # # pred_masks_dev2[pred_masks_dev2>=0.5] = 1
# # # pred_masks_dev2[pred_masks_dev2<0.5] = 0

# # # pred_masks_dev3 = model3.predict(img_dev, batch_size = 32, verbose = 1) 
# # # preds_all = np.copy(pred_masks_dev3)

# # # pred_masks_dev3[pred_masks_dev3>=0.5] = 1
# # # pred_masks_dev3[pred_masks_dev3<0.5] = 0
# # # masks_deformables = []
# # # for idx, init_mask in enumerate (pred_masks_dev3):
    # # # img = img_dev[idx]
    # # # res1,res2,res3 = chanvese(I = img.squeeze(), init_mask = init_mask[:,:,0], max_its = 200, display = False, alpha= 0.5 , shape_w = 0) 
    # # # mask_deformable = np.expand_dims(res1, axis=2)            
    # # # mask_deformable = mask_deformable.astype('uint8')
    # # # masks_deformables.append(mask_deformable)
# # # masks_deformables =  np.array(masks_deformables)     
        
# # # # ##########################################################################################################################
# # # # # # # ## # # # # # # # # # # # # apply pseudo-labeling # # # # # # # # # # # # # # # # # 

# # # # for thresh1 in [0.0001, 0.003, 0.01, 0.1]:
    # # # # mask_final = np.copy(pred_masks_dev3)

    # # # # reliable_mask = np.ones_like(preds_all, dtype='uint16') 
    # # # # # reliable_tmp = np.ones_like(preds_all, dtype='uint16')
    # # # # print("this is thresholdAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
    # # # # print(thresh1)
    # # # # reliable_tmp = np.logical_or(preds_all < thresh1, preds_all > 1 - thresh1)
    # # # # reliable_mask[reliable_tmp] = 0
    # # # # mask_final [reliable_mask ==1] = masks_deformables [reliable_mask ==1]
    
# # # for i in range(len(img_dev)):
# # # # for i in [44]:
    # # # plt.subplot(3, 2, 1)
    # # # plt.imshow(img_dev[i] , cmap='gray')
    
    # # # plt.subplot(3, 2, 2)
    # # # plt.imshow(mask_dev[i] , cmap='gray')
    
    # # # plt.subplot(3, 2, 3)
    # # # plt.imshow(pred_masks_dev3[i] ,  cmap='gray')   

    # # # plt.subplot(3, 2, 4)
    # # # plt.imshow(pred_masks_dev2[i] ,  cmap='gray')  
    
    # # # plt.subplot(3, 2, 5)
    # # # plt.imshow(masks_deformables[i] ,  cmap='gray')  

    # # # plt.subplot(3, 2, 6)
    # # # plt.imshow(pred_masks_dev[i] ,  cmap='gray')
    
    # # # plt.title(i)
    
    # # # plt.show()




# 3- ACDC


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


def normalize_image(image):
    # Normalize pixel values to range [0, 1]
    min_val = np.min(image)
    max_val = np.max(image)
    return (image - min_val) / (max_val - min_val)
    


base_directory = r"C:\Users\Motamed-Lab\Desktop\ACDC\Resources"
gt_files, corresponding_files = find_gt_and_corresponding_files(base_directory)

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
all_images = resize(all_images, (all_images.shape[0], crop_size, crop_size),
                        mode='reflect', preserve_range=True)
                        
all_masks = resize(all_masks, (all_masks.shape[0], crop_size, crop_size),
                        mode='reflect', preserve_range=True)        
img_dev = all_images 
mask_dev = all_masks 
our_weights = "C:/Users/Motamed-Lab/Desktop/ACDC_c/ACDC_c/AAAI_lung/AC3/50/nofilter/weights/0/0.01/0.01/model_A.h5"   
pl_weights = "C:/Users/Motamed-Lab/Desktop/ACDC_c/ACDC_c/AAAI_lung/PLA/50/nofilter/weights/0/model_A.h5"      
sup_weights =  "C:/Users/Motamed-Lab/Desktop/ACDC_c/ACDC_c/AAAI_lung/suponly/50/weights/0/model_A.h5"    
print(sup_weights)
model1 = fcn_model(input_shape, num_classes, weights = our_weights)  

model2 = fcn_model(input_shape, num_classes, weights = pl_weights)  
model3 = fcn_model(input_shape, num_classes, weights = sup_weights)  


pred_masks_dev = model1.predict(img_dev, batch_size = 8, verbose = 1) 
print(sup_weights)
pred_masks_dev[pred_masks_dev>=0.5] = 1
pred_masks_dev[pred_masks_dev<0.5] = 0

pred_masks_dev2 = model2.predict(img_dev, batch_size = 8, verbose = 1) 
pred_masks_dev2[pred_masks_dev2>=0.5] = 1
pred_masks_dev2[pred_masks_dev2<0.5] = 0

pred_masks_dev3 = model3.predict(img_dev, batch_size = 8, verbose = 1) 
pred_masks_dev3[pred_masks_dev3>=0.5] = 1
pred_masks_dev3[pred_masks_dev3<0.5] = 0
# # # masks_deformables = []
# # # for idx, init_mask in enumerate (pred_masks_dev3):
    # # # img = img_dev[idx]
    # # # res1,res2,res3 = chanvese(I = img.squeeze(), init_mask = init_mask[:,:,0], max_its = 200, display = False, alpha= 0.5 , shape_w = 0) 
    # # # mask_deformable = np.expand_dims(res1, axis=2)            
    # # # mask_deformable = mask_deformable.astype('uint8')
    # # # masks_deformables.append(mask_deformable)
# # # masks_deformables =  np.array(masks_deformables)     
        
# # # # ##########################################################################################################################
# # # # # # # ## # # # # # # # # # # # # apply pseudo-labeling # # # # # # # # # # # # # # # # # 


single_mask = np.argmax(pred_masks_dev, axis=-1)
single_mask2 = np.argmax(pred_masks_dev2, axis=-1)
single_mask3 = np.argmax(pred_masks_dev3, axis=-1)
    
for i in range(len(img_dev)):
# for i in [44]:
    plt.subplot(3, 2, 1)
    plt.imshow(img_dev[i] , cmap='gray')
    
    plt.subplot(3, 2, 2)
    plt.imshow(mask_dev[i] , cmap=cmap_1)
    
    plt.subplot(3, 2, 3)
    plt.imshow(single_mask[i] ,  cmap=cmap_1)   

    plt.subplot(3, 2, 4)
    plt.imshow(single_mask2[i] ,  cmap=cmap_1)  
    
    # plt.subplot(3, 2, 5)
    # plt.imshow(masks_deformables[i] ,  cmap='gray')  

    plt.subplot(3, 2, 5)
    plt.imshow(single_mask3[i] ,  cmap=cmap_1)
    
    plt.title(i)
    
    plt.show()

        
        
        