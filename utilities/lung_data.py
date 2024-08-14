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
    skip_slices = 10

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
all_masks[all_masks==2]=1
print("***********************************")
print(all_images.shape)
print(all_masks.shape)

num_samples = all_images.shape[0]
perm_index = np.random.permutation(num_samples)

# Shuffle the images and masks using the permutation index
all_images_s = all_images[perm_index]
all_masks_s = all_masks[perm_index]

# print(all_images[100])
for i in range (1, all_images_s.shape[0]):
    plt.subplot(1, 2, 1)
    plt.imshow(all_images_s[i] , cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(all_masks_s[i] , cmap='gray')
    plt.title(i)
    
    plt.show()