import os

# def find_gt_files(base_directory, file_extension=".nii"):
    # gt_files = []
    # for root, dirs, files in os.walk(base_directory):
        # for dir_name in dirs:
            # if dir_name.endswith('_gt.nii'):
                # gt_folder_path = os.path.join(root, dir_name)
                # for file_name in os.listdir(gt_folder_path):
                    # if file_name.endswith(file_extension):
                        # gt_files.append(os.path.join(gt_folder_path, file_name))
                        # break  # Assuming only one relevant file per _gt.nii folder
    # return gt_files

# if __name__ == "__main__":
    # base_directory = r"C:\Users\Motamed-Lab\Desktop\ACDC\Resources"
    # gt_files = find_gt_files(base_directory)

    # if gt_files:
        # print("Found the following files:")
        # for file in gt_files:
            # print(file)
    # else:
        # print("No files found.")
        
        
def find_gt_and_corresponding_files(base_directory, gt_file_extension=".nii", corresponding_file_extension=".nii"):
    gt_files = []
    corresponding_files = {}

    for root, dirs, files in os.walk(base_directory):
        for dir_name in dirs:
            if dir_name.endswith('_gt.nii'):
                gt_folder_path = os.path.join(root, dir_name)
                corresponding_folder_name = dir_name.replace('_gt', '')
                corresponding_folder_path = os.path.join(root, corresponding_folder_name)

                for file_name in os.listdir(gt_folder_path):
                    if file_name.endswith(gt_file_extension):
                        gt_files.append(os.path.join(gt_folder_path, file_name))
                        break  # Assuming only one relevant file per _gt.nii folder

                corresponding_files[corresponding_folder_path] = []

                for file_name in os.listdir(corresponding_folder_path):
                    if file_name.endswith(corresponding_file_extension):
                        corresponding_files[corresponding_folder_path].append(os.path.join(corresponding_folder_path, file_name))

    return gt_files, corresponding_files

