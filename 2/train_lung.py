#!/usr/bin/env python2.7
from keras.losses import binary_crossentropy
from keras.losses import CategoricalCrossentropy
from fcn_mode import*
import pydicom, cv2, re, math, shutil
import os, fnmatch, sys
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
#from itertools import izip
# from fcn_mode import *
# from U_net import *
from helpers import center_crop, lr_poly_decay, get_SAX_SERIES
# from U_net import *
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt
import tensorflow as tf

print(tf.__version__)
seed = 1234
np.random.seed(seed)

def save_results(folder_name, file_name, data):
    # Get the current directory where the Python script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # print(current_dir)
    # Create a directory to save the file in
    directory = folder_name
    if not os.path.exists(os.path.join(current_dir, directory)):
        os.makedirs(os.path.join(current_dir, directory))

    # List of numbers to save to a file
 
    # Define the filename and path of the file to save
    filename = file_name
    filepath = os.path.join(current_dir, directory, filename)

    # Open the file in write mode and save the list of numbers
    with open(filepath, "w") as f:
        for number in data:
            f.write(str(number) + "\n")
            
    # Confirm that the file has been saved by printing its contents
    with open(filepath, "r") as f:
        file_contents = f.read()
        # print(file_contents)
        
def get_elements(list, indices):
    elements = []
    for i in indices:
        elements.append(list[i])
    return elements
    
def dice_coef(y_true, y_pred, smooth=0.0):
    # print(np.array(y_true).shape)
    '''Average dice coefficient per batch.'''
    axes = (1,2,3)
    intersection = K.sum(y_true * y_pred, axis=axes)
    summation = K.sum(y_true, axis=axes) + K.sum(y_pred, axis=axes)
    
    return K.mean((2.0 * intersection + smooth) / (summation + smooth), axis=0)



    

def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred, smooth=10.0)
    
# def jaccard_coef(y_true, y_pred, smooth=0.0):
    # '''Average jaccard coefficient per batch.'''
    # axes = (1,2,3)
    # intersection = K.sum(y_true * y_pred, axis=axes)
    # union = K.sum(y_true, axis=axes) + K.sum(y_pred, axis=axes) - intersection
    # return K.mean( (intersection + smooth) / (union + smooth), axis=0)

    
class Contour(object):
    def __init__(self, ctr_path):
        self.ctr_path = ctr_path
        match = re.search(r'\\([^\\]*)\\contours-manual\\IRCCI-expert\\IM-0001-(\d{4})-.*', ctr_path)
        self.case = match.group(1)
        self.img_no = int(match.group(2))
        self.slice_no =  math.floor(self.img_no/20) if self.img_no%20 !=0 else math.floor(self.img_no/20)-1
        self.ED_flag = True if ((self.img_no%20) < 10 and (self.img_no % 20) !=0) else False
        self.is_weak = 0
   
    
    def __str__(self):
        return 'Contour for case %s, image %d' % (self.case, self.img_no)
    
    __repr__ = __str__
def read_contour(contour, data_path):
    filename = 'IM-0001-%04d.dcm' % ( contour.img_no)
    full_path = os.path.join(data_path, contour.case,'DICOM', filename)
    f = pydicom.dcmread(full_path)
    img = f.pixel_array.astype('int')
    mask = np.zeros_like(img, dtype='uint8') # shape is 256, 256


    coords = np.loadtxt(contour.ctr_path, delimiter=' ').astype('int')
    # print(coords.shape)   (num_points , 2)
    # print("this is coords shape")
    cv2.fillPoly(mask, [coords], 1)
    if img.ndim < 3:
        img = img[..., np.newaxis]
        mask = mask[..., np.newaxis]
    
    return img, mask

def map_all_contours(contour_path, contour_type, shuffle=True):
    contours = [os.path.join(dirpath, f)
        for dirpath, dirnames, files in os.walk(contour_path)
        for f in fnmatch.filter(files,
                        'IM-0001-*-'+contour_type+'contour-manual.txt')]
    #for dirpath, dirnames, files in os.walk(contour_path):
    #    print(files)
    if shuffle:
        print('Shuffling data')
        np.random.shuffle(contours)
        
    print('Number of examples: {:d}'.format(len(contours)))
    contours = map(Contour, contours)
    
    return contours
    
def export_all_contours(contours, data_path, crop_size):
    print('\nProcessing {:d} images and labels ...\n'.format(len(list(contours))))
    print(len(list(contours)))
    images = np.zeros((len(list(contours)), crop_size, crop_size, 1))
    masks = np.zeros((len(list(contours)), crop_size, crop_size, 1))
    for idx, contour in enumerate(contours):
        img, mask = read_contour(contour, data_path)
        img = center_crop(img, crop_size=crop_size)
        mask = center_crop(mask, crop_size=crop_size)
        images[idx] = img
        masks[idx] = mask
        
        # shifted_mask = np.zeros_like(mask)
        # shift_y = 10
        # shift_x = 10
        # shifted_mask[shift_y:100, shift_x:100] = mask[0:100-shift_y, 0:100-shift_x] 
        # # masks[idx] = shifted_mask
        # mask = mask.squeeze()
        # # Find the bounding box of the foreground
        # nonzero_pixels = np.argwhere(mask > 0)
        # min_row, min_col = np.min(nonzero_pixels, axis=0)
        # max_row, max_col = np.max(nonzero_pixels, axis=0)

        # # Extract the foreground region from the mask
        # foreground = mask[min_row:max_row+1, min_col:max_col+1]

        # # Scale the foreground using interpolation
        # scale_factor = 2  # Adjust the scale factor as needed
        # new_height = int(foreground.shape[0] * scale_factor)
        # new_width = int(foreground.shape[1] * scale_factor)
        # scaled_foreground = cv2.resize(foreground, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

        # # Create an expanded mask with the same size as the original mask
        # expanded_mask = np.zeros_like(mask)

        # # Place the scaled foreground back into the expanded mask
        # expanded_mask[min_row:max_row+1, min_col:max_col+1] = scaled_foreground
        # # Resize the mask using interpolation
        # # Display the original and shifted masks
        # plt.subplot(1, 2, 1)
        # plt.imshow(mask.squeeze(), cmap='gray')
        # plt.title('Original Mask')

        # plt.subplot(1, 2, 2)
        # plt.imshow(expanded_mask, cmap='gray')
        # plt.title('Shifted Mask')
        # plt.show()

    return images, masks  

    
    
# def mean_teacher(X_labeled, Y_labeled, X_unlabeled, img_dev, mask_dev, num_epochs, alpha = 0.99):
    # # Get input shape
    # input_shape = (crop_size, crop_size, 1)
    # num_classes = 2    
    # best_result = 0
    # best_epoch = 0
    # curr_iter = 0
    # mini_batch_size = 1
    # epochs = 40
    # model = fcn_model(input_shape, num_classes, weights=None)
    # sgd = tf.keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
    # # Create student model and teacher model
    # student_model = model
    # teacher_model = model
    # teacher_model.set_weights(student_model.get_weights())  # Initialize teacher model with student's weights

    # # Define loss functions
    # # labeled_loss_fn =  binary_crossentropy
    # # consistency_loss_fn = tf.keras.losses.MeanSquaredError()
    # # Define optimizer
    # optimizer = sgd

    # # Compile the student model with combined loss
    # base_lr = K.eval(student_model.optimizer.lr)
    # max_iter = (len(X_labeled) / mini_batch_size) * epochs
    # lrate = lr_poly_decay(student_model, base_lr, curr_iter, max_iter, power=0.5)

    # # Training loop
    # for epoch in range(num_epochs):
    
        # print('\nMain Epoch {:d}\n'.format(epoch+1))
        # print('\nLearning rate: {:6f}\n'.format(lrate))        # Iterate over labeled data batches
        # train_result = []
        # for X_batch_labeled, Y_batch_labeled in zip(X_labeled, Y_labeled):
            # # Train on labeled batch
            # X_batch_labeled = np.expand_dims (X_batch_labeled , axis = 0)
            # Y_batch_labeled = np.expand_dims (Y_batch_labeled , axis = 0)
            # # print(X_batch_labeled.shape)
            # student_model.compile(optimizer=sgd, loss= 'binary_crossentropy',
                        # metrics=[dice_coef, jaccard_coef], run_eagerly = True)
            # labeled_loss_value = student_model.train_on_batch(X_batch_labeled, Y_batch_labeled)

            # curr_iter += 1
            # lrate = lr_poly_decay(student_model, base_lr, curr_iter,
                                  # max_iter, power=0.5)
            # train_result.append(labeled_loss_value)
        # train_result = np.asarray(train_result)
        # train_result = np.mean(train_result, axis=0).round(decimals=10)
        # # print(student_model.metrics_names, train_result)
        # # print("this is sup only train")
        
        

                        
        # result1 = student_model.evaluate(img_dev, mask_dev, batch_size=32)
        # result1 = np.round(result1, decimals=10)
        
        # print(student_model.metrics_names, result1)
        # print("this is sup only dev")        

        # student_model.compile(optimizer, loss = mean_teacher_loss, metrics=[dice_coef, jaccard_coef])
        
        # train_result2 = []
                                        
        # for X_batch_unlabeled in X_unlabeled:    
        
            # # Forward pass on student model for unlabeled data
            # X_batch_unlabeled = np.expand_dims (X_batch_unlabeled , axis = 0)
            
            # student_logits = student_model.predict (X_batch_unlabeled, verbose=0) 

            # teacher_logits = teacher_model.predict (X_batch_unlabeled, verbose=0) 

            # # Compute consistency loss
            # # consistency_loss_value = consistency_loss_fn(student_logits, teacher_logits)

            # # Train on unlabeled batch with consistency loss
            # res = student_model.train_on_batch(X_batch_unlabeled, np.concatenate((student_logits, teacher_logits), axis=-1))
            # train_result2.append(res)
        # train_result2 = np.asarray(train_result2)
        # train_result2 = np.mean(train_result2, axis=0).round(decimals=10)
        # # print(student_model.metrics_names, train_result2)
        # # print("this is sub only plus consistency train")
        
        # # Update teacher model weights with exponential moving average
        # teacher_weights = teacher_model.get_weights()
        # student_weights = student_model.get_weights()
            
        # new_teacher_weights = [(1.0 - alpha) * tw + alpha * sw for tw, sw in zip(teacher_weights, student_weights)]
        # teacher_model.set_weights(new_teacher_weights)
        # # Print or log the loss values or metrics for monitoring


        # print('\nEvaluating final dev set ...')
        
        
        # student_model.compile(optimizer=sgd, loss= 'binary_crossentropy',
                        # metrics=[dice_coef, jaccard_coef], run_eagerly = True)
                        
        # result2 = student_model.evaluate(img_dev, mask_dev, batch_size=32)
        # result2 = np.round(result2, decimals=10)
        
        # print(student_model.metrics_names, result2)
        # print("this is final dev")
        # # save_file = '_'.join(['model', contour_type]) + '.h5'
        # # if not os.path.exists(folder_name):
            # # os.makedirs(folder_name)
        # # save_path = os.path.join(folder_name, save_file)
        
        # # if result[1] > best_result:
            # # best_result = result[1]
            # # best_epoch = e+1
            # # model.save_weights(save_path)        
        
        
        
    # # Return the trained student model
    # return student_model
    
    
def train_module(contour_type, folder_name, train_images, train_masks, test_images, test_masks, loss, lr_input ):
    crop_size = 150
    input_shape = (crop_size, crop_size, 1)
    num_classes = 4
    
    best_result = 0
    best_epoch = 0
    print(np.max(train_images[0]))                          

    #weights = 'C:\\Users\\r_beh\\cardiac-segmentation-master\\cardiac-segmentation-master\\model_logs_backupl\\sunnybrook_i_epoch_40.h5'
    model = fcn_model(input_shape, num_classes, weights=None)
    sgd = tf.keras.optimizers.SGD(lr=lr_input , momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='CategoricalCrossentropy', metrics=[ dice2, dice3, dice4])
    

                    
    kwargs = dict(
    rotation_range=180,
    zoom_range=0.0,
    width_shift_range=0.0,
    height_shift_range=0.0,
    horizontal_flip=True,
    vertical_flip=True,
    )
    image_datagen = ImageDataGenerator(**kwargs)
    mask_datagen = ImageDataGenerator(**kwargs)

    epochs = 40
    mini_batch_size = 1
    img_train = train_images
    mask_train = train_masks
    img_dev = test_images
    mask_dev = test_masks
    # image_generator = image_datagen.flow(img_train, shuffle=False,
                                    # batch_size=mini_batch_size, seed=seed)
    # mask_generator = mask_datagen.flow(mask_train, shuffle=False,
                                    # batch_size=mini_batch_size, seed=seed)
    # train_generator = zip(image_generator, mask_generator)
    
    max_iter = (len(img_train) / mini_batch_size) * epochs
    curr_iter = 0
    if loss == 'AC':
        x = 1
    else:
        x = 1
    base_lr = x * K.eval(model.optimizer.lr)
    lrate = lr_poly_decay(model, base_lr, curr_iter, max_iter, power=0.8)
    
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        
    results_file = os.path.join(folder_name, "results.txt")
    
    # with open(results_file, "w") as file:
        # file.write("Epoch\tTraining Results\tValidation Results\n")
    for e in range(epochs):
        if e ==0:
            length_coeff = 1
        else:
            length_coeff = 1
    
        if loss == 'CE':
            model.compile(optimizer=sgd, loss= 'CategoricalCrossentropy',
                        metrics=[dice2,dice3,dice4,jac2,jac3,jac4], run_eagerly = True)
        elif loss == 'AC':
            model.compile(optimizer = sgd, loss = AAAI_Loss,
                        metrics=[dice1, jaccard_coef, hausdorff_distance], run_eagerly = True) 

        elif loss == 'AC2':
            model.compile(optimizer = sgd, loss = AAAI2_Loss(length_coeff),
                        metrics=[dice1, jaccard_coef, hausdorff_distance], run_eagerly = True)      

        elif loss == 'AC3':
            model.compile(optimizer = sgd, loss = AAAID_Loss,
                        metrics=[dice2,dice3,dice4,jac2,jac3,jac4], run_eagerly = True)  
                        
                    
        print('\nMain Epoch {:d}\n'.format(e+1))
        print('\nLearning rate: {:6f}\n'.format(lrate))
        train_result = []
        for iteration in range(int(len(img_train)/mini_batch_size)):
            # img, mask = next(train_generator)
            img = np.expand_dims (img_train [iteration], axis = 0)
            mask = np.expand_dims (mask_train [iteration], axis = 0)
            # img = img_train [iteration]
            # mask = mask_train [iteration]
            # print("hereherhereherherherherhe")
            # print("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF")
            # print(type(img))
            res = model.train_on_batch(img, mask)
            
            curr_iter += 1
            lrate = lr_poly_decay(model, base_lr, curr_iter,
                                  max_iter, power=0.8)
            train_result.append(res)
        train_result = np.asarray(train_result)
        train_result = np.mean(train_result, axis=0).round(decimals=10)
        print(model.metrics_names, train_result)
        print('\nEvaluating dev set ...')
        
        model.compile(optimizer = sgd, loss= 'CategoricalCrossentropy',
                        metrics=[ dice2, dice3, dice4, jac2, jac3, jac4 ], run_eagerly = True)

                    
        result = model.evaluate(img_dev, mask_dev, batch_size= 8)
        result = np.round(result, decimals=10)
        with open(results_file, "a") as file:
            file.write("{}\t{}\t{}\n".format(e+1, train_result, result))    

        print(model.metrics_names, result)
        save_file = '_'.join(['model', contour_type]) + '.h5'
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        save_path = os.path.join(folder_name, save_file)
        
        if result[1] > best_result:
            best_result = result[1]
            best_epoch = e+1
            model.save_weights(save_path)
    return best_result, best_epoch



        
        

if __name__== '__main__':
   # if len(sys.argv) < 3:
   #     sys.exit('Usage: python %s <i/o> <gpu_id>' % sys.argv[0])
    contour_type = sys.argv[1]
    print(contour_type)
    #training_dataset= sys.argv[2]
    #os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[2]
    crop_size = 150

    print('Mapping ground truth '+contour_type+' contours to images in train...')

    #train_ctrs = map_all_contours(TRAIN_CONTOUR_PATH, contour_type, shuffle=True)
    test_ctrs = list(map_all_contours(TEST_CONTOUR_PATH, contour_type, shuffle=True))
    test_ctrs = test_ctrs[0:len(test_ctrs)//10]
    train_ctrs = list(map_all_contours(TRAIN_CONTOUR_PATH, contour_type, shuffle=True))


    #train_ctrs_validation = map_all_contours(TRAIN_CONTOUR_PATH_validation, contour_type, shuffle=True)
    # picked_contours = choose_n_contours (numbers_of_contours, list(train_ctrs_original), "/") 
    print('Done mapping training set')

    # split = int(0*len(a))
    # train_ctrs=a[split:]
    # #dev_ctrs = b[0:split]
    # print(len(a))
    # print("before")

    print(len(train_ctrs))
    
    print('\nBuilding Train dataset ...')
    img_train, mask_train = export_all_contours(train_ctrs,
                                                TRAIN_IMG_PATH,
                                                crop_size=crop_size)
    # print(np.array(mask_train).shape)
    # print("mask train shape")    (num_ctrs, 100, 100, 1)
    
    print('\nBuilding Dev dataset ...')
    img_dev, mask_dev = export_all_contours(test_ctrs,
                                            TEST_IMG_PATH,
                                            crop_size=crop_size)
    
    input_shape = (crop_size, crop_size, 1)
    num_classes = 4
    #weights = 'C:\\Users\\r_beh\\cardiac-segmentation-master\\cardiac-segmentation-master\\model_logs_backupl\\sunnybrook_i_epoch_40.h5'
    model = fcn_model(input_shape, num_classes, weights=None)

    # sgd = tf.keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
    # model.compile(optimizer=sgd, loss= dice_coef_loss,
                  # metrics=['accuracy', dice_coef, jaccard_coef], run_eagerly = True)
    kwargs = dict(
        rotation_range=180,
        zoom_range=0.0,
        width_shift_range=0.0,
        height_shift_range=0.0,
        horizontal_flip=True,
        vertical_flip=True,
    )
    image_datagen = ImageDataGenerator(**kwargs)
    mask_datagen = ImageDataGenerator(**kwargs)

    epochs = 40
    mini_batch_size = 1

    image_generator = image_datagen.flow(img_train, shuffle=False,
                                    batch_size=mini_batch_size, seed=seed)
    mask_generator = mask_datagen.flow(mask_train, shuffle=False,
                                    batch_size=mini_batch_size, seed=seed)
    train_generator = zip(image_generator, mask_generator)
    
    max_iter = (len(train_ctrs) / mini_batch_size) * epochs
    curr_iter = 0
    base_lr = K.eval(model.optimizer.lr)
    lrate = lr_poly_decay(model, base_lr, curr_iter, max_iter, power=0.5)
    for e in range(epochs):
        print('\nMain Epoch {:d}\n'.format(e+1))
        print('\nLearning rate: {:6f}\n'.format(lrate))
        train_result = []
        for iteration in range(int(len(img_train)/mini_batch_size)):
            img, mask = next(train_generator)
            res = model.train_on_batch(img, mask, sample_weight = 1)
            
            curr_iter += 1
            lrate = lr_poly_decay(model, base_lr, curr_iter,
                                  max_iter, power=0.5)
            train_result.append(res)
        train_result = np.asarray(train_result)
        train_result = np.mean(train_result, axis=0).round(decimals=10)
        print(model.metrics_names, train_result)
        print('\nEvaluating dev set ...')
        result = model.evaluate(img_dev, mask_dev, batch_size=32)
        result = np.round(result, decimals=10)
        print(model.metrics_names, result)
        save_file = '_'.join(['sunnybrook', contour_type,
                              'epoch', str(e+1)]) + '.h5'
        if not os.path.exists('realtime'):
            os.makedirs('realtime')
        save_path = os.path.join('realtime', save_file)
        #print(save_path)
        model.save_weights(save_path)



