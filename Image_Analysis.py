# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 11:26:39 2025

@author: Aaron Foote
"""

#The purpose of this file is to develop and train a convolutional neural network that is able to classify a given microscope
#image of nanoparticles in solution as toxic or non-toxic.  Nanoparticles that have aggregated appear under the microscope as black blots
#that can vary in size and shape. An excess amount of these aggregates can cause toxcity when the nanoparticles are incubated with living
#cells.  Some microscope images contain no aggregates, and some are completely full of aggregates, in which case the toxicity level is clear.

#This project is designed for the cases in which the toxicity level is borderline, and may or may not have a toxic impact on cells.
#The materials used to create the nanoparticles are expensive, and it takes time and expertise to produce and disperse the nanoparticles
#Furthermore, testing the toxicity of a given batch of nanoparticles is a time intesive process, often taking a week or more.

#Being able to predict if a given batch of nanoparticles is toxic quickly from a microscopy image is valuable, and is the goal of the following code.

   
#import necessary libraries
import imageio as iio
import os 
import shutil
import pathlib
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import image_dataset_from_directory
import random
import boto3

#What follows is a few pre-processing funcitons that allowed me to label and organize the images used for model training. 

#function that randomizes all images in a folder and renames them
#so the differences in the order they were taken do not influence the model accuracy (ie. non toxic images were lumped together and vice versa)
def randomize_image_names(folder_path, category):
    """Randomizes the names of image files in the specified folder."""
    numbers = list(range(1, 1281))
    random.shuffle(numbers)
    for filename in os.listdir(folder_path):
        old_path = os.path.join(folder_path, filename)
        index = random.randint(0, len(numbers)-1)
        element = numbers[index]
        del numbers[index]
        new_filename =str(category) + '.' + str(element) + ".jpg"
        new_path = os.path.join(folder_path, new_filename)
        os.rename(old_path, new_path)


def retrieve_from_AWS(bucket_name, folder_name, file_name):
    s3 = boto3.client('s3')
    s3.download_file(bucket_name, folder_name, file_name)
    with open('FILE_NAME', 'wb') as f:
        s3.download_fileobj('amzn-s3-demo-bucket', 'OBJECT_NAME', f)

#These paths will need to be modified depending on the location of the full dataset
#the code below assumes we will be taking a small sample of a larger set of images, which may or may not be true

original_dir = pathlib.Path('Mixed_Images')
#contains images that are numbered and labled "good" or "bad" depending on the level of toxicity measured when tested in live cells
#as of now, the folder contains 2560 total images, with 1280 "good" (non-toxic) and 1280 "bad" (toxic) levels of aggregation

new_base_dir = pathlib.Path('Model_Images')
#folder organized to contain train, test, and validation folders of nanoparticle samples that are both good and bad

#Function that creates subfolders for training, testing, and validation.         
def categorize_images(subset_name, start_index, end_index):
    for category in ('good', 'bad'):
        dir = new_base_dir / subset_name / category
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.makedirs(dir)
        fnames = [f"{category}.{i}.jpg" for i in range(start_index, end_index)]
        for fname in fnames:
            shutil.copy(src=original_dir / fname, dst=dir / fname)
            
#The following three lines subdivide the total subset of images into a train, validation, and test set
#The choice of start and end index will depend on the total number of images to train the model on
categorize_images("train", start_index=1, end_index=513)
categorize_images("validation", start_index=513, end_index=769)
categorize_images("test", start_index=769, end_index=1281)
                
# Create the new folder if it doesn't exist
def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

#The images were collected by a microscope with image size of 2048 by 1536 pixels
#since the typical aggregate was much smaller than the total size of the image, and to increase the number of images tested,
#each image was split into 64 smaller images of dimensions 256 x 192 pixels using the function below.
#Note this code only splits "good" dispersion images and needs to be replicated for "bad" images in order to split those images as well.
def Divide_Image(image, split_number, image_num):
    create_folder("Good Dispersions\Sub_Images")
    dim1_len = image.shape[0]
    dim2_len = image.shape[1]
    start1 = 0
    end1 = int(dim1_len/split_number)
    name=1
    while end1 <= dim1_len:
        start2 = 0
        end2 = int(dim2_len/split_number)
        while end2 <= dim2_len:
            sub_image = image[start1:end1, start2:end2]
            file_name = "sub_image_" + str(name+(64*(image_num-1))) + '.jpg'
            new_image_path = os.path.join('Good Dispersions\Sub_Images', file_name)
            iio.imwrite(new_image_path, sub_image)
            start2 += int(dim2_len/split_number)
            end2 += int(dim2_len/split_number)
            name+=1
        start1 += int(dim1_len/split_number)    
        end1 += int(dim1_len/split_number)

#high level function that splits and organizes original images into appropriate sizes and folders
def Organize_Images(num_images):
    image_num = 1
    while image_num <= num_images:
        image_name = "Good Dispersions/Sample_" + str(image_num) + ".jpg"
        img = iio.imread(image_name)
        Divide_Image(img, 8, image_num)
        image_num += 1
    folder_name = "Good Dispersions\Sub_Images"
    randomize_image_names(folder_name, 'good')
        
        
#The following code is part of the preprocessing step of any model, namely organizing the desired
#images into the correct directories, converting images into floating point tensors, resizing them
#(if needed) and packing them into batches

train_dataset = image_dataset_from_directory(
    new_base_dir / 'train',
    image_size = (256, 192),
    batch_size = 32)

validation_dataset = image_dataset_from_directory(
    new_base_dir / 'validation',
    image_size = (256, 192),
    batch_size = 32)

test_dataset = image_dataset_from_directory(
    new_base_dir / 'test',
    image_size = (256, 192),
    batch_size = 32)        
        
        
#The model consists of a multilayer convolutional neural network with alternating convolution and max pooling layers
inputs = keras.Input(shape=(256, 192, 3)) #the model expects RGB images of 256x192 pixels
x = layers.Rescaling(1.0/255)(inputs) #rescale inputs to the [0,1] range, assumes current range is [0, 255]
x = layers.Conv2D(filters=32, kernel_size=3, activation='relu')(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=64, kernel_size=3, activation='relu')(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=128, kernel_size=3, activation='relu')(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=256, kernel_size=3, activation='relu')(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=256, kernel_size=3, activation='relu')(x)
x = layers.Flatten()(x)
outputs = layers.Dense(1, activation='sigmoid')(x) #output is only 1 layer (and sigmoid is chosen) since this is a binary classifcation problem
model = keras.Model(inputs=inputs, outputs=outputs)

#model.compile(loss='binary_crossentropy',
              #optimizer='rmsprop',
              #metrics=['accuracy'])


#code for fitting the model, including a callback to save only the epoch that trains for the highest accuracy

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath='dispersion_convnet.keras',
        save_best_only=True,
        monitor='val_loss')
    ]

history = model.fit(
    train_dataset,
    epochs=30,
    validation_data=validation_dataset, callbacks=callbacks)

#Final code to be executed with program is run
#comment out when debugging for faster protoyping
#test_model = keras.models.load_model("dispersion_convnet.keras")
#test_loss, test_acc = test_model.evaluate(test_dataset)
#print(f"Test accuracy: {test_acc:.3f}")

#Last iteration led to a test accuracy of 96% in classifying toxic vs non-toxic dispersions

















