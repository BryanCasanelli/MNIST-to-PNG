# ------------------------------------------------------ #
# Developed by Bryan Casanelli, bryancasanelli@gmail.com #
# ------------------------------------------------------ #
# ----------------------------------------------- #
# This code decompress the MNIST image files      #
# availables at http://yann.lecun.com/exdb/mnist/ #
# The files are decompressed in PNG format and    #
# are separated based on their label              #
# ----------------------------------------------- #
import os
import numpy as np
from PIL import Image
import multiprocessing as mp
from multiprocessing.pool import Pool

# Load files
test_images_file = open("t10k-images.idx3-ubyte","rb")
test_labels_file = open("t10k-labels.idx1-ubyte","rb")
train_images_file = open("train-images.idx3-ubyte","rb")
train_labels_file = open("train-labels.idx1-ubyte","rb")

# Create output folders
dir_main = "MNIST images"
dir_test = f"{dir_main}/Test"
dir_train = f"{dir_main}/Train"
os.makedirs(dir_main,exist_ok=True)
os.makedirs(dir_test,exist_ok=True)
os.makedirs(dir_train,exist_ok=True)
for i in range(10):
    os.makedirs(f'{dir_test}/{i}',exist_ok=True)
    os.makedirs(f'{dir_train}/{i}',exist_ok=True)

# Function to read a binary file
def read_bin(file):
    output = []
    while True:
        byte = file.read(1)
        if byte == b"":
            break
        else:
            output += [int(byte.hex(),16)]
    return output

# Function to decompress images
def decompress_image_from_array(array,number):
    images = []
    count = 16
    for i in range(number):
        image = np.empty([28,28],dtype=np.uint8)
        for j in range(28):
            for z in range(28):
                image[j,z] = 255 - array[count]
                count += 1
        images += [image]
    return images

#Function to save images
def save_img(images,labels,dir):
    count = 1
    for image,label in zip(images,labels):
        img = Image.fromarray(image,"L")
        img.save(f'{dir}/{label}/{count}.png')
        count += 1

# Decompress test images
def decompress_test_images():
    test_images_array = read_bin(test_images_file)
    test_images = decompress_image_from_array(test_images_array,10000)
    test_labels = read_bin(test_labels_file)
    save_img(test_images,test_labels[8::],dir_test)

# Decompress train images
def decompress_train_images():
    train_images_array = read_bin(train_images_file)
    train_images = decompress_image_from_array(train_images_array,60000)
    train_labels = read_bin(train_labels_file)
    save_img(train_images,train_labels[8::],dir_train)

# Decompress
test = mp.Process(target=decompress_test_images).start()
train = mp.Process(target=decompress_train_images).start()
