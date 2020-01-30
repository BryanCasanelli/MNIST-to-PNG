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
from numba import jit

# Files path
test_images_dir = "t10k-images.idx3-ubyte"
test_labels_dir = "t10k-labels.idx1-ubyte"
train_images_dir = "train-images.idx3-ubyte"
train_labels_dir = "train-labels.idx1-ubyte"

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
def read_bin(dir):
    file = open(dir,"rb")
    size = os.path.getsize(dir)
    output = [file.read(1) for i in range(size)]
    return output

# Split a list
def chunk(list, size):
    output = []
    for i in range(0, len(list), size):
        output += [list[i:i + size]]
    return output

# Join a list with sublists
def join(list):
    output = []
    for aux in list:
        output += aux
    return output

# Function to convert binary into decimal
def bin_to_dec(bin):
    output = [int(aux.hex(),16) for aux in bin]
    return output

def bin_to_dec_thread(bin,threads):
    p = mp.Pool(threads)
    bin = chunk(bin,int(len(bin)/threads))
    output = p.map(bin_to_dec,bin)
    return join(output)

# Function to decompress images
def decompress_image_from_array(array):
    images = []
    count = 0
    for i in range(int(len(array)/(28*28))):
        image = np.empty([28,28],dtype=np.uint8)
        for j in range(28):
            for z in range(28):
                image[j,z] = 255 - array[count]
                count += 1
        images += [image]
    return images

def decompress_image_from_array_thread(array, threads):
    p = mp.Pool(threads)
    array = array[16::]
    array = chunk(array,int(len(array)/(28*28*8))*28*28)
    output = p.map(decompress_image_from_array,array)
    return join(output)

# Function to save images
def save_img(opt):
    images, labels, dir, names = opt
    for image,label,name in zip(images,labels,names):
        img = Image.fromarray(image,"L")
        img.save(f'{dir}/{label}/{name}.png')

def save_img_thread(images, labels, dir, threads):
    p = mp.Pool(threads)
    names = [i+1 for i in range(len(images))]
    names = chunk(names,int(len(names)/threads))
    images = chunk(images,int(len(images)/threads))
    labels = chunk(labels,int(len(labels)/threads))
    opt = []
    for i in range(len(names)):
        opt += [[images[i],labels[i],dir,names[i]]]
    p.map(save_img,opt)

# Decompress
def decompress(dir1,dir2,dir3):
    threads = min(mp.cpu_count(),10)
    images_binary = read_bin(dir1)
    images_array = bin_to_dec_thread(images_binary,threads)
    images = decompress_image_from_array_thread(images_array,threads)
    labels_binary = read_bin(dir2)
    labels_array = bin_to_dec_thread(labels_binary,threads)
    save_img_thread(images,labels_array[8::],dir3,threads)

# Decompress
if __name__ == "__main__":
    decompress(test_images_dir,test_labels_dir,dir_test)
    decompress(train_images_dir,train_labels_dir,dir_train)
