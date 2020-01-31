# MNIST-to-PNG
Python script to transform the binary MNIST dataset (http://yann.lecun.com/exdb/mnist/) into image files.

The files are decompressed in PNG format and are separated based on their label.

There is two versions: a single core script and a multicore script. The multicore script uses a max of 10 CPUs.

To excecute the script go to the folder where the MNIST files are and type: python3 data_to_img.py

Developed by Bryan Casanelli, bryancasanelli@gmail.com.
