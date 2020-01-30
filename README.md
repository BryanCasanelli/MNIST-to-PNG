# MNIST-to-PNG
Convert MNIST files into PNG images

This code decompress the MNIST image files availables at http://yann.lecun.com/exdb/mnist/.

The files are decompressed in PNG format and are separated based on their label.

There is two versions: a single core script and a multicore script. The multicore script uses a max of 10 CPUs.

To excecute the script go to the folder where the MNIST files are and type: python3 data_to_img.py
