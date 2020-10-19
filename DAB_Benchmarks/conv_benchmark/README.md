Please edit Makefile and change the library/include paths used to find CuDNN-related libraries.

The benchmark was built with CUDA 8.0.61 and CuDNN 7.1

Currently, the benchmark runs backward filter convolution with consecutive integers. At the end of the kernel, the benchmark prints out the gradient of the weights. To change the dimensions, copy one of the txt files from the resnet_configs/ directory and rename it to config.txt

e.g cp resnet_configs/config_block1_1.txt config.txt

To build:
Make

To run:
./benchmark

