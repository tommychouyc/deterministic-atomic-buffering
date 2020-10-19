#include <sstream>
#include <fstream>
#include <stdlib.h>
#include <cstdlib>
#include <assert.h>

#include <cuda.h> // need CUDA_VERSION
#include <cudnn.h>

#include "error_util.h"
#include "configs.h"

union fb
{
	unsigned int byte;
	float f;
};

// Generate uniform numbers [0, 1)
// Taken from cudnn conv_sample
void initTensor(float* tensor, int size)
{
    static unsigned seed = 123456789;
    for (int i = 0; i < size; i++)
    {
	seed = (1103515245*seed + 12345) & 0xffffffff;
	tensor[i] = float(seed)*2.3283064e-10;
    }
}

void initTensorOnes(float* tensor, int size)
{
    for (int i = 0;i < size; i++)
    {
	tensor[i] = float(1);
    }
}

void initTensorZeros(float* tensor, int size)
{
    for (int i = 0;i < size; i++)
    {
	tensor[i] = 0;
    }
}


void initTensorCons(float* tensor, int size)
{
    for (int i = 0;i < size; i++)
    {
	tensor[i] = float(i);
    }
}

int main(int argc, char *argv[])
{
    config c;

    const cudnnConvolutionBwdFilterAlgo_t algs[7] = 
    	{
		CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
		CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1,
		CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT,
		CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3,
	        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD, //not implemented	
		CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED,
		CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING
    	};

    cudnnHandle_t cudnnHandle;
    cudnnTensorDescriptor_t iDesc;
    cudnnTensorDescriptor_t oDesc;
    cudnnFilterDescriptor_t fDesc;
    cudnnConvolutionDescriptor_t cDesc;

    std::ifstream config("config.txt");
    std::string line;

    int lineNum = 0;

    int configValues[18];
    int defaultConfig[18] = {};

    fb float_byte;

    while (std::getline(config, line))
    {
	    configValues[lineNum] = std::atoi(line.c_str());
    	lineNum++;
    }
    if (lineNum == 18)
    {
        memcpy(&c, configValues, 18*4);
    }
    else
    {
        memcpy(&c, defaultConfig, 18*4);
    }
    
	std::cout << "Input Tensor Dim: " << c.i_n << "x" << c.i_c << "x" << c.i_h << "x" << c.i_w << std::endl;

    std::cout << "Output Tensor Dim: " << c.o_n << "x" << c.o_c << "x" << c.o_h << "x" << c.o_w << std::endl;

    std::cout << "Filter Dim: " << c.f_k << "x" << c.f_c << "x" << c.f_h << "x" << c.f_w << std::endl;

    std::cout << "Convolution: Padding: [" << c.pad_h << "," << c.pad_w << "] Stride: [" << c.stride_h << "," << c.stride_w << "] Dilation: [" << c.u << "," << c.v << "]" << std::endl;
    
    int selectedAlgNum;

    float alpha = 1;
    float beta  = 0;

    size_t sizeInBytes;

    float* input = new float[c.i_n*c.i_c*c.i_h*c.i_w];
    float* output = new float[c.o_n*c.o_c*c.o_h*c.o_w];
    float* weight = new float[c.f_k*c.f_c*c.f_h*c.f_w];

    std::ifstream inputTxt("input.txt");
    std::ifstream outputTxt("output.txt");
    std::ifstream weightTxt("weight.txt");
    
    int index = 0;

    while (std::getline(inputTxt, line) && index < (c.i_n*c.i_c*c.i_h*c.i_w))
    {
	float_byte.byte = (unsigned int) std::strtol(line.c_str(), NULL, 16);
	input[index] = float_byte.f;
    	index++;
    }

    if (index < (c.i_n*c.i_c*c.i_h*c.i_w))
    {
	    std::cout << "Not enough inputs from input.txt. Using random inputs" << std::endl;
    	initTensorCons(input, c.i_n*c.i_c*c.i_h*c.i_w);
    }

    index = 0;
//	input[5] = 2.0;
    while (std::getline(outputTxt, line) && index < (c.o_n*c.o_c*c.o_h*c.o_w))
    {
	float_byte.byte = (unsigned int) std::strtol(line.c_str(), NULL, 16);
	    output[index] = float_byte.f;
    	index++;
    }

    if (index < (c.o_n*c.o_c*c.o_h*c.o_w))
    {
	    std::cout << "Not enough outputs from output.txt. Using random outputs" << std::endl;
    	initTensorOnes(output, c.o_n*c.o_c*c.o_h*c.o_w);
    }
    index = 0;

    void* workSpace = NULL;
    void* dInput = NULL;
    void* dOutput = NULL;
    void* dWeight = NULL;

    if (checkCmdLineFlag(argc, (const char**) argv, "alg"))
    {

	selectedAlgNum = getCmdLineArgumentInt(argc, (const char**) argv, "alg");
	
	assert(selectedAlgNum < 6);

	std::cout << "Algorithm "<< selectedAlgNum << " selected" << std::endl;    
    }
    else
    {
	selectedAlgNum = 0;
	std::cout << "No algorithm specified. By default, algorithm 0 selected" << std::endl;    
    }

    checkCUDNN(cudnnCreate(&cudnnHandle));

    checkCUDNN(cudnnCreateTensorDescriptor(&iDesc));
    checkCUDNN(cudnnSetTensor4dDescriptorEx(iDesc, CUDNN_DATA_FLOAT, c.i_n, c.i_c, c.i_h, c.i_w, c.i_c*c.i_h*c.i_w, c.i_h*c.i_w, c.i_w, 1));
    
    checkCUDNN(cudnnCreateTensorDescriptor(&oDesc));
    checkCUDNN(cudnnSetTensor4dDescriptorEx(oDesc, CUDNN_DATA_FLOAT, c.o_n, c.o_c, c.o_h, c.o_w, c.o_c*c.o_h*c.o_w, c.o_h*c.o_w, c.o_w, 1));

    checkCUDNN(cudnnCreateFilterDescriptor(&fDesc));
    checkCUDNN(cudnnSetFilter4dDescriptor(fDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, c.f_k, c.f_c, c.f_h, c.f_w));

    checkCUDNN(cudnnCreateConvolutionDescriptor(&cDesc));
    checkCUDNN(cudnnSetConvolution2dDescriptor(cDesc, c.pad_h, c.pad_w, c.stride_h, c.stride_w, c.u, c.v, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnnHandle, iDesc, oDesc, cDesc, fDesc, algs[selectedAlgNum], &sizeInBytes));

    std::cout << "Workspace size: " << sizeInBytes << std::endl;
    checkCudaErrors(cudaMalloc(&workSpace, sizeInBytes)); // workspace
    checkCudaErrors(cudaMalloc(&dInput, 4*c.i_w*c.i_h*c.i_c*c.i_n)); // input

    if (dInput != NULL)
    {
	checkCudaErrors(cudaMemcpy(dInput, input, 4*c.i_w*c.i_h*c.i_c*c.i_n, cudaMemcpyHostToDevice));
    }
    checkCudaErrors(cudaMalloc(&dOutput, 4*c.o_w*c.o_h*c.o_c*c.o_n)); // output grad
    
    if (dOutput != NULL)
    {
	checkCudaErrors(cudaMemcpy(dOutput, output, 4*c.o_w*c.o_h*c.o_c*c.o_n, cudaMemcpyHostToDevice));
    }

    checkCudaErrors(cudaMalloc(&dWeight, 4*c.f_k*c.f_c*c.f_w*c.f_h)); // weights

    if (dWeight != NULL)
    {
	checkCudaErrors(cudaMemcpy(dWeight, weight, 4*c.f_k*c.f_c*c.f_w*c.f_h, cudaMemcpyHostToDevice));
    }

    if (dInput == NULL || dOutput == NULL || dWeight == NULL)
    {
	    std::cout << "Insufficient memory" << std::endl;
	    return 0;
    }
    std::cout << "Ptrs: Input: " << dInput  << "\tOutput: " << dOutput << "\tWeights: " << dWeight << "\tWS: " << workSpace << std::endl;
    checkCUDNN(cudnnConvolutionBackwardFilter(cudnnHandle, &alpha, iDesc, dInput, oDesc, dOutput, cDesc, algs[selectedAlgNum], workSpace, sizeInBytes, &beta, fDesc, dWeight));

    checkCudaErrors(cudaMemcpy(weight, dWeight, 4*c.f_c*c.f_k*c.f_w*c.f_h, cudaMemcpyDeviceToHost));

    for (int i = 0; i < c.f_c*c.f_k*c.f_w*c.f_h; i++)
    {
	float_byte.f = weight[i];
	printf("%f %x %f\n", weight[i], float_byte.byte, float_byte.f);
    }
}
