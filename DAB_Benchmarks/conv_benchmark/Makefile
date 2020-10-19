NVCC = ${CUDA_INSTALL_PATH}/bin/nvcc
CUDNN_PATH = ${CUDA_INSTALL_PATH}/cudnn/7.1

LIBRARIES = -L${CUDA_INSTALL_PATH}/lib64 -L${CUDNN_PATH}/lib64 -I${CUDNN_PATH}/include -lcudart -lcublas_static -lcudnn_static -lculibos -lstdc++ -lm -ldl -lpthread --cudart shared  

GENCODE_FLAGS=-gencode arch=compute_70,code=compute_70 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_53,code=sm_53

target:
	${NVCC} ${LIBRARIES} ${GENCODE_FLAGS} back_benchmark.cpp -o benchmark
