CUDA_HOME?=/usr/local/cuda
CUDA_INC:=${CUDA_HOME}/include
NVCC:=${CUDA_HOME}/bin/nvcc
ARCH?=sm_61
PYTHON?=$(shell which python)
TF_INC?=$(shell ${PYTHON} -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB?=$(shell ${PYTHON} -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
CXX?=g++
EXTERTNAL_INC?= -I ${TF_INC}/external/nsync/public

all: roi_pooling.so

roi_pooling_op.cu.o: roi_pooling_op.cu.cc
	${NVCC} -std=c++11 -c -o roi_pooling_op.cu.o roi_pooling_op.cu.cc -I ${TF_INC} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -arch=${ARCH}


roi_pooling.so: roi_pooling_op.cu.o roi_pooling_op.cc
	${CXX} -std=c++11 -shared -D_GLIBCXX_USE_CXX11_ABI=0 -o roi_pooling.so roi_pooling_op.cc roi_pooling_op.cu.o -I ${TF_INC} -D GOOGLE_CUDA=1 -fPIC -lcudart -L ${CUDA_HOME}/lib64 -L ${TF_LIB} -ltensorflow_framework ${EXTERTNAL_INC}

clean:
	-rm *.so
	-rm *.o
