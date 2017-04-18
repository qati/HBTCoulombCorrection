#!/bin/bash

# This file builds KGInt module

#building the Python C++ interface
~/.usr/bin/swig -c++ -python KGInt.i

# building the CPU implementation with Intel compiler
g++ -fPIC -std=c++14 -O3 -c KGInt.cpp KGInt_wrap.cxx -I /home/qati/.anaconda3/pkgs/numpy-1.11.1-py35_0/lib/python3.5/site-packages/numpy/core/include -I/home/qati/.anaconda3/include/python3.5m

# building the GPU CUDA implementation
nvcc -Xcompiler -fPIC -std=c++11 -o KGInt_CUDA.o  -rdc=true -arch sm_50 -m64 -O3 -c KGInt.cu

# link CUDA and CPU implementation to one lib
ar rcs libKGInt.a KGInt.o KGInt_CUDA.o

# Test program
# -G -g & cuda-memcheck
nvcc -std=c++11 -arch sm_50 -rdc=true -O3 -G -g  -o test test.cpp -L . -l KGInt

# Python module
# old version LINK /OUT:_KGInt.pyd KGInt.obj KGInt_CUDA.obj KGInt_wrap.obj cudart.lib /DLL
nvcc -std=c++11 -arch sm_50 --shared -rdc=true -o _KGInt.so KGInt.o KGInt_CUDA.o KGInt_wrap.o


rm *.o