REM This file builds KGInt module

REM building the Python C++ interface
swig -c++ -python KGInt.i

REM building the CPU implementation with Intel compiler
clang++  -std=c++14 -O3 -c KGInt.cpp KGInt_wrap.cxx -I C:\Users\qati\Anaconda3\Lib\site-packages\numpy\core\include

REM building the GPU CUDA implementation
REM nvcc -o KGInt_CUDA.obj  -rdc=true -arch sm_50 -m64 -O3 -c KGInt.cu

REM link CUDA and CPU implementation to one lib
REM lib /nologo /out:KGInt.lib KGInt.obj KGInt_CUDA.obj

REM Test program
REM -G -g & cuda-memcheck
REM nvcc -arch sm_50 -rdc=true -O3 -G -g  -o test test.cpp -L . -l KGInt

REM Python module
REM old version LINK /OUT:_KGInt.pyd KGInt.obj KGInt_CUDA.obj KGInt_wrap.obj cudart.lib /DLL
REM nvcc -arch sm_50 --shared -rdc=true -o _KGInt.pyd KGInt.obj KGInt_CUDA.obj KGInt_wrap.obj
clang++ -fPIC -std=c++14 -O3 -shared -o _KGInt.pyd KGInt.o KGInt_wrap.o

del /s *.obj *.lib *.exp *.pdb *.o