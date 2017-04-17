swig -c++ -python KGInt.i
icl -O3 -c KGInt.cpp KGInt_wrap.cxx -I C:\Users\qati\Anaconda3\Lib\site-packages\numpy\core\include
LINK /OUT:_KGInt.pyd KGInt.obj KGInt_wrap.obj /DLL