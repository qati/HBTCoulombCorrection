%module  KGInt

%{
    #define SWIG_FILE_WITH_INIT
    
    #include "KGInt.hpp"
    #include "Coulomb2.hpp"
    #include "Coulomb3.hpp"
%}

%include "std_string.i"
%include "numpy.i"

%init %{
    import_array();
%}

%apply (int DIM1, double* ARGOUT_ARRAY1) {(int n1, double* result)}
%apply (int DIM1, double* IN_ARRAY1){(int n2, double *rs), (int n3, double * alphas), (int n4, double *errors), (int n5, double *error_rngs), (int n5, double *int_errors)}
%apply (int DIM1, double* IN_ARRAY1){(int n2, double *etas), (int n3, double * krs), (int n, double * Lrs)}


%rename (integrateLevyCPU) swig_integrateLevyCPU;
//%rename (integrateLevyGPU) swig_integrateLevyGPU;
%rename (integrateACPU) swig_integrateACPU;
//%rename (integrateAGPU) swig_integrateAGPU;


%inline %{
    double swig_integrateLevyCPU(int n1, double * result, int n2, double * rs, int n3, double * alphas, int n4, double * errors, int n5, double * error_rngs){
        if ( n2!=n3 || n3!=n4 || n4!=n5){
            PyErr_Format(PyExc_ValueError, "Array sizes doesn't match (%d, %d, %d, %d, %d)!", n1, n2, n3, n4, n5);
            return 0;
        }
        return integrateLevyCPU(n2, result, rs, alphas, errors, error_rngs);
    }
    
    /*double swig_integrateLevyGPU(int blockNum, int threadNum, int n1, double * result, int n2, double * rs, int n3, double * alphas, int n4, double * errors, int n5, double * error_rngs){
        if ( n2!=n3 || n3!=n4 || n4!=n5){
            PyErr_Format(PyExc_ValueError, "Array sizes doesn't match (%d, %d, %d, %d)!", n2, n3, n4, n5);
            return 0;
        }
        if ((blockNum*threadNum)!=n2){
            PyErr_Format(PyExc_ValueError, "Array sizes (%d) doesn't match with blocks(%d)*threads(%d)!", n2, blockNum, threadNum);
            return 0;
        }
        return integrateLevyGPU(blockNum, threadNum, result, rs, alphas, errors, error_rngs);
    }*/
    
    double swig_integrateACPU(int n1, double * result, int n2, double * etas, int n3, double * krs, int n4, double *errors, int n5, double * int_errors){
        if ( n2!=n3 || n3!=n4 || n4!=n5){
            PyErr_Format(PyExc_ValueError, "Array sizes doesn't match (%d, %d, %d, %d, %d)!", n1, n2, n3, n4, n5);
            return 0;
        }
        return integrateACPU(n2, result, etas, krs, errors, int_errors);
    }
    
    /*double swig_integrateAGPU(int blockNum, int threadNum, int n1, double * result, int n2, double * etas, int n3, double * krs, int n4, double *errors, int n5, double * int_errors){
        if ( n2!=n3 || n3!=n4 || n4!=n5){
            PyErr_Format(PyExc_ValueError, "Array sizes doesn't match (%d, %d, %d, %d, %d)!", n1, n2, n3, n4, n5);
            return 0;
        }
        return integrateAGPU(blockNum, threadNum, result, etas, krs, errors, int_errors);
    }*/
%}

%include "KGInt.hpp"
%include "Coulomb2.hpp"
%include "Coulomb3.hpp"


%template(Levyd) Levy<double>;
%template(Coulomb2d) Coulomb2<double>;
%template(Coulomb3d) Coulomb3<double>;



%pythoncode %{
    def integrateLevyCPU(rs, alphas, errors, error_rngs, time=True):
        res = _KGInt.integrateLevyCPU(len(rs)*2, rs, alphas, errors, error_rngs)
        if time:
            print("->CPU: elapsed time = %.1f ms"%res[0])
        return res[1].reshape(len(rs),2);
        
    #def integrateLevyGPU(blockNum, threadNum, rs, alphas, errors, error_rngs, time=True):
    #    res = _KGInt.integrateLevyGPU(blockNum, threadNum, len(rs)*2, rs, alphas, errors, error_rngs)
    #    if time:
    #        print("->GPU: elapsed time = %.1f ms"%res[0])
    #    return res[1].reshape(len(rs),2);

    def integrateACPU(etas, krs, errors, int_errors, time=True):
        res = _KGInt.integrateACPU(len(etas)*2, etas, krs, errors, int_errors)
        if time:
            print("->CPU: elapsed time = %.1f ms"%res[0])
        return res[1].reshape(len(etas),2);
        
    #def integrateAGPU(blockNum, threadNum, etas, krs, errors, int_errors, time=True):
    #    res = _KGInt.integrateAGPU(blockNum, threadNum, len(etas)*2, etas, krs, errors, int_errors)
    #    if time:
    #        print("->GPU: elapsed time = %.1f ms"%res[0])
    #    return res[1].reshape(len(etas),2);
    
    def hyp1f1_FFF(n1, kr, eta, eps):
        import numpy as np
        res = _KGInt.hyp1f1_FFF(2*n1, kr, eta, eps)
        return np.array(list(map(complex, res[::2], res[1::2])))
        
    def hyp1f1(a,b,z, x0, x1, N, eps=1e-8):
        import numpy as np
        res = _KGInt.hyp1f1(2*len(z), a, b, z.real, z.imag, eps, x0, x1, N)
        return np.array(list(map(complex, res[::2], res[1::2])))
%}