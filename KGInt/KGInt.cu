#include <iostream>
#include <string>
#include <iomanip>
#include <chrono> 


#include <thrust/tuple.h>
#include <thrust/complex.h>
#include <thrust/device_vector.h>
#include <nvfunctional>
#include <float.h>
#include <nppdefs.h>

#include "KGInt.hpp"

using namespace thrust;

#define DOUBLE_PREC

template <class T> using vector= device_vector<T>;

#ifndef M_PI
#define M_PI acos(-1.0)
#endif

#ifdef DOUBLE_PREC
typedef double selected;
#else
typedef float selected;
#endif

#include "functions.hpp"


template <class T=selected, int N=8, int halfN=N/2> class KGInt{
  T weights[N], nodes[N];
  T targetError, estimatedError;
  T integralKronrod, integralGauss, absIntegral, absDiffIntegral;
  T f1[N], f2[N];
  T fValue1, fValue2;
  T center, halfLength, mean;
  T _min;
  T _eps;
  int i,idx;
  nvstd::function<void(const T&, T&)> func;
  
  __device__ void integrateOneStep();
  
public:
  __device__ KGInt():
    nodes{
          0.99145537112081263920685469752632851664204433837033,
          0.94910791234275852452618968404785126240077093767062,
          0.86486442335976907278971278864092620121097230707409,
          0.74153118559939443986386477328078840707414764714139,
          0.58608723546769113029414483825872959843678075060436,
          0.40584515137739716690660641207696146334738201409937,
          0.20778495500789846760068940377324491347978440714517,
          0.00000000000000000000000000000000000000000000000000
      },
      weights{
          0.02293532201052922496373200805896959199356081127575,
          0.06309209262997855329070066318920428666507115721155,
          0.10479001032225018383987632254151801744375665421383,
          0.14065325971552591874518959051023792039988975724800,
          0.16900472663926790282658342659855028410624490030294,
          0.19035057806478540991325640242101368282607807545536,
          0.20443294007529889241416199923464908471651760418072,
          0.20948214108472782801299917489171426369776208022370
    }{
    // Gauss-nodes: 1, 3, 5, 7
    #ifdef DOUBLE_PREC
    _eps = DBL_EPSILON*50.;
    _min = NPP_MINABS_64F/_eps;
    #else
    _eps = FLT_EPSILON*50.;
    _min = NPP_MINABS_32F/_eps;
    #endif
  }
  
   __device__ tuple<double, double> integrate(const nvstd::function<void(const T&, T&)>&, const T&, const T&, const T&);
};

template<class T, int N, int halfN> __device__ void KGInt<T, N, halfN>::integrateOneStep(){
    integralGauss   = .0;
    absIntegral     = .0;
    for(i=0;i<(halfN-1);i++){
      idx = 2*i+1;
      func(center+nodes[idx]*halfLength, fValue1);
      func(center-nodes[idx]*halfLength, fValue2);
      integralGauss += weights[idx]*(fValue1+fValue2);
      absIntegral   += weights[idx]*(abs(fValue1)+abs(fValue2));
      f1[idx] = fValue1;
      f2[idx] = fValue2;
    }
    func(center, fValue1);
    integralGauss += weights[N-1]*fValue1;
    absIntegral   += weights[N-1]*abs(fValue1);
    f1[N-1] = fValue1;
    
    integralKronrod = integralGauss;
    for(i=0;i<halfN;i++){
      idx = 2*i;
      func(center+nodes[idx]*halfLength, fValue1);
      func(center-nodes[idx]*halfLength, fValue2);
      integralKronrod += weights[idx]*(fValue1+fValue2);
      absIntegral     += weights[idx]*(abs(fValue1)+abs(fValue2));
      f1[idx] = fValue1;
      f2[idx] = fValue2;
    }
    
    mean = integralKronrod/2.;
    absDiffIntegral = weights[N-1]*abs(f1[N-1]-mean);
    for(i=0;i<(N-1);i++){
      absDiffIntegral += weights[i]*(abs(f1[i]-mean)+abs(f2[i]-mean));
    }
    
    absIntegral *= halfLength;
    absDiffIntegral *= halfLength;
    
    estimatedError = abs(integralKronrod-integralGauss)*halfLength;
    
    integralKronrod *= halfLength;
    
    if (absDiffIntegral!=0. && estimatedError!= 0.){
      estimatedError = absDiffIntegral*min(1.,pow(estimatedError*200./absDiffIntegral,1.5));
    }
    
    if (absIntegral>_min){
      estimatedError = max(estimatedError, absIntegral*_eps);
    }
}


template<class T,int N, int halfN> __device__ tuple<double,double> KGInt<T,N, halfN>::integrate(const nvstd::function<void(const T&, T&)>& f, const T& a, const T& b, const T& error)
{  
  func = f;
  targetError = error;
  double z = a;
  double dz = 1e-8;
  double integral = 0.0;
  double ierror = 0.0;
  double bintegral =0.0;
  int j=0;
  int k=1;
  int count = 0;
  estimatedError = targetError;
  while(z<b){
    for(j=0;j<1000;j++){
      if (j!=0) dz *= targetError/estimatedError;
      center     = z+0.5*dz;
      halfLength = 0.5*dz;
      integrateOneStep();
      if (estimatedError<=targetError) break;
    }
    integral += integralKronrod;
    ierror += estimatedError;
    if (abs(integral-bintegral)<targetError){
        count += 1;
        if (count>50){
            return make_tuple(integral, ierror/float(k));
        }
    } else {
        count = 0;
        k += 1;
    }
    bintegral = integral;
    z += dz;
    if (estimatedError<targetError){
      dz *= 2;
    }
  }
  return make_tuple(integral, ierror/float(k));
}


__global__ void integrateLevy(double * res, selected * rs, selected * alphas, selected * errors, selected * errors_rng){
      int idx = blockDim.x*blockIdx.x+threadIdx.x;
      KGInt<selected> imod;
      selected alpha = alphas[idx];
      selected r     = rs[idx];
      auto imv = imod.integrate([&alpha, &r](const selected& x, selected& y)->void{
          y = Levy3D(x, alpha, r);
      }, .0, Levy3D_int_upper_limit(alpha, errors_rng[idx]), errors[idx]);
      res[2*idx] = get<0>(imv)/(2*M_PI*M_PI*rs[idx]);
      res[2*idx+1] = get<1>(imv)/(2*M_PI*M_PI*rs[idx]);
}

__global__ void integrateA(double * res, selected * etas, selected * krs, selected * errors, selected * int_errors){
      int idx = blockDim.x*blockIdx.x+threadIdx.x;
      KGInt<selected> imod;
      selected eta = etas[idx];
      Hypergeometric<selected> hyp1,hyp2;
      hyp1.set_eps(errors[idx]);
      hyp2.set_eps(errors[idx]);
      hyp1.set_ab(complex<double>(1., eta), complex<double>(1.,0.));
      hyp2.set_ab(complex<double>(1.,-eta), complex<double>(1.,0.));
      auto func = [h1=&hyp1,h2=&hyp2,kr=krs[idx] ](const double& y, double& r)->void{
          r = (   (*h1)(complex<double>(-kr*(1-y)))*(
                  (*h2)(complex<double>( kr*(1-y)))+
                  (*h2)(complex<double>( kr*(1+y)))
          )).real();
      };
      
      auto imv = imod.integrate(func, -1., 1., int_errors[idx]);
      res[2*idx]   = get<0>(imv)*2*M_PI*eta/(exp(2*M_PI*eta)-1);
      res[2*idx+1] = get<1>(imv)*2*M_PI*eta/(exp(2*M_PI*eta)-1);
}


std::string setGPU(int device)
{
    cudaError_t err;
    int countDevice;
    cudaGetDeviceCount(&countDevice);
    err = cudaGetLastError();
    if (err != cudaSuccess){
        return std::string("cudaGetDeviceCount error: ") + std::string(cudaGetErrorString(err));
    }
    if (device>=countDevice){
        return "Unavailable device! You selected device "+std::to_string(device)+" but the cudaDeviceCount="+std::to_string(countDevice);
    }
    cudaSetDevice(device);
    err = cudaGetLastError();
    if (err != cudaSuccess){
        return std::string("cudaSetDevice error: ") + std::string(cudaGetErrorString(err));
    }
    return "Device "+std::to_string(device)+" selected!";
}


double integrateLevyGPU(const int& blockNum, const int& threadNum, double * result, double * rs, double * alphas, double *errors, double * error_rngs)
{
    selected *d_rs, *d_alphas, *d_errors, *d_error_rngs;
    double * d_res;
    
    const int N = blockNum*threadNum;
    
    cudaMalloc(&d_rs, N*sizeof(selected)); 
    cudaMalloc(&d_alphas, N*sizeof(selected));
    cudaMalloc(&d_errors, N*sizeof(selected));
    cudaMalloc(&d_error_rngs, N*sizeof(selected));
    cudaMalloc(&d_res, 2*N*sizeof(double));
    
    selected * _rs = new selected[N];
    selected *_alphas = new selected[N];
    selected * _errors = new selected[N];
    selected * _error_rngs = new selected[N];
    for(int i=0;i<N;i++){
        _rs[i] = (selected)rs[i];
        _alphas[i] = (selected)alphas[i];
        _errors[i] = (selected)errors[i];
        _error_rngs[i] = (selected)error_rngs[i];
    }
        
    cudaMemcpy(d_rs, _rs, N*sizeof(selected), cudaMemcpyHostToDevice);
    cudaMemcpy(d_alphas, _alphas, N*sizeof(selected), cudaMemcpyHostToDevice);
    cudaMemcpy(d_errors, _errors, N*sizeof(selected), cudaMemcpyHostToDevice);
    cudaMemcpy(d_error_rngs, _error_rngs, N*sizeof(selected), cudaMemcpyHostToDevice);
    
    auto start = std::chrono::high_resolution_clock::now();
    integrateLevy<<<blockNum, threadNum>>>(d_res,d_rs, d_alphas, d_errors, d_error_rngs);
    cudaDeviceSynchronize();
    auto finish = std::chrono::high_resolution_clock::now();
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
      std::cerr << "Error: "<< cudaGetErrorString(err) << std::endl;
      return -1.;
    }
    
    cudaMemcpy(result, d_res, 2*N*sizeof(double), cudaMemcpyDeviceToHost);
    
    cudaFree(d_rs);
    cudaFree(d_alphas);
    cudaFree(d_errors);
    cudaFree(d_error_rngs);
    cudaFree(d_res);
    
    delete [] _rs;
    delete [] _alphas;
    delete [] _errors;
    delete [] _error_rngs;

    double ti = std::chrono::duration_cast<std::chrono::milliseconds>(finish-start).count();
    return ti;
}

double integrateAGPU(const int& blockNum, const int& threadNum, double * result, double * etas, double * krs, double *errors, double * int_errors)
{
    selected *d_etas, *d_krs, *d_errors, *d_int_errors;
    double * d_res;
    
    const int N = blockNum*threadNum;
    
    cudaMalloc(&d_etas, N*sizeof(selected)); 
    cudaMalloc(&d_krs, N*sizeof(selected));
    cudaMalloc(&d_errors, N*sizeof(selected));
    cudaMalloc(&d_int_errors, N*sizeof(selected));
    cudaMalloc(&d_res, 2*N*sizeof(double));
    
    selected * _etas       = new selected[N];
    selected * _krs        = new selected[N];
    selected * _errors     = new selected[N];
    selected * _int_errors = new selected[N];
    for(int i=0;i<N;i++){
        _etas[i]       = (selected)etas[i];
        _krs[i]        = (selected)krs[i];
        _errors[i]     = (selected)errors[i];
        _int_errors[i] = (selected)int_errors[i];
    }
        
    cudaMemcpy(d_etas, _etas, N*sizeof(selected), cudaMemcpyHostToDevice);
    cudaMemcpy(d_krs,  _krs, N*sizeof(selected), cudaMemcpyHostToDevice);
    cudaMemcpy(d_errors, _errors, N*sizeof(selected), cudaMemcpyHostToDevice);
    cudaMemcpy(d_int_errors, _int_errors, N*sizeof(selected), cudaMemcpyHostToDevice);
    
    auto start = std::chrono::high_resolution_clock::now();
    integrateA<<<blockNum, threadNum>>>(d_res,d_etas, d_krs, d_errors, d_int_errors);
    cudaDeviceSynchronize();
    auto finish = std::chrono::high_resolution_clock::now();
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
      std::cerr << "Error: "<< cudaGetErrorString(err) << std::endl;
      return -1.;
    }
    
    cudaMemcpy(result, d_res, 2*N*sizeof(double), cudaMemcpyDeviceToHost);
    
    cudaFree(d_etas);
    cudaFree(d_krs);
    cudaFree(d_errors);
    cudaFree(d_int_errors);
    cudaFree(d_res);
    
    delete [] _etas;
    delete [] _krs;
    delete [] _errors;
    delete [] _int_errors;

    double ti = std::chrono::duration_cast<std::chrono::milliseconds>(finish-start).count();
    return ti;
}
