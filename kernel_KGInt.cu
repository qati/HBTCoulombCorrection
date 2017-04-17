#include <iostream>
#include <string>
#include <iomanip>
#include <chrono> 

#include <nvfunctional>
#include <float.h>
#include <nppdefs.h>

#define M_PI acos(-1.0)

template <class T=float, int N=8, int halfN=N/2> class KGInt{
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
    _eps = DBL_EPSILON*50.;
    _min = NPP_MINABS_64F/_eps;
  }
  
   __device__ double integrate(const nvstd::function<void(const T&, T&)>&, const T&, const T&, const T&);
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


template<class T,int N, int halfN> __device__ double KGInt<T,N, halfN>::integrate(const nvstd::function<void(const T&, T&)>& f, const T& a, const T& b, const T& error)
{  
  func = f;
  targetError = error;
  double z = a;
  double dz = 1e-8;
  double integral = 0.0;
  int j=0;
  while(z<b){
    for(j=0;j<1000;j++){
      center     = z+0.5*dz;
      halfLength = 0.5*dz;
      integrateOneStep();
      if (estimatedError<=targetError) break;
      dz *=targetError/estimatedError;
    }
    integral += integralKronrod;
    z += dz;
    if (estimatedError<targetError){
      dz *= 2;
    }
  }
  return integral;
}



extern "C" {
  __global__ void integrate(double * res, double * rs, double * alphas, double * errors, double * errors_rng){
      int idx = blockDim.x*blockIdx.x+threadIdx.x;
      KGInt<double> imod;
      res[idx] = imod.integrate([&alpha=alphas[idx], &r=rs[idx]](const double& x, double& y)->void{
         y = x*sin(x*r)*exp(-pow(x, alpha));
      }, .0, powf(-logf(errors_rng[idx]),1/alphas[idx]), errors[idx]);
      res[idx] /= 2*M_PI*M_PI*rs[idx];
  }
}


int main(int argc, char **argv)
{
  if (argc<3){
    std::cerr << "Pass r and alpha!" << std::endl;
    return -1;
  }
  const int threads = 512, blocks=10;
  const int N = threads*blocks;
  
  double rs[N], alphas[N], *d_rs, *d_alphas;
  double errors[N], errors_rng[N],  *d_errors, *d_errors_rng;
  double * d_res;
  double res[N];

  cudaMalloc(&d_rs, N*sizeof(double)); 
  cudaMalloc(&d_alphas, N*sizeof(double));
  cudaMalloc(&d_errors, N*sizeof(double));
  cudaMalloc(&d_errors_rng, N*sizeof(double));
  cudaMalloc(&d_res, N*sizeof(double));

  for (int i = 0; i < N; i++) {
    rs[i] = std::stod(argv[1]);
    alphas[i] = std::stod(argv[2]);
    errors[i] = 1e-8;
    errors_rng[i] = 1e-9;
  }
  
  std::cout<<"r="<<rs[0]<<", alpha="<<alphas[0]<<std::endl;

  cudaMemcpy(d_rs, rs, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_alphas, alphas, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_errors, errors, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_errors_rng, errors_rng, N*sizeof(double), cudaMemcpyHostToDevice);

  auto start = std::chrono::high_resolution_clock::now();
  integrate<<<blocks, threads>>>(d_res,d_rs, d_alphas, d_errors, d_errors_rng);
  cudaError_t err = cudaGetLastError();
  cudaDeviceSynchronize();
  if (err != cudaSuccess) 
    std::cerr << "Error: "<< cudaGetErrorString(err) << std::endl;
  auto finish = std::chrono::high_resolution_clock::now();
  
  cudaMemcpy(res, d_res, N*sizeof(double), cudaMemcpyDeviceToHost);
  
  cudaFree(d_rs);
  cudaFree(d_alphas);
  cudaFree(d_errors);
  cudaFree(d_errors_rng);
  cudaFree(d_res);
  
  double ti = std::chrono::duration_cast<std::chrono::milliseconds>(finish-start).count();
  std::cout << "Elapsed time: " << std::setprecision(5)<< ti<< " ms\n";
  
  std::cerr << std::setprecision(15) << res[0] << std::endl;

  double a=0.0;
  for (int i = 1; i < N; i++) {
    a += abs(res[i]-res[0]);
  }
    std::cerr << std::setprecision(15) << a << std::endl; 

  return 0;
}