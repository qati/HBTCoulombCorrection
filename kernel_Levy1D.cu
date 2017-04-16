#include <curand.h>
#include <curand_kernel.h>

extern "C" {
    __global__ void init( unsigned long long int* seed, curandState * state){
        int id = threadIdx.x;
        curand_init(*seed, id, 0, &state[id]);
    }
    
    __device__ float pi(float &x, float *pars){
      return expf(-powf(fabsf(x*pars[2]), pars[1]));
    }
    
    __device__ float f(float &x, float *pars){
      return cosf(x*pars[0]);
    }
    
    __global__  void mcmc(curandState* states, unsigned int * num_samples, float * Pars, int * npar, 
                          float * Sigma, float * result){
        int id            = threadIdx.x;
        curandState state = states[id];
        unsigned int N    = *num_samples;
        float sigma       = *Sigma;
        float *pars       = new float[*npar];
        memcpy(pars, &Pars[*npar*id], *npar*sizeof(float));
        
        float xi = curand_uniform(&state);
        float xg = 0.0;
        float s  = 0.0;
        for(unsigned int i=0;i<N;i++){
            xg = sigma*curand_normal(&state)+xi;
            if (curand_uniform(&state)<pi(xg, pars)/pi(xi, pars)){
              xi = xg;
            }
            s += f(xi, pars);
        }
        result[id] = s/float(N);
        delete pars;
    }
}