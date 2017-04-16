#include <curand.h>
#include <curand_kernel.h>


extern "C" {
    __global__ void init( unsigned long long int* seed, curandState * state){
        int id = threadIdx.x;
        curand_init(*seed, id, 0, &state[id]);
    }
    
    __device__ void pi(const float &x, float *pars, float &p){
        p = expf(-powf(fabsf(x*pars[2]), pars[1]));
    }
    
    __device__ void f(const float &x, float *pars, float &s){
        s += x*sinf(x*pars[0]);
    }
    
    __global__  void mcmc(curandState* states, unsigned int * num_samples, float * Pars, int * npar, 
                          float * Sigma, float * result){
        int id            = threadIdx.x;
        curandState state = states[id];
        unsigned int N    = *num_samples;
        float sigma       = *Sigma;
        float *pars       = new float[*npar];
        memcpy(pars, &Pars[*npar*id], *npar*sizeof(float));
        
        float xi   = curand_uniform(&state);
        float xg   = 0.0;
        float s    = 0.0;
        float p_xi = 0.0;
        float p_xg = 0.0;
        pi(xi, pars, p_xi);
        for(unsigned int i=0;i<N;i++){
            xg = sigma*curand_normal(&state)+xi;
            pi(xg, pars, p_xg);
            if (curand_uniform(&state)<(p_xg/p_xi)){
              xi   = xg;
              p_xi = p_xg;
            }
            f(xi, pars, s);
        }
        result[id] = s/float(N);
        delete pars;
    }
}