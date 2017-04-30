#ifndef __FUNCTIONS_HPP__
#define __FUNCTIONS_HPP__

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

#ifndef M_PI
#define M_PI acos(-1.0)
#endif

/**
 * 3D Levy
 */

template <class T> CUDA_HOSTDEV inline T Levy3D(const T& x, const T& alpha, const T& r){
    return x*sin(x*r)*exp(-pow(x, alpha));
}

template <class T> CUDA_HOSTDEV inline T Levy3D_int_upper_limit(const T& alpha, const T& error){
    return pow(-log(error),1./alpha);
}


/**
 * Gamma function using Lanczos approximation.
 */
 
 #ifdef __CUDACC__
 namespace GammaConstsCUDA{
 __device__ const selected g = 9.;
 __device__ const selected coeff[11]={
     1.000000000000000174663, 5716.400188274341379136, -14815.30426768413909044, 14291.49277657478554025, -6348.160217641458813289, 1301.608286058321874105, 
     -108.1767053514369634679, 2.605696505611755827729, -0.7423452510201416151527e-2, 0.5384136432509564062961e-7, -0.4023533141268236372067e-8
 };
 }
 #endif
 
 template<class numtype=double> class Gamma
 {
 private:
     static const int N = 11;
     #ifndef __CUDACC__
     static constexpr numtype g = 9.;
     static constexpr numtype coeff[N]={
         1.000000000000000174663, 5716.400188274341379136, -14815.30426768413909044, 14291.49277657478554025, -6348.160217641458813289, 1301.608286058321874105, 
         -108.1767053514369634679, 2.605696505611755827729, -0.7423452510201416151527e-2, 0.5384136432509564062961e-7, -0.4023533141268236372067e-8
     };
     #endif
     complex<numtype> G, A_g;
     int i;
 public:
     CUDA_HOSTDEV const complex<numtype>& operator()(const complex<numtype>& z){
         #ifdef __CUDACC__
         using GammaConstsCUDA::g;
         using GammaConstsCUDA::coeff;
         #endif
         A_g = coeff[0];
         for(i=1;i<N;i++){
             A_g += coeff[i] / (z+numtype(i-1));
         }
         G = sqrt(2*M_PI) * pow(z+g-0.5, z-0.5)*exp(-(z+g-0.5))*A_g;
         return G;
     }
 };
#ifndef __CUDACC__
template<class numtype> constexpr numtype Gamma<numtype>::coeff[];
template<class numtype> constexpr numtype Gamma<numtype>::g;
template<class numtype> const int Gamma<numtype>::N;
#endif


/**
 * Hypergeometric function: 1F1(a,b,z)
 */
template<class T > class Hypergeometric{
private:
    Gamma<T> gamma;
    T eps;
    complex<T> a,b;
    complex<T> result1, result2, result3, term1, term2;
    complex<T> amb,gbdga, gbdgbma;
    complex<T> g1,g2;
    unsigned long int n;
public:
    
    CUDA_HOSTDEV Hypergeometric(const T& _eps, const complex<T>& _a, const complex<T>& _b){
        set_ab(_a, _b);
        set_eps(_eps);
    }
    
    CUDA_HOSTDEV Hypergeometric(const T& _eps){
        set_eps(_eps);
    }
    
    CUDA_HOSTDEV Hypergeometric(){}
    
    CUDA_HOSTDEV void set_eps(const T& _eps){
        eps = _eps;
    }
    
    CUDA_HOSTDEV void set_ab(const complex<T>& _a, const complex<T>& _b){
        a       = _a;
        b       = _b;
        amb     = a-b;
        gbdgbma = gamma(b)/gamma(-amb);
        gbdga   = gamma(b)/gamma(a);
    }
    
    CUDA_HOSTDEV complex<T> operator()(const complex<T>& z){
        if (abs(z)<30.){
            result3 = F_series(z);
        } else {
            result3 = F_asym(z);
        }
        return result3;
    }

private:
    CUDA_HOSTDEV const complex<T>& F_series(const complex<T>&);
    CUDA_HOSTDEV const complex<T>& F_asym(const complex<T>&);
    CUDA_HOSTDEV void G(const complex<T>&);
};


template<class T> CUDA_HOSTDEV const complex<T>& Hypergeometric<T>::F_series(const complex<T>& z){
    result1 = 0.;
    for(term1=1.,n=0;abs(term1)>eps;++n){
        result1 += term1;
        term1 *= (a+complex<T>(n))*z/(b+complex<T>(n))/(n+1.);
    }
    return result1;
}

template<class T> CUDA_HOSTDEV void Hypergeometric<T>::G(const complex<T>& z){
    g1 = 0.;
    g2 = 0.;
    for(term1=1.,term2=1.,n=0;abs(term1)>eps && abs(term2)>eps;++n){
        g1 += term1;
        g2 += term2;
        // a = a, b=a-b+1 ambp1
        term1 *= ((a+complex<T>(n))*(amb+complex<T>(n+1)))/(-z*(n+1.));
        // a=b-a, b=1.-a,
        term2 *= ((-amb+complex<T>(n))*(complex<T>(n+1)-a))/(z*(n+1.));
    }
}

template<class T> CUDA_HOSTDEV const complex<T>& Hypergeometric<T>::F_asym(const complex<T>& z){
    G(z);
    result2 = gbdgbma*pow(-z,-a)*g1+gbdga*exp(z)*pow(z,amb)*g2;
    return result2;
}

/*template< class T, class T2 > CUDA_HOSTDEV const T2& Hypergeometric<T, T2>::FFF(const T& y){
    result1 = 0.;
    result2 = 0.;
    result3 = 0.;
    z1 = z2 = z3 = 0.;
    z1.imag(-kr*(1-y));
    z2.imag(+kr*(1-y));
    z3.imag(+kr*(1+y));
    a = b = 1.;
    nmax = 0;
    if (kr<0.5) nmax = 10;
    a.imag(eta);
    b.imag(-eta);
    for(n=0, term1=1., term2=1., term3=1.;(nmax>0 && n<nmax) || (abs(term1)>eps || abs(term2)>eps || abs(term3)>eps);n++){
        result1 += term1;
        result2 += term2;
        result3 += term3;
        denom = 1./ ( (n+1.)*(n+1.) );
        term1 *= (a+T2(n))*denom*z1;
        tmp    = (b+T2(n))*denom;
        term2 *= tmp*z2;
        term3 *= tmp*z3;
    }
    result1 *= (result2+result3);
    return result1;
}*/

#endif