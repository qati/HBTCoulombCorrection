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

#include <unordered_map>

/**
 * General functions.
 */
template<class T> inline T SQR(const T& x){
    return x*x;
}


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
 public:
     CUDA_HOSTDEV  complex<numtype> operator()(const complex<numtype>& z){
         #ifdef __CUDACC__
         using GammaConstsCUDA::g;
         using GammaConstsCUDA::coeff;
         #endif
         complex<numtype> G, A_g;
         int i;
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
 /**
  * Hypergeometric function: 1F1(a,b,z)
  */
 template<class T > class Hypergeometric{
 private:
     Gamma<T> gamma;
     T eps;
     complex<T> a,b;
     complex<T> gbdga, gbdgbma;
     bool table = false;
     unordered_map<unsigned long long int, complex<T> > cacheTable;
     unsigned long long int N;
     T dx, x0, x1; 
     
     inline unsigned long long int hash(const complex<T>& z){
         return floor((z.imag()-x0)/dx);
     }
     
     CUDA_HOSTDEV void fillCache(){
         cacheTable.clear();
         complex<T> z(0., x0);
         for(unsigned long long int i = 1;i<=N;i++){
             cacheTable[hash(z)] = get(z);
             z.imag(x0+dx*i);
         }
     }
     
     inline complex<T> getFromCache(const complex<T>& z){
         auto low = hash(z);
         auto y0 = cacheTable[low-1],
              y1 = cacheTable[low],
              y2 = cacheTable[low+1];
         auto xx0 = x0+dx*(low-1), xx1 = x0+dx*(low), xx2=x0+dx*(low+1);
         auto x = z.imag();
         if (abs(y0)<1e-8){
             y0 = get(complex<T>(0., x0+dx*low-dx));
             cacheTable[low-1] = y0;
         }
         if (abs(y1)<1e-8){
             y1 = get(complex<T>(0., x0+dx*low));
             cacheTable[low] = y1;
         }
         if (abs(y2)<1e-8){
             y2 = get(complex<T>(0., x0+dx*low+dx));
             cacheTable[low+1] = y2;
         }
         //return y0+(y1-y0)*(z.imag()-x0-dx*low)/dx;
         return ( (x-xx1)*(x-xx2)/(xx0-xx1)/(xx0-xx2) )*y0+( (x-xx0)*(x-xx2)/(xx1-xx0)/(xx1-xx2) )*y1 + ( (x-xx1)*(x-xx0)/(xx2-xx1)/(xx2-xx0) )*y2;
     }
     
     CUDA_HOSTDEV void achieveError(){
         dx = (x1-x0)/T(N);
         fillCache();
         T point;
         int i;
         int n;
         int NN=20;
         T dp = (1-0.01)/T(NN);
         complex<T> z(0.,0.);
         complex<T> v;
         cout << "Start N="<<N<<endl;
         while(true){
             n = 0;
             for(i=0;i<NN;i++){
                 point = x0+dx*N*(0.01+dp*i);
                 z.imag(point);
                 v = get(z);
                 auto vv = getFromCache(z);
                 if (abs(vv)<1e-9){
                     auto low = hash(z);
                     cacheTable[low] = get(complex<T>(0., x0+dx*low));
                     vv = getFromCache(z);
                 }
                 if ((abs(vv-v)/abs(v))<1e-7){
                     n++;
                 }
                }
             if (n==NN) break;
             N *= 2;
             dx = (x1-x0)/T(N);
             fillCache();
         }
     }
     
 public:
     
     CUDA_HOSTDEV Hypergeometric(const T& _eps, const complex<T>& _a, const complex<T>& _b){
         set_ab(_a, _b);
         set_eps(_eps);
     }
     
     CUDA_HOSTDEV Hypergeometric(const T& _eps){
         set_eps(_eps);
     }
     
     CUDA_HOSTDEV Hypergeometric(){}
     
     CUDA_HOSTDEV void setCache(T _x0, T _x1, unsigned int _N){
         x0 = _x0;
         x1 = _x1;
         N  = _N;
         dx = (x1-x0)/T(N);
         //achieveError();
         fillCache();
         table = true;
     }
     
     CUDA_HOSTDEV void set_eps(const T& _eps){
         eps = _eps;
     }
     
     CUDA_HOSTDEV void set_ab(const complex<T>& _a, const complex<T>& _b){
         a       = _a;
         b       = _b;
         gbdgbma = gamma(b)/gamma(b-a);
         gbdga   = gamma(b)/gamma(a);
     }
     
     CUDA_HOSTDEV complex<T> get(const complex<T>& z){
         complex<T> result;
         if (abs(z)<30.){
             result = F_series(z);
         } else {
             result = F_asym(z);
         }
         return result;
     }
     
     CUDA_HOSTDEV complex<T> operator()(const complex<T>& z){
         complex<T> r;
         if (table && abs(z)<x1 && abs(z)>x0){
             r = getFromCache(z);
         } else {
             r = get(z);
         }
         return r;
     }
 
 private:
      complex<T> F_series(const complex<T>&);
      complex<T> F_asym(const complex<T>&);
 };
 
 
 template<class T> CUDA_HOSTDEV complex<T> Hypergeometric<T>::F_series(const complex<T>& z){
     complex<T> result = 0.;
     complex<T> term;
     T n, err;
     for(term=1.,n=0.,err=1.;err>eps;++n){
         result += term;
         err = abs(term)/abs(result);
         term *= (a+n)*z/(b+n)/(n+1.);
     }
     return result;
 }
 
 template<class T> CUDA_HOSTDEV complex<T> Hypergeometric<T>::F_asym(const complex<T>& z){
     complex<T> F1 = gbdgbma*pow(-z,-a),
                F2 = gbdga*exp(z)*pow(z,a-b);
     complex<T> result, tmp, term1, term2;
     T err(1), n;
     for(term1=1.,term2=1.,n=0.;err>eps;++n){
         tmp     = F1*term1+F2*term2;
         result += tmp;
         err = abs(tmp)/abs(result);
         // a = a, b=a-b+1 ambp1
         term1 *= (  (a+n)*(a-b+1.+n)  ) / ( -z*(n+1.) );
         // a=b-a, b=1.-a,
         term2 *= ( (b-a+n)*(n+1.-a)  ) / (  z*(n+1.) );
     }
     return result;
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