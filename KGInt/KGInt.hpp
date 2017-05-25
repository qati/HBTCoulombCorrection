#ifndef __KGINT_HPP__
#define __KGINT_HPP__

#include <string>
#include <complex>

/**
 * Selecting GPU!
 */
//std::string setGPU(int device);

/**
 * Calculate Levy integral on CPU
 */
double integrateLevyCPU(const int& N, double * result, double * rs, double * alphas, double *errors, double * error_rngs);

/**
 * Calculate Levy integral on GPU
 */
//double integrateLevyGPU(const int& blockNum, const int& threadNum, double * result, double * rs, double * alphas, double *errors, double * error_rngs);


/**
 * Calculate A^(1)+A^(2) on CPU
 */
 double integrateACPU(const int& N, double * result, double * etas, double * krs, double *errors, double * int_errors);
 
 /**
  * Calculate A^(1)+A^(2) on GPU
  */
 //double integrateAGPU(const int& blockNum, const int& threadNum, double * result, double * etas, double * krs, double *errors, double * int_errors);

/**
 * Hyp1f1 function
 */
 void hyp1f1_FFF(int n1, double * result, double kr, double eta, double eps);
 
 void hyp1f1(int n1, double * result, const std::complex<double>& a, const std::complex<double>& b, int n2, double* rs, int n3, double* krs, const double& eps, double x0, double x1, int N);



/**
 * Levy reader
 */
#include <vector>
#include <cmath>

#ifndef M_PI
#define M_PI acos(-1.0)
#endif

template<class T> class Levy {
private:
    unsigned int N;
    T R, alpha, limit1, limit2, dr, error;
    std::vector<T> Lr;
    
    inline unsigned long long int hash(const T& r){
        return floor((r-limit1)/dr);
    }
    
    T asym_low(const T&);
    T asym_hi(const T&);
    inline T interpolate(const T& r){
        auto low = hash(r);
        return Lr[low]+(Lr[low+1]-Lr[low])*(r-limit1-dr*low)/dr;
    }
public:
    Levy(const T& R, const T& alpha, const T& limit1, const T& limit2, const T& dr, int n, T * Lrs, const T& error){
        this->R = R;
        this->alpha = alpha;
        this->limit1 = limit1;
        this->limit2 = limit2;
        this->dr = dr;
        this->error = error;
        N = n;
        for(int i=0;i<N;i++){
            Lr.push_back(Lrs[i]);
        }
    }
    
    T operator()(const T& r){
        T res;
        if (r<=limit1) res = asym_low(r);
        else if (r>=limit2) res = asym_hi(r);
        else res = interpolate(r);
        return res;
    }
    
    void get(int n1, double * result, int n2, double * rs){
        for(int i=0;i<n2;i++){
            result[i] = (*this)(rs[i]);
        }
        return;
    }

    inline T get_alpha(){
        return alpha;
    }
};

template<class T> T Levy<T>::asym_low(const T& r){
    T term(1.), res(0.);
    for(int k=0;term>error;k++){
        term = ( tgamma((k+3)/alpha)/tgamma(k+3) ) * sin(M_PI*(k+3)/2.)  * (k+2)*(k+1) * pow(r, k);
        res  += term;
    }
    return -res/(2*M_PI*M_PI*alpha);
}

template<class T> T Levy<T>::asym_hi(const T& r){
    T term(1.), res(0.);
    for(int k=1;term>error;k++){
        term = (-1*(k&1)+1*!(k&1)) * ( tgamma(alpha*k)/tgamma(k) ) * sin(M_PI*alpha*k/2.) * (alpha*k+1.) / pow(r, alpha*k+3);
        res  += term;
        break;
    }
    return -alpha*res/(2*M_PI*M_PI);
}



/**
 * Coulomb correction: 2 particle
 */
#include "Coulomb2.hpp"

/**
 * Coulomb correction: 3 particle
 */
#include "Coulomb3.hpp"


double intLevy(Levy<double> * levy, double maxr, double error);
/*{
    KGInt<double> m;
    auto res = m.integrate([l=levy](const double& r, double& y){
        y = (*l)(r);
    }, 0., maxr, error);
    
    return get<0>(res);
}*/

#endif