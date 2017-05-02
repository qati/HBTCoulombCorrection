#include <iostream>
#include <cmath>
#include <vector>
#include <functional>
#include <string>
#include <algorithm>
#include <limits>
#include <array>
#include <iomanip>
#include <future>
#include <chrono>
#include <tuple>
#include <complex>

#include "KGInt.hpp"

using namespace std;

#define DOUBLE_PREC

#include "functions.hpp"

#ifndef M_PI
#define M_PI acos(-1.0)
#endif



template <class T=double, int N=8, int halfN=N/2> class KGInt{
  vector<T> weights, nodes;
  T targetError, estimatedError;
  T integralKronrod, integralGauss, absIntegral, absDiffIntegral;
  array<T, 8> f1, f2;
  T fValue1, fValue2;
  T center, halfLength, mean;
  T _min;
  T _eps;
  int i,idx;
  function<void(const T&, T&)> func;
  
  void integrateOneStep();
  
public:
  KGInt(){
    nodes = {
          0.99145537112081263920685469752632851664204433837033,
          0.94910791234275852452618968404785126240077093767062,
          0.86486442335976907278971278864092620121097230707409,
          0.74153118559939443986386477328078840707414764714139,
          0.58608723546769113029414483825872959843678075060436,
          0.40584515137739716690660641207696146334738201409937,
          0.20778495500789846760068940377324491347978440714517,
          0.00000000000000000000000000000000000000000000000000
    };
    weights = {
          0.02293532201052922496373200805896959199356081127575,
          0.06309209262997855329070066318920428666507115721155,
          0.10479001032225018383987632254151801744375665421383,
          0.14065325971552591874518959051023792039988975724800,
          0.16900472663926790282658342659855028410624490030294,
          0.19035057806478540991325640242101368282607807545536,
          0.20443294007529889241416199923464908471651760418072,
          0.20948214108472782801299917489171426369776208022370
    };
    // Gauss-nodes: 1, 3, 5, 7
    _eps = numeric_limits<T>::epsilon()*50.;
    _min = numeric_limits<T>::min()/_eps;
  }
  
  tuple<double, double> integrate(const function<void(const T&, T&)>&, const T&,  const T&,  const T&);
  
};

template<class T, int N, int halfN> void KGInt<T, N, halfN>::integrateOneStep(){
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


template<class T,int N, int halfN> tuple<double, double> KGInt<T,N, halfN>::integrate(const function<void(const T&, T&)>& f,  const T& a,  const T& b,  const T& error)
{  
  func = f;
  targetError = error;
  double z = a;
  double dz = 1e-8;
  double integral = 0.0;
  double integralError = 0.0;
  int j;
  int k=1;
  double integralBefore=0.0;
  unsigned int conv=0;
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
    integralError += abs(estimatedError);
    if (abs(integralBefore-integral)<targetError){
        conv += 1;
        if (conv>50) return make_tuple(integral, integralError/double(k));
    } else {
        conv = 0;
        k += 1;
    }
    integralBefore = integral;
    z += dz;
    if (estimatedError<targetError){
      dz *= 2;
    }
  }
  return make_tuple(integral, integralError/double(k));
}



double integrateLevyCPU(const int& N, double * result, double * rs, double * alphas, double *errors, double * error_rngs)
{
    KGInt<double> m[N];
    
    vector< function<void(const double&, double&)> > funcs;
    
    for(int i=0;i<N;i++){
        funcs.push_back([alpha=alphas[i], r=rs[i]](const double& x, double& y)->void{
            y = Levy3D(x, alpha, r);
        });
    }
    
    vector< future< tuple<double, double> > > tasks;
    auto start = std::chrono::high_resolution_clock::now();
    for(int i=0;i<N;i++){
      tasks.push_back(async(launch::async, &KGInt<double>::integrate, &m[i], funcs[i], 0., Levy3D_int_upper_limit(alphas[i], error_rngs[i]), errors[i]));
    }
    
    for(int i=0;i<N;i++){
        auto res = tasks[i].get();
        result[2*i+0] = get<0>(res)/(2*M_PI*M_PI*rs[i]);
        result[2*i+1] = get<1>(res)/(2*M_PI*M_PI*rs[i]);
    }
    auto finish = std::chrono::high_resolution_clock::now();
    
    double ti = std::chrono::duration_cast<std::chrono::milliseconds>(finish-start).count();
    
    return ti;
}

double integrateACPU(const int& N, double * result, double * etas, double * krs, double *errors, double * int_errors)
{
    KGInt<double> m[N];
    Hypergeometric<double> hyps1[N], hyps2[N];
    
    vector< function<void(const double&, double&)> > funcs;
    
    for(int i=0;i<N;i++){
        hyps1[i].set_eps(errors[i]);
        hyps2[i].set_eps(errors[i]);
        hyps1[i].set_ab(complex<double>(1., etas[i]), complex<double>(1.,0.));
        hyps2[i].set_ab(complex<double>(1.,-etas[i]), complex<double>(1.,0.));
        funcs.push_back([hyp1=&hyps1[i],hyp2=&hyps2[i],kr=krs[i] ](const double& y, double& r)->void{
            r = (   (*hyp1)(complex<double>(-kr*(1-y)))*(
                    (*hyp2)(complex<double>( kr*(1-y)))+
                    (*hyp2)(complex<double>( kr*(1+y)))
            )).real();
        });
    }
    
    vector< future< tuple<double, double> > > tasks;
    auto start = std::chrono::high_resolution_clock::now();
    for(int i=0;i<N;i++){
      tasks.push_back(async(launch::async, &KGInt<double>::integrate, &m[i], funcs[i], -1., 1., int_errors[i]));
    }
    
    for(int i=0;i<N;i++){
        auto res = tasks[i].get();
        result[2*i+0] = get<0>(res)*2*M_PI*etas[i]/(exp(2*M_PI*etas[i])-1);
        result[2*i+1] = get<1>(res)*2*M_PI*etas[i]/(exp(2*M_PI*etas[i])-1);
    }
    auto finish = std::chrono::high_resolution_clock::now();
    
    double ti = std::chrono::duration_cast<std::chrono::milliseconds>(finish-start).count();
    
    return ti;
}

void hyp1f1_FFF(int n1, double * result, double kr, double eta, double eps)
{
    //    A=F(1+1j*eta, 1, -1j*kr*(1-y))*(F(1-1j*eta, 1, 1j*kr*(1-y))+F(1-1j*eta, 1, 1j*kr*(1+y)))

    Hypergeometric<double> hyp1(eps), hyp2(eps);
    complex<double> tmp;
    int N = int(n1/2);
    double dy = 2./double(N);
    double y;
    complex<double> z1,z2,z3;
    hyp1.set_ab(complex<double>(1., eta), complex<double>(1.,0));
    hyp2.set_ab(complex<double>(1.,-eta), complex<double>(1.,0));
    for(int n=0;n<N;n++){
        y = -1+n*dy;
        z1 = complex<double>(0., -kr*(1-y));
        z2 = complex<double>(0.,  kr*(1-y));
        z3 = complex<double>(0.,  kr*(1+y));
        tmp = hyp1(z1)*(hyp2(z2)+hyp2(z3));
        result[2*n+0] = tmp.real();
        result[2*n+1] = tmp.imag();
    }
}


void hyp1f1(int n1, double * result, const complex<double>& a, const complex<double>& b, int n2, double* rs, int n3, double* krs, const double& eps){
    Hypergeometric<double> hyp(eps);
    hyp.set_ab(a,b);
    complex<double> tmp;
    for(int i=0;i<n2;i++){
        tmp = hyp(complex<double>(rs[i], krs[i]));
        result[2*i+0] = tmp.real();
        result[2*i+1] = tmp.imag();
    }
}



/*
double integrateACPU_slow(const int& N, double * result, double * etas, double * krs, double *errors, double * error_rngs)
{
    KGInt<double> m[N];
    Hypergeometric<double, 1e-8> hyps1[N], hyps2[N], hyps3[N];
    
    vector< function<void(const double&, double&)> > funcs;
    
    for(int i=0;i<N;i++){
        hyps1[i].set_ab(complex<double>(1.,etas[i]), 1.);
        hyps2[i].set_ab(complex<double>(1.,-etas[i]), 1.);
        hyps2[i].set_ab(complex<double>(1.,-etas[i]), 1.);
        funcs.push_back([hyp1=hyps1[i],hyp2=hyps2[i],hyp3=hyps3[i], kr=krs[i]](const double& x, double& y)->void{
            y = ( hyp1.F(complex<double>(0., -kr*(1-x))) * (
                  hyp2.F(complex<double>(0., +kr*(1-x))) +
                  hyp3.F(complex<double>(0., +kr*(1+x)))  ) ).real();
        });
    }
    
    vector< future< tuple<double, double> > > tasks;
    auto start = std::chrono::high_resolution_clock::now();
    for(int i=0;i<N;i++){
      tasks.push_back(async(launch::async, &KGInt<double>::integrate, &m[i], funcs[i], -1., 1.));
    }
    
    for(int i=0;i<N;i++){
        auto res = tasks[i].get();
        result[2*i+0] = get<0>(res)*2*M_PI*etas[i]/(exp(2*M_PI*etas[i])-1);
        result[2*i+1] = get<1>(res)*2*M_PI*etas[i]/(exp(2*M_PI*etas[i])-1);
    }
    auto finish = std::chrono::high_resolution_clock::now();
    
    double ti = std::chrono::duration_cast<std::chrono::milliseconds>(finish-start).count();
    
    return ti;
}*/