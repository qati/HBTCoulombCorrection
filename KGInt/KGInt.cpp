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

#include "KGInt.hpp"

#define M_PI acos(-1.0)


using namespace std;

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


double integrate(const int& N, double * result, double * rs, double * alphas, double *errors, double * error_rngs)
{
    KGInt<double> m[N];
    
    vector< function<void(const double&, double&)> > funcs;
    
    for(int i=0;i<N;i++){
        funcs.push_back([&alpha=alphas[i], &r=rs[i]](const double& x, double& y)->void{
            y = x*sin(x*r)*exp(-pow(x, alpha));
        });
    }
    
    vector< future< tuple<double, double> > > tasks;
    auto start = std::chrono::high_resolution_clock::now();
    for(int i=0;i<N;i++){
      tasks.push_back(async(launch::async, &KGInt<double>::integrate, &m[i], funcs[i], 0., pow(-log(error_rngs[i]),1/alphas[i]), errors[i]));
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