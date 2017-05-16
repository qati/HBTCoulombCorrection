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

#include "KGInt_class.hpp"



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


void hyp1f1(int n1, double * result, const complex<double>& a, const complex<double>& b, int n2, double* rs, int n3, double* krs, const double& eps, double x0, double x1, int N){
    Hypergeometric<double> hyp(eps);
    hyp.set_ab(a,b);
    hyp.setCache(x0, x1, N);
    complex<double> tmp;
    for(int i=0;i<n2;i++){
        tmp = hyp(complex<double>(rs[i], krs[i]));
        result[2*i+0] = tmp.real();
        result[2*i+1] = tmp.imag();
    }
}


double intLevy(Levy<double> * levy, double maxr, double error)
{
    KGInt<double> m;
    auto res = m.integrate([l=levy](const double& r, double& y){
        y = (*l)(r)*r*r;
    }, 0., maxr, error);
    
    return get<0>(res);
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