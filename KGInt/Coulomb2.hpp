#ifndef __COULONB_HPP__
#define __COULONB_HPP__

#include <random>
#include <complex>

using namespace std;

#include "functions.hpp"
#include "KGInt_class.hpp"


template<class T> class Coulomb2 {
private:
    Levy<T>* levy;
    Hypergeometric<T> hyp;
    Gamma<T> gamma;
    KGInt<T> m;
    T eps;
    T pion_mass;
    T eta, k;
    complex<T> norm;
    T sigma;
    T R;
    
    random_device rd;
    mt19937 generator;
    uniform_real_distribution<T> uniform, uniform_theta, uniform2;
    normal_distribution<T> normal;
    
    inline complex<T> psi1(const T& r, const T& costheta){
        return norm*exp(complex<T>(0., k*r))*hyp(complex<T>(0.,-k*r*(1-costheta)));
    }
    
    inline complex<T> psi(const T& r, const T& costheta){
        return (psi1(r, costheta)+psi1(r, -costheta))/sqrt(2);
    }
    
    inline T apsi(const T& r, const T& costheta){
        return abs( psi(r, costheta) );
    }

    inline T apsi0(const T& r){
        return 1+cos(2*k*r);
    }
    
    inline T S(const T& r){
        return (*levy)(r/R)/R/R/R;
    }
    
    inline T rndnorm(const T& xi){
        return sigma*normal(generator)+xi;
    }
    
public:
    Coulomb2(Levy<T> * _levy, T _pion_mass=139.5701835, T _eps=1e-8) : levy(_levy), pion_mass(_pion_mass), eps(_eps), sigma(1.), R(1.),
        uniform(0.,1.),uniform_theta(0., M_PI), normal(0., 1.), uniform2(0,100){
        generator.seed(rd());
    };
    
    T integrateTheta(const T& r){
        auto res = m.integrate([&, this](const T& x, T& result)->void{
            result = this->apsi(r, x);
        }, -1, 1, eps);
        return get<0>(res);
    }
    
    T integrateTheta_MCMC(const T& r, const unsigned long long& path){
        T s = 0.;
        T theta;
        for(unsigned long long int i = 0; i<path;i++){
            theta = uniform_theta(generator);
            s += apsi(r,cos(theta))*sin(theta);
        }
        return s*M_PI/T(path);
    }
    
    void set_k(const T& _k){
        k = _k;
        eta = 0.25*pion_mass/137.0359917 / k;
        hyp.set_eps(eps);
        hyp.set_ab(complex<T>(1.,eta), complex<T>(1.,0.));
        hyp.setCache(-100, 100, 1000000);
        norm = gamma(complex<T>(1.,eta))*exp(-M_PI*eta*0.5);
    }
    
    void set_sigma(const T& _sigma){
        sigma = _sigma;
    }
    
    void set_R(const T& _R){
        R = _R;
    }
    
    T integrateLevy(const T& maxr, const T& error){
        auto res = m.integrate([l=levy](const double& r, double& y){
            y = (*l)(r)*r*r;
        }, 0., maxr, error);
        
        return get<0>(res);
    }
    
    void Levy(int n1, double * result, const unsigned long long int& path){
        T ri(0.), rg(0.);
        T p_ri = S(ri), p_rg = 0.;
        for(unsigned long long int i=0;i<path;i++){
            rg = abs(rndnorm(ri));
            p_rg = S(rg);
            if (uniform(generator)<(p_rg/p_ri)){
                ri = rg;
                p_ri = p_rg;
            }
            result[i] = ri;
        }
    }
    
    T integrate(const unsigned long long int& path){
        T s = 0.;
        T s0 = 0.;
        T ri(0.), rg(0.), theta(0.);
        T p_ri = S(ri), p_rg = 0.;
        for(unsigned long long int i=0;i<path;i++){
                rg    = abs(rndnorm(ri));
                p_rg  = S(rg);
                theta = uniform_theta(generator);
                if (uniform(generator)<(p_rg/p_ri)){
                    ri   = rg;
                    p_ri = p_rg;
                }
                s  += SQR(apsi(ri, cos(theta)))*sin(theta)*ri*ri;
                s0 += SQR(apsi0(ri*cos(theta)))*sin(theta)*ri*ri;
        }
        //s  *=  M_PI*2*M_PI / T(path);
        //s0 *=  M_PI*2*M_PI / T(path);
        return s0/s;
    }
    
    T integrate(const T& _k, const unsigned long long int& path){
        set_k(_k);
        return integrate(path);
    }

};

#endif