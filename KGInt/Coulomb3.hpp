#ifndef __COULONB3_HPP__
#define __COULONB3_HPP__

#include <random>
#include <complex>

using namespace std;

#include "functions.hpp"

#ifndef __COULONB2_HPP__
const double HBARC = 197.3269788;
#endif

template<class T> class Coulomb3 {
private:
    Levy<T>* levy;
    Hypergeometric<T> hyp[3];
    Gamma<T> gamma;
    KGInt<T> m;
    T eps;
    T pion_mass;
    T eta[3], kabs[3];
    complex<T> norm[3];
    T sigma;
    T R;
    T r[3][3];
    T k[3][3];
    T rabs[3];
    
    random_device rd;
    mt19937 generator;
    uniform_real_distribution<T> uniform, uniform_theta, uniform_phi;
    normal_distribution<T> normal;
    
    inline T dot(T v1[3], T v2[3]){
        return v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2];
    }
    
    inline void copy(T v1[3], T v2[3]){
        v1[0] = v2[0];
        v1[1] = v2[1];
        v1[2] = v2[2];
    }
    
    inline void dcopy(T v1[3][3], T v2[3][3]){
        v1[0][0] = v2[0][0];
        v1[0][1] = v2[0][1];
        v1[0][2] = v2[0][2];
        v1[1][0] = v2[1][0];
        v1[1][1] = v2[1][1];
        v1[1][2] = v2[1][2];
        v1[2][0] = v2[2][0];
        v1[2][1] = v2[2][1];
        v1[2][2] = v2[2][2];
    }
    
    inline void set(T v[3], const T& x1, const T& x2, const T& x3){
        v[0] = x1;
        v[1] = x2;
        v[2] = x3;
    }
    
    inline void copy(T v[3], const T& a1, T v1[3], const T& a2, T v2[3]){
        v[0] = a1*v1[0]+a2*v2[0];
        v[1] = a1*v1[1]+a2*v2[1];
        v[2] = a1*v1[2]+a2*v2[2];
    }
    
    inline complex<T> psi0(int k_idx, int r_idx){
        return exp(2.*complex<T>(0.,-dot(k[k_idx], r[r_idx])/HBARC)/3.);
    }
    
    inline complex<T> psi1(int k_idx, int r_idx){
        return norm[k_idx]*exp(2.*complex<T>(0., kabs[k_idx]*rabs[r_idx]/HBARC/3.))
                *hyp[k_idx](complex<T>(0.,(
                            -kabs[k_idx]*rabs[r_idx]
                            +dot(k[k_idx], r[r_idx])
                    )/HBARC));
    }
    
    inline T apsi0(){
        return abs( (psi0(0,0)*psi0(1,1)*psi0(2,2)+
                     psi0(0,1)*psi0(1,2)*psi0(2,0)+
                     psi0(0,2)*psi0(1,0)*psi0(2,1)+
                     psi0(0,1)*psi0(1,0)*psi0(2,2)+
                     psi0(0,0)*psi0(1,2)*psi0(2,1)+
                     psi0(0,2)*psi0(1,1)*psi0(2,0)
                    )/sqrt(6) );
    }
    
    inline T apsi1(){
        return abs( (psi1(0,0)*psi1(1,1)*psi1(2,2)+
                     psi1(0,1)*psi1(1,2)*psi1(2,0)+
                     psi1(0,2)*psi1(1,0)*psi1(2,1)+
                     psi1(0,1)*psi1(1,0)*psi1(2,2)+
                     psi1(0,0)*psi1(1,2)*psi1(2,1)+
                     psi1(0,2)*psi1(1,1)*psi1(2,0)
                    )/sqrt(6) );
    }
    
    /**
     * constant factor ( sqrt(27)*(2pi)^3 * R^6 )^{-1}
     */
    inline T S3(T _r[3], const T& phi){
        return exp(-(SQR(_r[0])+SQR(_r[1])+_r[0]*_r[1]*cos(phi))/(3*R*R));
    }
    
    inline T rndnorm(const T& xi){
        return sigma*normal(generator)+xi;
    }
        
public:
    Coulomb3(Levy<T> * _levy, T _pion_mass=139.5701835, T _eps=1e-8) : levy(_levy), pion_mass(_pion_mass), eps(_eps), sigma(1.), R(1.),
        uniform(0.,1.),uniform_theta(0., M_PI), normal(0., 1.), uniform_phi(0,2*M_PI){
        generator.seed(rd());
        status = SUCCESS;
    }
    
    void set_k(const T& k1, const T& k2, const T& k3){
        if ( (k1+k2)<k3 || (k1+k3)<k2 || (k2+k3)<k1){
            status = KTRIANGLE_ERROR;
            return;
        }
        kabs[0] = k1;
        kabs[1] = k2;
        kabs[2] = k3;        
        for(int i=0;i<3;i++){
            eta[i] = 0.25*pion_mass/137.0359917 / kabs[i];
            hyp[i].set_eps(eps);
            hyp[i].set_ab(complex<T>(1.,eta[i]), complex<T>(1.,0.));
            hyp[i].setCache(-100, 100, 1000000);
            norm[i] = gamma(complex<T>(1.,eta[i]))*exp(-M_PI*eta[i]*0.5);
        }
        T theta = acos( (SQR(k1)+SQR(k2)-SQR(k3))/(2*k1*k2) );
        set(k[0], k1, 0, 0);
        set(k[1], k2*sin(theta)/sqrt(2), k2*sin(theta)/sqrt(2), k2*cos(theta));
        copy(k[2], -1., k[0], -1., k[1]);
        status = SUCCESS;
    }
    
    void set_sigma(const T& _sigma){
        sigma = _sigma;
    }
    
    void set_R(const T& _R){
        R = _R/pow(2, 1./levy->get_alpha());
    }
    
    T integrate(const unsigned long long int& path){
        using param_type = typename decltype(uniform_theta)::param_type;

        T s = 0.;
        T s0 = 0.;
        
        T rabs_prop[3] = {0,0,0}, r_prop[3][3]={{0,0,0},{0,0,0},{0,0,0}};
        T theta1i(0), theta2i(0), phi1i(0), phi2i(0), phii(0), r1i(0), r2i(0);
        T theta1g(0), theta2g(0), phi1g(0), phi2g(0), phig(0), r1g(0), r2g(0);
        T dV;
        
        T p_ri = S3(rabs_prop, phig), p_rg;
        copy(rabs, rabs_prop);
        dcopy(r, r_prop);
        
        for(unsigned long long int i=0;i<path;i++){
                
                rabs_prop[0] = rndnorm(rabs[0]);
                rabs_prop[1] = rndnorm(rabs[1]);
                phig         = uniform_theta(generator);
                rabs_prop[2] = sqrt(SQR(rabs_prop[0])+SQR(rabs_prop[1])-2*rabs_prop[0]*rabs_prop[1]*cos(phig));
                
                phi2g   = uniform_phi(generator);
                theta1g = uniform_theta(generator);
                theta2g = uniform_theta(generator, param_type(abs(phig-theta1g),M_PI-abs(phig-theta1g-M_PI)));
                T tmp = ( cos(phig)-cos(theta1g)*cos(theta2g) )/( sin(theta1g)*sin(theta2g)  );
                // TODO: theta1g or theta2g 0 or pi 
                if (abs(tmp)>1){
                    tmp = 0.;
                }
                phi1g   = phi2g + acos(tmp);
                
                set(r_prop[0],  rabs_prop[0]*sin(theta1g)*cos(phi1g),
                                rabs_prop[0]*sin(theta1g)*sin(phi1g),
                                rabs_prop[0]*cos(theta1g));                 
                set(r_prop[1],  rabs_prop[1]*sin(theta2g)*cos(phi2g),
                                rabs_prop[1]*sin(theta2g)*sin(phi2g),
                                rabs_prop[1]*cos(theta2g));
                copy(r_prop[2], -1., r_prop[0], -1., r_prop[1]);
                
                //TODO problem
                /*if (rabs_prop[2]!=sqrt(SQR(r_prop[2][0])+SQR(r_prop[2][1])+SQR(r_prop[2][2]))){
                    status = RTRIANGLE_ERROR;
                    return sqrt(SQR(r_prop[2][0])+SQR(r_prop[2][1])+SQR(r_prop[2][2]));
                }*/
                
                p_rg = S3(rabs_prop, phig);
                if (uniform(generator)<(p_rg/p_ri)){
                    copy(rabs, rabs_prop);
                    dcopy(r, r_prop);
                    theta1i = theta1g;
                    theta2i = theta2g;
                    p_ri = p_rg;
                }
                
                dV = rabs[0]*rabs[0]*sin(theta1i)*rabs[1]*rabs[1]*sin(theta2i);
                s  += SQR(apsi1())*dV;
                s0 += SQR(apsi0())*dV;
        }
        //s  *=  M_PI*2*M_PI / T(path);
        //s0 *=  M_PI*2*M_PI / T(path);
        return s0/s;
    }
    
    T integrate(const T& k1, const T& k2, const T& k3, const unsigned long long int& path){
        set_k(k1,k2,k3);
        if (status!=SUCCESS){
            return -1;
        }
        return integrate(path);
    }
    
    
    int status;
    const int SUCCESS = 0;
    const int KTRIANGLE_ERROR = 1;
    const int RTRIANGLE_ERROR = 2;
};

#endif