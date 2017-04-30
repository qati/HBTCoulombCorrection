#ifndef __KGINT_HPP__
#define __KGINT_HPP__

#include <string>
#include <complex>

/**
 * Selecting GPU!
 */
std::string setGPU(int device);

/**
 * Calculate Levy integral on CPU
 */
double integrateLevyCPU(const int& N, double * result, double * rs, double * alphas, double *errors, double * error_rngs);

/**
 * Calculate Levy integral on GPU
 */
double integrateLevyGPU(const int& blockNum, const int& threadNum, double * result, double * rs, double * alphas, double *errors, double * error_rngs);


/**
 * Calculate A^(1)+A^(2) on CPU
 */
 double integrateACPU(const int& N, double * result, double * etas, double * krs, double *errors, double * int_errors);
 
 /**
  * Calculate A^(1)+A^(2) on GPU
  */
 double integrateAGPU(const int& blockNum, const int& threadNum, double * result, double * etas, double * krs, double *errors, double * int_errors);

/**
 * Hyp1f1 function
 */
 void hyp1f1_FFF(int n1, double * result, double kr, double eta, double eps);
 
 void hyp1f1(int n1, double * result, double kr, double eta, double eps);

 void hyp1f1(int n1, double * result, const std::complex<double>& a, const std::complex<double>& b, int n2, double* rs, int n3, double* krs, const double& eps=1e-8);

#endif