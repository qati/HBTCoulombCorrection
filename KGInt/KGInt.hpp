#include <string>

std::string setGPU(int device);
double integrate(const int& N, double * result, double * rs, double * alphas, double *errors, double * error_rngs);
double integrate_GPU(const int& N, double * result, double * rs, double * alphas, double *errors, double * error_rngs);