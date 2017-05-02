#include <iostream>
#include <string>
#include <iomanip>
#include <chrono> 
#include <complex>

using namespace std;

#include "functions.hpp"
#include "KGInt.hpp"



int main(int argc, char **argv)
{
  if (argc<4){
    std::cerr << "Pass r and alpha error!" << std::endl;
    return -1;
  }
  
  double  result[2], rs[1], alphas[1], errors[1], error_rngs[1];
  
  rs[0]     = std::stod(argv[1]);
  alphas[0] = std::stod(argv[2]);
  errors[0] = std::stod(argv[3]);
  error_rngs[0] = 1e-9;
  
  double time = integrateLevyGPU(1,1, result, rs, alphas, errors, error_rngs); 
  
  std::cerr << "-->CUDA" <<std::endl;
  
  std::cerr << "Elapsed time: " << std::setprecision(2) << time << " ms" << std::endl;
  std::cerr << "Result: " << std::setprecision(15) << result[0] << ", error="<<std::setprecision(15) << result[1] << std::endl;
  std::cerr << "-------------" << std::endl;
  
  
  std::cerr << "-->CPU parallel" << std::endl;
  time = integrateLevyCPU(1, result, rs, alphas, errors, error_rngs); 
  std::cerr << "Elapsed time: " << std::setprecision(2) << time << " ms" << std::endl;
  std::cerr << "Result: " << std::setprecision(15) << result[0] << ", error="<<std::setprecision(15) << result[1] << std::endl<<std::endl;
  
  
  std::cerr << "-->FFF calculation" << std::endl;
  double eta=0.2, kr=10;
  Hypergeometric<double> h;
  h.set_eps(1e-8);
  h.set_ab(complex<double>(1.,eta), complex<double>(1.,0.));
  double s1, s2;
  complex<double> tmp;
  int i=0;
  auto start = std::chrono::high_resolution_clock::now();
  double dy = 2./100000.;
  for(;i<100000;i++){
      tmp =  h(complex<double>(0.,kr*(-1+i*dy)));
      s1 += tmp.real();
      s2 += tmp.imag();
  }
  auto finish = std::chrono::high_resolution_clock::now();
  double ti = std::chrono::duration_cast<std::chrono::milliseconds>(finish-start).count();
  std::cerr << "Elapsed time: " << std::setprecision(2) << ti << " ms" << std::endl;
  //std::cerr << "Result: real=" << std::setprecision(15) << result[0] << ", imag="<<std::setprecision(15) << result[1] << std::endl<<std::endl;
  
  Gamma<double> gamma;
  std::cerr << std::endl<<"------------------" <<std::endl<<"gamma(1+0j)=" << gamma(complex<double>(1.,0))<<std::endl;
  complex<double> a(0., -0.98), b(1.,0.), z(0., 49);
  complex<double> F1 = pow(-z,-a)/gamma(b-a),
             F2 = exp(z)*pow(z,a-b)/gamma(a);
  cerr << "F1="<<F1 <<", F2="<<F2 << endl;
  F1 = gamma(b)*pow(-z,-a)/gamma(b-a);
  F2 = gamma(b)*exp(z)*pow(z,a-b)/gamma(a);
  cerr << "with gamma(b); F1="<<F1<<", F2="<<F2<<endl;
  return 0;
}