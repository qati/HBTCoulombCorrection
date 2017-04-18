#include <iostream>
#include <string>
#include <iomanip>
#include <chrono> 

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
  
  double time = integrate_GPU(1,1, result, rs, alphas, errors, error_rngs); 
  
  std::cerr << "-->CUDA" <<std::endl;
  
  std::cerr << "Elapsed time: " << std::setprecision(2) << time << " ms" << std::endl;
  std::cerr << "Result: " << std::setprecision(15) << result[0] << ", error="<<std::setprecision(15) << result[1] << std::endl;
  std::cerr << "-------------" << std::endl;
  
  
  std::cerr << "-->CPU parallel" << std::endl;
  time = integrate(1, result, rs, alphas, errors, error_rngs); 
  std::cerr << "Elapsed time: " << std::setprecision(2) << time << " ms" << std::endl;
  std::cerr << "Result: " << std::setprecision(15) << result[0] << ", error="<<std::setprecision(15) << result[1] << std::endl<<std::endl;
  return 0;
}