#ifndef CorrFunc_evaluator_H
#define CorrFunc_evaluator_H

#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <sstream>
#include <cmath>
using namespace std;

class CorrFunc_evaluator
{
 public:
  CorrFunc_evaluator(const char* filename);
  virtual ~CorrFunc_evaluator();
  float get_value_CC(const double alpha, const double k, const double Rcc) const;
  float read_table(const double alpha, const double k, const double Rcc) const;
  float interpolate_in_table(const double alpha, const double k, const double Rcc) const;
  float Levy_func(const double alpha, const double k, const double Rcc) const;
  float getValue(const double alpha, const double k, const double Rcc) const; // the user should use this!!!
 
 private:
  float*** CC_array;
  double alpha_min;
  double alpha_max;
  double k_min;
  double k_max;
  double Rcc_min;
  double Rcc_max;
  int Nalpha;
  int Nk;
  int NRcc;
  double d_alpha;
  double d_k;
  double d_Rcc;

  void Init(const char* filename);
};

#endif // CorrFunc_evaluator_H
