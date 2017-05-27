#include <iostream>
#include "CorrFunc_evaluator.h"
#include "Reader.hpp"

namespace CONSTS{
  const double HBARC =  197.3269788; //MeV*fm
}
    
using namespace std;

Reader::Reader()
{
    cfe = new CorrFunc_evaluator("/phenix/plhf/nmarci/others/Levy_coulcorr_database/new/CC_integral_hbarOK.dat");
}

Reader::~Reader()
{
    delete cfe;
}
    
double Reader::get(double l, double k, double alpha, double R)
{
    return (1+l*(cfe->getValue(alpha, k, R*pow(2, 1./alpha))-1) ) / (1+l*exp(-fabs(pow(2.*k*R/CONSTS::HBARC,alpha))));
}
