#include <iostream>
#include <cmath>
#include <vector>
#include <functional>
#include <string>
#include <algorithm>
#include <limits>

using namespace std;

class KGInt{
  vector<double> weights, nodes;
  vector<int> gnodes, knodes;
public:
  int status;
  KGInt(){
    status = 0;
    nodes = {
      -0.991455371120813, 0.991455371120813,
      -0.949107912342759, 0.949107912342759,
      -0.864864423359769, 0.864864423359769,
      -0.741531185599394, 0.741531185599394,
      -0.586087235467691, 0.586087235467691,
      -0.405845151377397, 0.405845151377397,
      -0.207784955007898, 0.207784955007898,
      0.0
    };
    weights = {
      0.022935322010529, 0.022935322010529,
      0.063092092629979, 0.063092092629979,
      0.104790010322250, 0.104790010322250,
      0.140653259715525, 0.140653259715525,
      0.169004726639267, 0.169004726639267,
      0.190350578064785, 0.190350578064785,
      0.204432940075298, 0.204432940075298,
      0.209482141084728
    };
    gnodes = {2, 3, 6, 7, 10, 11, 14};
    knodes = {0,  1,  4,  5,  8,  9, 12, 13};
  }
  double integrate(const function<double(const double&)>&, const double&, const double&, const double & error, int);
};

double KGInt::integrate(const function<double(const double&)>& f, const double& a, const double& b, const double& error=1e-5, int n=50)
{
  double A = 0.5*(b-a);
  double B = 0.5*(b+a);
  double g7 = 0., k = 0.;
  double absIntegral =  0.0;
  vector<double> fs;
  double t;
  for(const int& i : gnodes){
    t = f(A*nodes[i]+B);
    fs.push_back(t);
    g7 += weights[i]*t;
    absIntegral += weights[i]*abs(t);
  }
  k = g7;
  for(const int& i : knodes){
    t = f(A*nodes[i]+B);
    fs.push_back(t);
    k += weights[i]*t;
    absIntegral += weights[i]*abs(t);
  }
  
  double mean = k/2.;
  double absDiffIntegral = 0.0;
  for(int i=0;i<weights.size();i++){
    absDiffIntegral += weights[i]*abs(fs[i]-mean);
  }
  
  absIntegral *= A;
  absDiffIntegral *= A;
  
  double estimatedError = abs(k-g7)*A;
  
  if (absDiffIntegral>0.  && estimatedError!=0.){
    estimatedError = absDiffIntegral*min(1.,pow(estimatedError*200./absDiffIntegral,1.5));
  }
  
  if (absIntegral>numeric_limits<double>::min()/(50.*numeric_limits<double>::epsilon())){
    estimatedError = max(estimatedError, absIntegral*numeric_limits<double>::min()*50.);
  }
  
  if (estimatedError>error && n>0){
    return integrate(f, a, 0.5*(a+b), error, n-1)+integrate(f, 0.5*(a+b), b, error, n-1);
  }
  
  if (n<1){
    status=-1;
  }

  return (k)*A;
}


int main(int argc, char** argv)
{
    if (argc<4){
      cerr << "Usage: integrate.exe alpha R error";
      return -1;
    }
    double alpha = stod(argv[1]);
    double R = stod(argv[2]);
    double error = stod(argv[3]);
    cerr << "alpha="<<alpha <<" ;R="<<R<<"; error="<<error<<endl;
    KGInt m;
    cout << m.integrate(
                [&alpha, &R](const double& x)->double{
                  return x*sin(x*R)*exp(-pow(x, alpha));
                }, 0.0, pow(-log(error),1/alpha), error)<<endl;
                
  cout << endl << m.status<<endl;
  return 0;
}