#include <iostream>
#include <cmath>
#include <vector>
#include <functional>
#include <string>

using namespace std;

class KGInt{
  vector<double> weights, nodes;
  vector<int> gnodes, knodes;
public:
  KGInt(){
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

double KGInt::integrate(const function<double(const double&)>& f, const double& a, const double& b, const double& error=1e-5, int n=1000)
{
  double A = 0.5*(b-a);
  double B = 0.5*(b+a);
  double g7 = 0., k8 = 0.;
  for(const int& i : gnodes){
    g7 += weights[i]*f(A*nodes[i]+B);
  }
  for(const int& i : knodes){
    k8 += weights[i]*f(A*nodes[i]+B);
  }
  
  if (pow(200*abs(k8-g7), 1.5)>error && n>0){
    //cout <<"error=" << pow(200*abs(k8-g7), 1.5) <<"; a1="<<a <<"; b1="<<0.5*(a+b)<<"; a2="<<0.5*(a+b)<<"; b2="<<b << endl;
    return integrate(f, a, 0.5*(a+b), error, n-1)+integrate(f, 0.5*(a+b), b, error, n-1);
  }
  
  return (g7+k8)*A;
}


int main(int argc, char** argv)
{
    if (argc<4){
      cerr << "Usage: integrate.exe a b error";
      return -1;
    }
    cerr << "a=" << argv[1] << "; b="<<argv[2] <<"; error="<<stod(argv[3])<<endl;
    KGInt m;
    cout << m.integrate(
                [](const double& x)->double{
                  return x*sin(x*20)*exp(-pow(x, 0.7)/2);
                }, stod(argv[1]), stod(argv[2]), stod(argv[3]));
    return 0;
}