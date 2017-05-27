#include "CorrFunc_evaluator.h"
#define HBARC 197.327

CorrFunc_evaluator::CorrFunc_evaluator(const char* filename)
{
  CC_array = NULL;
  Init(filename);
}

CorrFunc_evaluator::~CorrFunc_evaluator()
{
  if(CC_array) delete [] CC_array;
}

float CorrFunc_evaluator::Levy_func(const double alpha, const double k, const double Rcc) const
{
  return 1. + exp(- 0.5 * pow(2. * k * Rcc / HBARC, alpha));
}

float CorrFunc_evaluator::getValue(const double alpha, const double k, const double Rcc) const
{
  if(k >= k_max - .001) return Levy_func(alpha, k, Rcc);
  return interpolate_in_table(alpha, k, Rcc);
}

float CorrFunc_evaluator::interpolate_in_table(const double alpha, const double k, const double Rcc) const
{
  if(!CC_array)
    return 0;
  int ik = floor((k - k_min) / d_k);
  double klo = k_min + ik * d_k;
  int ialpha = floor((alpha - alpha_min) / d_alpha);
  double alphalo = alpha_min + ialpha * d_alpha;
  int iRcc = floor((Rcc - Rcc_min) / d_Rcc);
  double Rcclo = Rcc_min + iRcc * d_Rcc;
  if(ik < 0 || ik > Nk || ialpha < 0 || ialpha > Nalpha || iRcc < 0 || iRcc > NRcc)
    return 0;
  if(ik < Nk && ialpha < Nalpha && iRcc < NRcc)
  { 
    double klo_lint_iRcc = CC_array[ialpha][ik][iRcc] * (klo - k + d_k) / d_k + CC_array[ialpha][ik + 1][iRcc] * (k - klo) / d_k;
    double khi_lint_iRcc = CC_array[ialpha + 1][ik][iRcc] * (klo - k + d_k) / d_k + CC_array[ialpha + 1][ik + 1][iRcc] * (k - klo) /d_k;
    double klo_lint_iRcc_plus = CC_array[ialpha][ik][iRcc + 1] * (klo - k + d_k) / d_k + CC_array[ialpha][ik + 1][iRcc + 1] * (k - klo) / d_k;
    double khi_lint_iRcc_plus = CC_array[ialpha + 1][ik][iRcc + 1] * (klo - k + d_k) / d_k + CC_array[ialpha + 1][ik + 1][iRcc + 1] * (k - klo) /d_k;

    double alphalo_lint = klo_lint_iRcc * (alphalo - alpha + d_alpha) / d_alpha + khi_lint_iRcc * (alpha - alphalo) / d_alpha;
    double alphahi_lint = klo_lint_iRcc_plus * (alphalo - alpha + d_alpha) / d_alpha + khi_lint_iRcc_plus * (alpha - alphalo) / d_alpha;
    return alphalo_lint * (Rcclo - Rcc + d_Rcc) / d_Rcc + alphahi_lint * (Rcc - Rcclo) / d_Rcc;
  }
  if(ik == Nk && ialpha < Nalpha && iRcc < NRcc)
  {
    double Rcclo_lint = CC_array[ialpha][ik][iRcc] * (Rcclo - Rcc + d_Rcc) / d_Rcc + CC_array[ialpha][ik][iRcc + 1] * (Rcc - Rcclo) / d_Rcc;
    double Rcchi_lint = CC_array[ialpha + 1][ik][iRcc] * (Rcclo - Rcc + d_Rcc) / d_Rcc + CC_array[ialpha + 1][ik][iRcc + 1] * (Rcc - Rcclo) /d_Rcc;
    return Rcclo_lint * (alphalo - alpha + d_alpha) / d_alpha + Rcchi_lint * (alpha - alphalo) / d_alpha;
  }
  if(ialpha == Nalpha && ik < Nk && iRcc < NRcc)
  {
    double klo_lint = CC_array[ialpha][ik][iRcc] * (klo - k + d_k) / d_k + CC_array[ialpha][ik + 1][iRcc] * (k - klo) / d_k;
    double khi_lint = CC_array[ialpha][ik][iRcc + 1] * (klo - k + d_k) / d_k + CC_array[ialpha][ik + 1][iRcc + 1] * (k - klo) /d_k;
    return klo_lint * (Rcclo - Rcc + d_Rcc) / d_Rcc + khi_lint * (Rcc - Rcclo) / d_Rcc;
  }
  if(iRcc == NRcc && ik < Nk && ialpha < Nalpha)
  {
    double klo_lint = CC_array[ialpha][ik][iRcc] * (klo - k + d_k) / d_k + CC_array[ialpha][ik + 1][iRcc] * (k - klo) / d_k;
    double khi_lint = CC_array[ialpha + 1][ik][iRcc] * (klo - k + d_k) / d_k + CC_array[ialpha + 1][ik + 1][iRcc] * (k - klo) /d_k;
    return klo_lint * (alphalo - alpha + d_alpha) / d_alpha + khi_lint * (alpha - alphalo) / d_alpha;
  }
  else if(iRcc == NRcc && ik == Nk && ialpha < Nalpha)
    return CC_array[ialpha][ik][iRcc] * (alphalo - alpha + d_alpha) / d_alpha + CC_array[ialpha + 1][ik][iRcc] * (alpha - alphalo) / d_alpha;
  else if(iRcc == NRcc && ik < Nk && ialpha == Nalpha)
    return CC_array[ialpha][ik][iRcc] * (klo - k + d_k) / d_k + CC_array[ialpha][ik + 1][iRcc] * (k - klo) / d_k;
  else if(iRcc < NRcc && ik == Nk && ialpha == Nalpha)
    return CC_array[ialpha][ik][iRcc] * (Rcclo - Rcc + d_Rcc) / d_Rcc + CC_array[ialpha][ik][iRcc + 1] * (Rcc - Rcclo) / d_Rcc;
  else if(ik == Nk && ialpha == Nalpha && iRcc == NRcc)
    return CC_array[ialpha][ik][iRcc];
  return 0;
}

float CorrFunc_evaluator::read_table(const double alpha, const double k, const double Rcc) const
{
  if(!CC_array)
    return 0;
  int ialpha = floor((alpha - alpha_min) / d_alpha); 
  int ik = floor((k - k_min) / d_k); 
  int iRcc = floor((Rcc - Rcc_min) / d_Rcc); 
  if(ialpha >= 0 && ialpha < (Nalpha + 1))
    if(ik >= 0 && ik < (Nk + 1))
      if(iRcc >= 0 && iRcc < (NRcc + 1))
      return CC_array[ialpha][ik][iRcc];
  return 0;
}

void CorrFunc_evaluator::Init(const char* filename)
{ 
  ifstream infile;
  infile.open(filename, ios::in | ios::binary);
  cout << "Initializing CorrFunc_evaluator table in memory using file: " << filename << endl;
  infile.read((char*)&alpha_min, sizeof(alpha_min));  cout << "  alpha_min = " << alpha_min << endl;
  infile.read((char*)&alpha_max, sizeof(alpha_max));  cout << "  alpha_max = " << alpha_max << endl;
  infile.read((char*)&Nalpha,    sizeof(Nalpha));     cout << "  Nalpha = " << Nalpha << endl;
  infile.read((char*)&k_min,     sizeof(k_min));      cout << "  k_min = " << k_min << endl;
  infile.read((char*)&k_max,     sizeof(k_max));      cout << "  k_max = " << k_max << endl;
  infile.read((char*)&Nk,        sizeof(Nk));         cout << "  Nk = " << Nk << endl;
  infile.read((char*)&Rcc_min,   sizeof(Rcc_min));    cout << "  Rcc_min = " << Rcc_min << endl;
  infile.read((char*)&Rcc_max,   sizeof(Rcc_max));    cout << "  Rcc_max = " << Rcc_max << endl;
  infile.read((char*)&NRcc,      sizeof(NRcc));       cout << "  NRcc = " << NRcc << endl;
  d_alpha = (alpha_max - alpha_min) * 1.0 / Nalpha;
  d_k = (k_max - k_min) * 1.0 / Nk;
  d_Rcc= (Rcc_max - Rcc_min) * 1.0 / NRcc;
  cout << "    d_alpha: " << d_alpha  << " d_k: " << d_k << " d_Rcc: " << d_Rcc << endl;

  float value = 0;
  CC_array = new float**[Nalpha + 1];
  for(int ialpha = 0; ialpha < (Nalpha + 1); ialpha++)
  {
    CC_array[ialpha] = new float*[Nk + 1];
    for(int ik = 0; ik < (Nk + 1); ik++)
      CC_array[ialpha][ik] = new float[NRcc + 1];
  }

  for(int ialpha = 0; ialpha < (Nalpha + 1); ialpha++)
    for(int ik = 0; ik < (Nk + 1); ik++)
      for(int iRcc = 0; iRcc < (NRcc + 1); iRcc++)
      {
        infile.read((char*)&value, sizeof(value));
        CC_array[ialpha][ik][iRcc] = value;
      }
  cout << "CorrFunc_table loaded." << endl;

  for(int ialpha = 0; ialpha < (Nalpha + 1); ialpha++)
    for(int iRcc = 0; iRcc < (NRcc + 1); iRcc++)
    {
      float diff_expected = Levy_func(alpha_min + d_alpha * ialpha, k_max, Rcc_min + d_Rcc * iRcc) - CC_array[ialpha][Nk][iRcc];
      for(int ik = 0; ik < (Nk + 1); ik++)
        CC_array[ialpha][ik][iRcc] += diff_expected;
    }
  cout << "CorrFunc_table patched." << endl;
  infile.close();
}
