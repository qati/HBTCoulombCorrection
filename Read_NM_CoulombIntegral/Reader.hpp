class CorrFunc_evaluator;

class Reader{
    CorrFunc_evaluator *cfe;
public:
    Reader();
    ~Reader();
    double get(double l, double k, double alpha, double R);
};