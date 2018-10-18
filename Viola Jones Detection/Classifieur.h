//
// Created by jiabin on 2018/5/27.
//

#ifndef VIOLADETECTION_CLASSIFIEUR_H
#define VIOLADETECTION_CLASSIFIEUR_H


#include <cmath>

class Classifieur {

public:
    Classifieur(int idrank, int idpos);

    virtual ~Classifieur();

    double getW1() const;

    void setW1(double w1);

    double getW2() const;

    void setW2(double w2);

    const int getIdproc() const;

    const int getIdposition() const;

    int result_of_classification(int Xi);

    void update(double epsilon, int Xi, int label);

    double getCoeff() const;

    void setCoeff(double coeff);

private:

    double w1;
    double w2;
    int idproc;
    int idposition;
    double coeff;

};


#endif //VIOLADETECTION_CLASSIFIEUR_H
