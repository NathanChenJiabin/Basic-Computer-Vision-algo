//
// Created by jiabin on 2018/5/27.
//

#include "Classifieur.h"

Classifieur::Classifieur(int idrank, int idpos): w1(1.0), w2(0.0), coeff(0.0) {
    this->idproc = idrank;
    this->idposition = idpos;
}

Classifieur::~Classifieur() {

}

double Classifieur::getW1() const {
    return w1;
}

void Classifieur::setW1(double w1) {
    Classifieur::w1 = w1;
}

double Classifieur::getW2() const {
    return w2;
}

void Classifieur::setW2(double w2) {
    Classifieur::w2 = w2;
}

const int Classifieur::getIdproc() const {
    return idproc;
}

const int Classifieur::getIdposition() const {
    return idposition;
}

int Classifieur::result_of_classification(int Xi){
    if(this->w1 * Xi + this->w2 >= 0){
        return 1;
    }else{
        return -1;
    }
}

void Classifieur::update(double epsilon, int Xi, int label){
    this->setW1(this->getW1() - epsilon*(this->result_of_classification(Xi)-label)*Xi);
    this->setW2(this->getW2() - epsilon*(this->result_of_classification(Xi)-label));
}

double Classifieur::getCoeff() const {
    return coeff;
}

void Classifieur::setCoeff(double coeff) {
    Classifieur::coeff = coeff;
}
