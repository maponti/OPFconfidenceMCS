#ifndef H_CLASSIFIER_LIBCOMB
#define H_CLASSIFIER_LIBCOMB

#include <cstdio>
#include <iostream>
#include <fstream> // file stream
#include <iomanip>
#include <string>
#include <cstdlib>
#include <cmath>

#include "Predictor.hpp"
#include "Data.hpp"

class Classifier: public Predictor{
    private:
        bool trained;
    protected:
        void setTrained(bool trained){
            this->trained = trained;
        }

        bool isTrained(){
            return trained;
        }



	public:
        virtual ~Classifier() {}
		virtual void train(Data *trainData) = 0;
		virtual void predict(Data *data) = 0;
        virtual Classifier* clone() const = 0;

        virtual void predict(Data *data, double *double_fault, double *q_static, double *ir_agree,
                                double *desagree, double *correlation){}
        void diversidade(int **matrix_label, int numclas, int namostras, Data *d,
                                double *double_fault, double *q_static, double *ir_agree,
                                double *desagree, double *correlation);

        virtual bool hasScore() const 
        {
            return false;
        }

        virtual float getScore(int index, int label) const 
        {
            return -1.0f;
        }
};

#endif
