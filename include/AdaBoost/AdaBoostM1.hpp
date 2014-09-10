#ifndef H_ADABOOST_M1_LIBCOMB
#define H_ADABOOST_M1_LIBCOMB

#include <cstdio>
#include <iostream>
#include <fstream> // file stream
#include <iomanip>
#include <string>
#include <cstdlib>
#include <cmath>
#include <random>
#include <array>
#include <iostream>
#include <vector>
#include <iterator>

#include "Predictor.hpp"
#include "Classifier.hpp"
#include "Data.hpp"

class AdaBoostM1: public Classifier{
	private:
        double *beta; /// alocar na funcao train
        /// criar destrutor para desalocar a memória

		Classifier **baseClassifiers;
		int nBaseClassifiers;

		Classifier **finalClassifiers;
		int nFinalClassifiers;

		Data *trainData;

		Data generateSubSet(double *w, int subSetSize);

	public:
		AdaBoostM1(Data *trainData, Classifier **classifiers, int nClassifiers, int nIterations);
		AdaBoostM1(Classifier **baseClassifiers, int nBaseClassifiers, int nIterations);

		//~AdaBoostM1();

        AdaBoostM1* clone() const;

		void train(Data *trainData, Classifier **classifiers, int nClassifiers, int nIterations);
		void train(Data *trainData);
		void train();

		void predict(Data *data);

		void predict(Data *data, double *double_fault, double *q_static, double *ir_agree,
                                double *desagree, double *correlation);
};

#endif
