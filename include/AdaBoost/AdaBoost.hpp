#ifndef H_ADABOOST_LIBCOMB
#define H_ADABOOST_LIBCOMB

#include <cstdio>
#include <iostream>
#include <fstream> // file stream
#include <iomanip>
#include <string>
#include <cstdlib>
#include <cmath>

#include "Predictor.hpp"
#include "Classifier.hpp"
#include "Data.hpp"

class AdaBoost: public Predictor{
	private:
        double *alpha; /// alocar na funcao train
        /// criar destrutor para desalocar a memória

		Classifier **classifiers;
		int nClassifiers;
		Data *trainData;

	public:
		AdaBoost(Data *trainData, Classifier **classifiers, int nClassifiers, int nIterations);

		void train(Data *trainData, Classifier **classifiers, int nClassifiers, int nIterations);
		void predict(Data *data);
};

#endif
