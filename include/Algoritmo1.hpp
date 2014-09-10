#ifndef H_ALGORITMO1_LIBCOMB
#define H_ALGORITMO1_LIBCOMB

#include <cstdio>
#include <iostream>
#include <iomanip>
#include <cstdlib>

#include "Classifier.hpp"
#include "Data.hpp"
#include "ConfusionMatrix.hpp"

class Algoritmo1: public Classifier{
	protected:
        Classifier **classifiers;
		int nClassifiers;
		Data *trainData;

		Data *positives;
		Data *negatives;

		double *w;

        void divideTainData();
		double calcW(Classifier *c);
		Data generateBootstrap(double ir);

	public:
        Algoritmo1(Classifier **classifiers, int nClassifiers);
        Algoritmo1(Classifier *classifier, int nClassifiers);

        Algoritmo1* clone() const;

		void train(Data *trainData);

		void predict(Data *data);

};

#endif
