#ifndef H_ALGORITMO2_LIBCOMB
#define H_ALGORITMO2_LIBCOMB

#include <cstdio>
#include <iostream>
#include <iomanip>
#include <cstdlib>

#include "Classifier.hpp"
#include "Data.hpp"
#include "ConfusionMatrix.hpp"

class Algoritmo2: public Classifier{
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
        Algoritmo2(Classifier **classifiers, int nClassifiers);
        Algoritmo2(Classifier *classifier, int nClassifiers);

        Algoritmo2* clone() const;

		void train(Data *trainData);

		void predict(Data *data);

};

#endif
