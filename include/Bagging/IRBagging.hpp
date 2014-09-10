#ifndef H_IRBAGGING_LIBCOMB
#define H_IRBAGGING_LIBCOMB

#include <cstdio>
#include <iostream>
#include <iomanip>
#include <cstdlib>

#include "Bagging.hpp"
#include "Data.hpp"

class IRBagging: public Bagging{
    private:
        double bootstrapIR;

	protected:
		Data *positives;
		Data *negatives;

        void divideTrainData();
        Data generateBootstrap();

	public:
        IRBagging(Classifier **classifiers, int nClassifiers);
        IRBagging(Classifier *classifier, int nClassifiers);

        void train(Data *trainData);
};

#endif
