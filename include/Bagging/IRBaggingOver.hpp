#ifndef H_IRBAGGINGOVER_LIBCOMB
#define H_IRBAGGINGOVER_LIBCOMB

#include <cstdio>
#include <iostream>
#include <iomanip>
#include <cstdlib>

#include "Bagging.hpp"
#include "Data.hpp"

class IRBaggingOver: public Bagging{
    private:
        double bootstrapIR;

	protected:
		Data *positives;
		Data *negatives;

        void divideTrainData();
        Data generateBootstrap();

	public:
        IRBaggingOver(Classifier **classifiers, int nClassifiers);
        IRBaggingOver(Classifier *classifier, int nClassifiers);

        void train(Data *trainData);
};

#endif
