#ifndef H_EB_BAGGING_LIBCOMB
#define H_EB_BAGGING_LIBCOMB

#include <cstdio>
#include <iostream>
#include <fstream> // file stream
#include <iomanip>
#include <string>
#include <cstdlib>
#include <random>
#include <ctime>

#include "Classifier.hpp"
#include "Data.hpp"
#include "Bagging.hpp"

using namespace std;

class EBBagging: public Bagging{
	private:
		Data *positives;
		Data *negatives;

        void divideTrainData();

		Data generateBootstrap();

    public:
        EBBagging(Classifier *classifier, int nClassifiers);
		EBBagging(Classifier **classifiers, int nClassifiers);

        void train(Data *trainData);
};

#endif
