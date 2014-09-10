#ifndef H_RB_BAGGING_LIBCOMB
#define H_RB_BAGGING_LIBCOMB

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

class RBBagging: public Bagging{
	private:
		Data *positives;
		Data *negatives;

		default_random_engine generator;

        void divideTrainData();

		Data generateBootstrap();

    public:
        RBBagging(Classifier *classifier, int nClassifiers);
		RBBagging(Classifier **classifiers, int nClassifiers);

        void train(Data *trainData);
};

#endif
