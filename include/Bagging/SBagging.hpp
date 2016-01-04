#ifndef H_S_BAGGING_LIBCOMB
#define H_S_BAGGING_LIBCOMB

#include <cstdio>
#include <iostream>
#include <fstream> // file stream
#include <iomanip>
#include <string>
#include <cstdlib>
#include <random>
#include <vector>
#include <ctime>

#include "Classifier.hpp"
#include "Data.hpp"
#include "Bagging.hpp"

using namespace std;

class SBagging: public Bagging{
	private:
		vector<Data *> classData;
		int *countClass;
		int newSize;
		int maxClass;
		void divideTrainData();

		Data generateBootstrap();
		virtual ~SBagging();

	public:
		SBagging(Classifier *classifier, int nClassifiers);
		SBagging(Classifier **classifiers, int nClassifiers);
		void train(Data *trainData);
};

#endif
