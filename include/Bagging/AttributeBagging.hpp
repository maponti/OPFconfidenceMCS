#ifndef H_ATTRIBUTE_BAGGING_LIBCOMB
#define H_ATTRIBUTE_BAGGING_LIBCOMB

#include <cstdio>
#include <iostream>
#include <fstream> // file stream
#include <iomanip>
#include <string>
#include <cstdlib>

#include "Classifier.hpp"
#include "Data.hpp"

class AttributeBagging{
	private:
		Classifier **classifiers;
		int nClassifiers;
		int subSetSize;
		Data *trainData;

		Data generateSubSet();

	public:
		AttributeBagging(Data *trainData, Classifier **classifiers, int nClassifiers, int subSetSize);

		void train(Data *trainData, Classifier **classifiers, int nClassifiers, int subSetSize);
		void predict(Data *data);
};

#endif
