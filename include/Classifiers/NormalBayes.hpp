#ifndef H_NORMAL_BAYES_LIBCOMB
#define H_NORMAL_BAYES_LIBCOMB

#include <cstdio>
#include <iostream>
#include <fstream> // file stream
#include <iomanip>
#include <string>
#include <cstdlib>

#include "Data.hpp"
#include "OpenCVClassifier.hpp"

#include <cv.h>
#include <highgui.h>
#include <ml.h>
#include <cvaux.h>

using namespace std;
using namespace cv;

class NormalBayes: public OpenCVClassifier{
	private:
        Data *trainData;
		CvNormalBayesClassifier normalBayes;

		Mat trainSamples;
		Mat trainTrueLabels;

	public:
        NormalBayes();
		NormalBayes(Data *data);

		void train(Data *trainData);
		void predict(Data *data);
		NormalBayes* clone() const;
};

#endif
