#ifndef H_OPENCV_CLASSIFIER_LIBCOMB
#define H_OPENCV_CLASSIFIER_LIBCOMB

#include <cstdio>
#include <iostream>
#include <fstream> // file stream
#include <iomanip>
#include <string>
#include <cstdlib>

#include "Data.hpp"
#include "Classifier.hpp"

#include <cv.h>
#include <highgui.h>
#include <ml.h>
#include <cvaux.h>

using namespace std;
using namespace cv;

class OpenCVClassifier: public Classifier{
	protected:
		Mat dataExtractFeatureVectors(Data *data);
		Mat dataExtractTrueLabels(Data *data);

	public:
		virtual void train(Data *trainData) = 0;
		virtual void predict(Data *data) = 0;
		virtual Classifier* clone() const = 0;
};

#endif
