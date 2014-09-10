#ifndef H_KNN_LIBCOMB
#define H_KNN_LIBCOMB

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

class KNN: public OpenCVClassifier{
	private:
        Data *trainData;

		CvKNearest knn;

		int k;

		Mat trainSamples;
		Mat trainTrueLabels;

	public:
        KNN(int k);
		KNN(Data *data, int k);

		int getK();
		void setK(int k);

		void train(Data *trainData);
		void predict(Data *data);
		KNN* clone() const;
};

#endif
