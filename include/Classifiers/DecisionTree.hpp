#ifndef H_DECISION_TREE_LIBCOMB
#define H_DECISION_TREE_LIBCOMB

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

class DecisionTree2: public OpenCVClassifier{
	private:
        Data *trainData;

		CvDTree decisionTree;

		Mat trainSamples;
		Mat trainTrueLabels;

	public:
        DecisionTree2();
		DecisionTree2(Data *data);

		void train(Data *trainData);
		void predict(Data *data);
		DecisionTree2* clone() const;
};

#endif
