//#ifndef H_MLP_LIBCOMB
//#define H_MLP_LIBCOMB
//
//#include <cstdio>
//#include <iostream>
//#include <fstream> // file stream
//#include <iomanip>
//#include <string>
//#include <cstdlib>
//
//#include "Data.hpp"
//#include "OpenCVClassifier.hpp"
//
//#include <cv.h>
//#include <highgui.h>
//#include <ml.h>
//#include <cvaux.h>
//
//using namespace std;
//using namespace cv;
//
//class MLP: public OpenCVClassifier{
//	private:
//		CvANN_MLP *mlp; /// fazer destrutor para desalocar
//
//		Mat trainSamples;
//		Mat trainTrueLabels;
//
//		Data *trainData;
//
//		int nt;
//
//	public:
//        MLP();
//		MLP(Data *data);
//
//		void train(Data *trainData);
//		void predict(Data *data);
//		MLP* clone() const;
//};
//
//#endif
