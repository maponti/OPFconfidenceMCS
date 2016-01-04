#ifndef H_DATA_LIBCOMB
#define H_DATA_LIBCOMB

#include <cstdio>
#include <iostream>
#include <fstream> // file stream
#include <iomanip>
#include <string>
#include <cstdlib>
#include <cmath>

using namespace std;

class Data{
	private:
		float **featureVectors;
		int *classificationLabels;
		int *trueLabels;
		int nSamples;
		int nFeatures;
		int nLabels;

		double distEuclidean(float *, float *);

	public:
		Data(int nSamples, int nFeatures, int nLabels);
		Data(string file);
		Data(string file, int binary);
		~Data();

		void setFeature(int nSample, int nFeature, float value);
		float getFeature(int nSample, int nFeature);

		float *getFeatures(int nSample);

		void setClassificationLabel(int nSample, int label);
		int getClassificationLabel(int nSample);

		void setTrueLabel(int nSample, int label);
		int getTrueLabel(int nSample);

		int getNSamples();
		int getNFeatures();
		int getNLabels();

		int getNearestNeighbor(int);

		void writeData();

		Data* clone() const;
};

#endif
