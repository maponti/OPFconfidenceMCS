#ifndef H_DATA_LIBCOMB
#define H_DATA_LIBCOMB

#include <cstdio>
#include <iostream>
#include <fstream> // file stream
#include <iomanip>
#include <string>
#include <cstdlib>

using namespace std;

class Data{
	private:
		float **featureVectors;
		int *classificationLabels;
		int *trueLabels;
		int nSamples;
		int nFeatures;
		int nLabels;

	public:
		Data(int nSamples, int nFeatures, int nLabels);
		Data(string file);
		~Data();

		void setFeature(int nSample, int nFeature, float value);
		float getFeature(int nSample, int nFeature);

		void setClassificationLabel(int nSample, int label);
		int getClassificationLabel(int nSample);

		void setTrueLabel(int nSample, int label);
		int getTrueLabel(int nSample);

		int getNSamples();
		int getNFeatures();
		int getNLabels();

		Data* clone() const;
};

#endif
