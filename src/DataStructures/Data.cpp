#include "Data.hpp"

Data::Data(int nSamples, int nFeatures, int nLabels){
	// sets # of samples, # of labels and # of features
	this->nFeatures = nFeatures;
	this->nSamples = nSamples;
	this->nLabels = nLabels;

	featureVectors = new float*[nSamples];
	for(int i = 0; i < nSamples; ++i){
		featureVectors[i] = new float[nFeatures];
	}
	classificationLabels = new int[nSamples];
	trueLabels = new int[nSamples];
}

Data::Data(string trainFilePath){
	ifstream	trainFileStream(trainFilePath.data(), ios::in);
	int			cntSamples,
				cntFeatures,
				auxTrueLabel;
	float		auxFeature;

	// exit program if ifstream could not open file
	if (!trainFileStream){
		cerr << "File could not be opened" << endl;
		exit(1);
	}

	// reads # of samples, # of features and # of labels
	trainFileStream >> nSamples >> nFeatures >> nLabels;
	///cout << nSamples << " " << getNLabels() << " " << nFeatures << "\n";

	// allocates memory
	featureVectors = new float*[nSamples];
	for(int i = 0; i < nSamples; ++i){
		featureVectors[i] = new float[nFeatures];
	}
	classificationLabels = new int[nSamples];
	trueLabels = new int[nSamples];

	// reads feature vectors
	for(cntSamples = 0; cntSamples < nSamples; ++cntSamples){
		for(cntFeatures = 0; cntFeatures < nFeatures; ++cntFeatures){
			trainFileStream >> auxFeature;
			setFeature(cntSamples, cntFeatures, (float)auxFeature);
		}
        trainFileStream >> auxTrueLabel;
		setTrueLabel(cntSamples, auxTrueLabel);
	}
}

Data::~Data(){
	for(int i = 0; i < this->nSamples; ++i){
		delete [] featureVectors[i];
	}
	delete [] featureVectors;

	delete [] classificationLabels;
	delete [] trueLabels;

	this->nFeatures = 0;
	this->nSamples = 0;
}

void Data::setFeature(int nSample, int nFeature, float value){
	/// fazer checagem de limites
	featureVectors[nSample][nFeature] = value;
}
float Data::getFeature(int nSample, int nFeature){
	/// fazer checagem de limites
	return featureVectors[nSample][nFeature];
}

void Data::setClassificationLabel(int nSample, int label){
	/// fazer checagem de limites
	classificationLabels[nSample] = label;
}
int Data::getClassificationLabel(int nSample){
	/// fazer checagem de limites
	return classificationLabels[nSample];
}

void Data::setTrueLabel(int nSample, int label){
	/// fazer checagem de limites
	trueLabels[nSample] = label;
}
int Data::getTrueLabel(int nSample){
	/// fazer checagem de limites
	return trueLabels[nSample];
}

int Data::getNSamples(){
	/// fazer checagem de limites
	return this->nSamples;
}

int Data::getNFeatures(){
	/// fazer checagem de limites
	return this->nFeatures;
}

int Data::getNLabels(){
	/// fazer checagem de limites
	return this->nLabels;
}

Data* Data::clone() const{
    Data *clone = new Data(nSamples, nFeatures, nLabels);

	for(int cntSamples = 0; cntSamples < nSamples; ++cntSamples){
		for(int cntFeatures = 0; cntFeatures < nFeatures; ++cntFeatures){
		    clone->setFeature(cntSamples, cntFeatures, featureVectors[cntSamples][cntFeatures]);
		}
		clone->setClassificationLabel(cntSamples, classificationLabels[cntSamples]);
		clone->setTrueLabel(cntSamples, trueLabels[cntSamples]);
	}

	return clone;
}
