#include "SBagging.hpp"

SBagging::SBagging(Classifier *classifier, int nClassifiers){
    this->nClassifiers = nClassifiers;
    this->classifiers = new Classifier*[nClassifiers];
    this->trainData = NULL;

    // creates the list of classifiers
    for(int i = 0; i < nClassifiers; ++i){
        this->classifiers[i] = classifier->clone();
    }
}

SBagging::SBagging(Classifier **classifiers, int nClassifiers){
    this->nClassifiers = nClassifiers;
    this->classifiers = new Classifier*[nClassifiers];
    this->trainData = NULL;

    // clones classifiers
    for(int i = 0; i < nClassifiers; ++i){
        this->classifiers[i] = classifiers[i]->clone();
    }
}

SBagging::~SBagging() {
	delete[] countClass;
	for (unsigned int i = 0; i < classData.size(); i++) {
		delete classData[i];
	}
}

void SBagging::divideTrainData(){
    /// USAR Except
    int nC = trainData->getNLabels();

    countClass = new int[nC]();
    int *kClass = new int[nC]();
    int nSamples = trainData->getNSamples();

    for(int i = 0; i < nSamples; ++i){
	int l = trainData->getTrueLabel(i)-1;
	if (l < 0 || l > nC-1) {
		cerr << "Error - label "<< l << " is not valid \n";
	}
	countClass[trainData->getTrueLabel(i)-1]++;
    }

    maxClass= 0;
    for(int i = 0; i < nC; i++) {
	    maxClass = (countClass[i] > maxClass) ? countClass[i] : maxClass;
    }
    newSize = maxClass * nC;

    for(int i = 0; i < nC; i++) {
	    classData.push_back(new Data(maxClass, trainData->getNFeatures(), trainData->getNLabels()));
    }

    for(int i = 0 ; i < nSamples; i++) {
	// copies the sample to each Data corresponding to a given class
	int l = trainData->getTrueLabel(i)-1;
	for(int j = 0; j < trainData->getNFeatures(); ++j){
		float val = trainData->getFeature(i,j);
		classData[l]->setFeature(kClass[l], j, val);
	}
	classData[l]->setTrueLabel(kClass[l], l+1);
	kClass[l]++;
   }

   delete[] kClass;

}

Data SBagging::generateBootstrap(){

    Data bootstrap(newSize, trainData->getNFeatures(), trainData->getNLabels());

    int nC = trainData->getNLabels();

    for(int i = 0; i < nC; i++) {
	    int k = countClass[i];
	    while (countClass[i] < classData[i]->getNSamples()) {
			int randI = rand() % countClass[i];
			int neibI = classData[i]->getNearestNeighbor(randI);
			float *a = classData[i]->getFeatures(randI);
			float *b = classData[i]->getFeatures(neibI);
			double w = ((double) rand() / (RAND_MAX)) + 1;
			for (int j = 0; j < classData[i]->getNFeatures(); j++) {
				float val = (1-w)*a[j] + w*b[j];
				classData[i]->setFeature(k, j, val);
			}
			classData[i]->setTrueLabel(k, i);
			k++;
			countClass[i]++;
	    }

    	    for (int k = 0; k < countClass[i]; ++k) {
	        // copies the sample
	        for(int j = 0; j < trainData->getNFeatures(); ++j){
	            bootstrap.setFeature(k, j, classData[i]->getFeature(k, j));
	        }
	        bootstrap.setTrueLabel(k, classData[i]->getTrueLabel(k));
	   }
    }

    return bootstrap;
}

void SBagging::train(Data *trainData){
    this->trainData = trainData->clone();
    divideTrainData();

    srand(time(NULL));

    for(int i = 0; i < nClassifiers; ++i){
        Data bootstrap = generateBootstrap();

	// testar antes daqui com dummy dataset

        classifiers[i]->train(&bootstrap);
    }
    combine->onTrain(classifiers, nClassifiers);
}
