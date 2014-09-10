#include "KNN.hpp"

KNN::KNN(int k){
    trainData = NULL;
    setK(k);
}

KNN::KNN(Data *data, int k){
    trainData = NULL;
	setK(k);
	train(data);
}

void KNN::train(Data *trainData){
    if(this->trainData != NULL){
        delete this->trainData;
    }
    this->trainData = trainData->clone();

	trainSamples = dataExtractFeatureVectors(trainData);
	trainTrueLabels = dataExtractTrueLabels(trainData);

	knn.train(trainSamples, trainTrueLabels, Mat(), false, getK(), false);
}

void KNN::predict(Data *data){
	int	nSamples = data->getNSamples(),
		cntSamples;
	Mat predictSamples = dataExtractFeatureVectors(data),
		labels(nSamples, 1, CV_32FC1);

	knn.find_nearest(predictSamples, getK(), &labels);

	for(cntSamples = 0; cntSamples < nSamples; ++cntSamples){
		data->setClassificationLabel(cntSamples, (int)labels.at<float>(cntSamples, 0));
	}
}

int KNN::getK(){
	return k;
}

void KNN::setK(int k){
	this->k = k;

	if(trainData != NULL){
        knn.train(trainSamples, trainTrueLabels, Mat(), false, getK(), false);
	}
}

KNN* KNN::clone() const{
    if(trainData != NULL){
        return new KNN(trainData, k);
    }
    return new KNN(k);
}
