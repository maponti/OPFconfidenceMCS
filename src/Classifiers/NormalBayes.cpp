#include "NormalBayes.hpp"

NormalBayes::NormalBayes(){
    trainData = NULL;
}

NormalBayes::NormalBayes(Data *data){
    trainData = NULL;
	train(data);
}

void NormalBayes::train(Data *trainData){
    if(this->trainData != NULL){
        delete this->trainData;
    }
    this->trainData = trainData->clone();

	trainSamples = dataExtractFeatureVectors(trainData);
	trainTrueLabels = dataExtractTrueLabels(trainData);

	normalBayes.train(trainSamples, trainTrueLabels);
}

void NormalBayes::predict(Data *data){
	int	nSamples = data->getNSamples(),
		cntSamples;
	Mat predictSamples = dataExtractFeatureVectors(data),
		labels(nSamples, 1, CV_32FC1);

	normalBayes.predict(predictSamples, &labels);

	for(cntSamples = 0; cntSamples < nSamples; ++cntSamples){
		data->setClassificationLabel(cntSamples, (int)labels.at<float>(cntSamples, 0));
	}
}

NormalBayes* NormalBayes::clone() const{
    if(trainData != NULL){
        return new NormalBayes(trainData);
    }
    return new NormalBayes();
}
