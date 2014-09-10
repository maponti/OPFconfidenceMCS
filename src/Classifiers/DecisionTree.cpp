#include "DecisionTree.hpp"

DecisionTree2::DecisionTree2(){
    trainData = NULL;
}

DecisionTree2::DecisionTree2(Data *data){
    trainData = NULL;
	train(data);
}

void DecisionTree2::train(Data *trainData){
    if(this->trainData != NULL){
        delete this->trainData;
    }
    this->trainData = trainData->clone();

	trainSamples = dataExtractFeatureVectors(trainData);
	trainTrueLabels = dataExtractTrueLabels(trainData);

	decisionTree.train(trainSamples, CV_ROW_SAMPLE, trainTrueLabels);
}

void DecisionTree2::predict(Data *data){
	int	nSamples = data->getNSamples(),
		cntSamples;
	Mat predictSamples = dataExtractFeatureVectors(data),
		labels(nSamples, 1, CV_32FC1);
    CvDTreeNode* resultNode;

	for(cntSamples = 0; cntSamples < nSamples; ++cntSamples){
	    resultNode = decisionTree.predict(predictSamples.row(cntSamples));

		data->setClassificationLabel(cntSamples, (int)resultNode->value);
	}
}

DecisionTree2* DecisionTree2::clone() const{
    if(trainData != NULL){
        return new DecisionTree2(trainData);
    }
    return new DecisionTree2();
}
