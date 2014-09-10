#include "OpenCVClassifier.hpp"

Mat OpenCVClassifier::dataExtractFeatureVectors(Data *data){
	int	nSamples = data->getNSamples(),
		nFeatures = data->getNFeatures(),
		cntSamples,
		cntFeatures;
	Mat features(nSamples, nFeatures, CV_32FC1);

	for(cntSamples = 0; cntSamples < nSamples; ++cntSamples){
	    // copy features
		for(cntFeatures = 0; cntFeatures < nFeatures; ++cntFeatures){
			features.at<float>(cntSamples, cntFeatures) = data->getFeature(cntSamples, cntFeatures);
		}
	}
	return features;
}

Mat OpenCVClassifier::dataExtractTrueLabels(Data *data){
	int	nSamples = data->getNSamples(),
		cntSamples;
	Mat trueLabels(nSamples, 1, CV_32FC1);

	for(cntSamples = 0; cntSamples < nSamples; ++cntSamples){
	    // copy true label
		trueLabels.at<float>(cntSamples, 0) = data->getTrueLabel(cntSamples);
	}

	return trueLabels;
}

