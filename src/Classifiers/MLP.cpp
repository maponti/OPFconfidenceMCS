//#include "MLP.hpp"
//
//MLP::MLP(){
//    trainData = NULL;
//    mlp = new CvANN_MLP;
//}
//
//MLP::MLP(Data *data){
//    trainData = NULL;
//    mlp = new CvANN_MLP;
//
//	train(data);
//}
//
//void MLP::train(Data *trainData){
//    if(this->trainData != NULL){
//        delete this->trainData;
//    }
//    this->trainData = trainData->clone();
//
//     /// TESTE
//    int layers_d[] = { trainData->getNFeatures(), 10,  trainData->getNLabels()};
//    Mat layers = Mat(1,3,CV_32SC1);
//    layers.at<int>(0,0) = layers_d[0];
//    layers.at<int>(0,1) = layers_d[1];
//    layers.at<int>(0,2) = layers_d[2];
//
//    // create the network using a sigmoid function with alpha and beta
//    // parameters 0.6 and 1 specified respectively (refer to manual)
//
//    mlp->create(layers);
//    /// ------------
//
//    nt = trainData->getNSamples();
//
//	trainSamples = dataExtractFeatureVectors(trainData);
//	trainTrueLabels = dataExtractTrueLabels(trainData);
//
//	mlp->train(trainSamples, trainTrueLabels, Mat());
//}
//
//void MLP::predict(Data *data){
//	int	nSamples = data->getNSamples(),
//		cntSamples,
//		max;
//	Mat predictSamples = dataExtractFeatureVectors(data),
//		labels(nt, data->getNLabels(), CV_32FC1);
//
//
//	for(cntSamples = 0; cntSamples < nSamples; ++cntSamples){
//        mlp->predict(predictSamples.row(cntSamples), labels);
//        max = 0;
//        for(int i = 1; i < data->getNLabels(); ++i){
//            if(labels.at<float>(cntSamples, i) > labels.at<float>(cntSamples, max)){
//                max = i;
//            }
//        }
//
//		data->setClassificationLabel(cntSamples, max);
//	}
//}
//
//MLP* MLP::clone() const{
//    if(trainData != NULL){
//        return new MLP(trainData);
//    }
//    return new MLP();
//}
