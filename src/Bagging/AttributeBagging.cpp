#include "AttributeBagging.hpp"

AttributeBagging::AttributeBagging(Data *trainData, Classifier **classifiers, int nClassifiers, int subSetSize){
    train(trainData, classifiers, nClassifiers, subSetSize);
}

void AttributeBagging::train(Data *trainData, Classifier **classifiers, int nClassifiers, int subSetSize){
    this->trainData = trainData;
    this->classifiers = classifiers;
    this->nClassifiers = nClassifiers;
    this->subSetSize = subSetSize;

    srand (time(NULL));

    for(int i = 0; i < nClassifiers; ++i){
//        Data subSet = generateSubSet();

//        classifiers[i]->train(&subSet);
    }
}

/*Data AttributeBagging::generateSubSet(){
    Data subSet(trainData->getNSamples(), subSetNFeatures, trainData->getNLabels());
    int features[subSetNFeatures];

    // select features without replacement
    int randFeature;
    for(int j = 0, int i = 0; i < subSetNFeatures;){
        randFeature = rand() % trainData->getNFeatures();

        for(j = 0; j < i; ++j){
            if(features[j] == randFeature){
                break;
            }
        }

        if(j == i){
            ++i;
        }
    }

    // copy selected features
    for(int i = 0; i < subSetNFeatures; ++i){
        // copy a feature
        for(int j = 0; j < trainData->getNSamples(); ++j){
            subSet.setFeature(j, features[i], trainData->getFeature(j, features[i]));
        }
    }

    // copy true labels
    for(int i = 0; i < trainData->getNSamples(); ++i){
        subSet.setTrueLabel(i, trainData->getTrueLabel(i));
    }

    return subSet;
}*/

void AttributeBagging::predict(Data *data){
    int classification[nClassifiers][data->getNSamples()];
    int votes[data->getNLabels()];
    int classificationLabel;

    // classifies the data with all classifiers
    for(int i = 0; i < nClassifiers; ++i){
        classifiers[i]->predict(data);

        for(int j = 0; j < data->getNSamples(); ++j){
            classification[i][j] = data->getClassificationLabel(j);
        }
    }

    for(int j = 0; j < data->getNSamples(); ++j){
        // clear votes
        for(int i = 0; i < data->getNLabels(); ++i){
            votes[i] = 0;
        }

        // compute # of votes
        for(int i = 0; i < nClassifiers; ++i){
           ++votes[classification[i][j]];
        }

        // compute classification label
        classificationLabel = 0;
        for(int i = 0; i < data->getNLabels(); ++i){
            if(votes[i] > votes[classificationLabel]){
                classificationLabel = i;
            }
        }

        // set classification
        data->setClassificationLabel(j, classificationLabel);
    }
}
