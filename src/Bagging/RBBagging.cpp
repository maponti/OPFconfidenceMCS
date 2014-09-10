#include "RBBagging.hpp"

RBBagging::RBBagging(Classifier *classifier, int nClassifiers){
    this->nClassifiers = nClassifiers;
    this->classifiers = new Classifier*[nClassifiers];
    this->trainData = NULL;

    // creates the list of classifiers
    for(int i = 0; i < nClassifiers; ++i){
        this->classifiers[i] = classifier->clone();
    }
}

RBBagging::RBBagging(Classifier **classifiers, int nClassifiers){
    this->nClassifiers = nClassifiers;
    this->classifiers = new Classifier*[nClassifiers];
    this->trainData = NULL;

    // clones classifiers
    for(int i = 0; i < nClassifiers; ++i){
        this->classifiers[i] = classifiers[i]->clone();
    }
}

void RBBagging::divideTrainData(){
    /// USAR Except
    if(trainData->getNLabels() != 2){
        cout << "ERRO RBBagging divideTrainData!!!" << endl;
        exit(EXIT_FAILURE);
    }

    int nPos = 0;
    int nNeg = 0;
    int nSamples = trainData->getNSamples();

    for(int i = 0; i < nSamples; ++i){
        if(trainData->getTrueLabel(i) == 0) ++nNeg;
        else ++nPos;
    }

    positives = new Data(nPos, trainData->getNFeatures(), trainData->getNLabels());
    negatives = new Data(nNeg, trainData->getNFeatures(), trainData->getNLabels());

    int cntNeg = 0, cntPos = 0;
    for(int i = 0;  i < nSamples; ++i){
        if(trainData->getTrueLabel(i) == 0){
            // copies the sample
            for(int j = 0; j < trainData->getNFeatures(); ++j){
                negatives->setFeature(cntNeg, j, trainData->getFeature(i, j));
            }
            negatives->setTrueLabel(cntNeg, trainData->getTrueLabel(i));
            ++cntNeg;
        }else{
            // copies the sample
            for(int j = 0; j < trainData->getNFeatures(); ++j){
                positives->setFeature(cntPos, j, trainData->getFeature(i, j));
            }
            positives->setTrueLabel(cntPos, trainData->getTrueLabel(i));
            ++cntPos;
        }
    }
}

Data RBBagging::generateBootstrap(){
    negative_binomial_distribution<int> distribution(positives->getNSamples(),0.5);
    int nPos = positives->getNSamples();
    int nNeg = distribution(generator);

    Data bootstrap(nPos + nNeg, trainData->getNFeatures(), trainData->getNLabels());
    int randSample;
    int nFeatures = trainData->getNFeatures();

    ///cout << "nNeg = " << nNeg << endl;

    // copies all positives
    for(int i = 0; i < nPos; ++i){
        // copies the sample
        for(int j = 0; j < nFeatures; ++j){
            bootstrap.setFeature(i, j, positives->getFeature(i, j));
        }
        bootstrap.setTrueLabel(i, positives->getTrueLabel(i));
    }

    for(int i = nPos; i < (nPos + nNeg); ++i){
        // selects a random sample
        randSample = rand() % negatives->getNSamples();

        // copies the sample
        for(int j = 0; j < nFeatures; ++j){
            bootstrap.setFeature(i, j, negatives->getFeature(randSample, j));
        }
        bootstrap.setTrueLabel(i, negatives->getTrueLabel(randSample));
    }

    return bootstrap;
}

void RBBagging::train(Data *trainData){
    this->trainData = trainData->clone();
    divideTrainData();

    srand(time(NULL));
    generator.seed(time(NULL));

    for(int i = 0; i < nClassifiers; ++i){
        Data bootstrap = generateBootstrap();

        classifiers[i]->train(&bootstrap);
    }
}