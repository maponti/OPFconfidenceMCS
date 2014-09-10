#include "IRBaggingOver.hpp"

IRBaggingOver::IRBaggingOver(Classifier **classifiers, int nClassifiers){
    this->nClassifiers = nClassifiers;
    this->classifiers = new Classifier*[nClassifiers];
    this->trainData = NULL;

    // clones classifiers
    for(int i = 0; i < nClassifiers; ++i){
        this->classifiers[i] = classifiers[i]->clone();
    }
}

IRBaggingOver::IRBaggingOver(Classifier *classifier, int nClassifiers){
    this->nClassifiers = nClassifiers;
    this->classifiers = new Classifier*[nClassifiers];
    this->trainData = NULL;

    // creates the list of classifiers
    for(int i = 0; i < nClassifiers; ++i){
        this->classifiers[i] = classifier->clone();
    }
}

void IRBaggingOver::train(Data *trainData){
    this->trainData = trainData->clone();
    divideTrainData();

    srand(time(NULL));

    //IR increase
    double dIR = ((double)negatives->getNSamples() / (double)positives->getNSamples() - 1.0) / (double)nClassifiers;

    // initial IR
    bootstrapIR = 1.0;
    for(int i = 0; i < nClassifiers; ++i){
        Data bootstrap = generateBootstrap();

        classifiers[i]->train(&bootstrap);

        bootstrapIR += dIR;
    }
}

void IRBaggingOver::divideTrainData(){
    /// USAR Except
    if(trainData->getNLabels() != 2){
        cout << "ERRO IRBaggingOver divideTrainData!!!" << endl;
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

Data IRBaggingOver::generateBootstrap(){
    int nNeg = negatives->getNSamples();
    int positiveSamples = positives->getNSamples();
    int nPos = (nNeg / bootstrapIR > positiveSamples) ? nNeg / bootstrapIR : positiveSamples;

    Data bootstrap(nPos + nNeg, trainData->getNFeatures(), trainData->getNLabels());
    int randSample;
    int nFeatures = trainData->getNFeatures();

    // copies all positives
    for(int i = 0; i < positiveSamples; ++i){
        // copies the sample
        for(int j = 0; j < nFeatures; ++j){
            bootstrap.setFeature(i, j, positives->getFeature(i, j));
        }
        bootstrap.setTrueLabel(i, positives->getTrueLabel(i));
    }

    for(int i = positiveSamples; i < nPos; ++i){
        // selects a random sample
        randSample = rand() % positiveSamples;

        // copies the sample
        for(int j = 0; j < nFeatures; ++j){
            bootstrap.setFeature(i, j, positives->getFeature(randSample, j));
        }
        bootstrap.setTrueLabel(i, positives->getTrueLabel(randSample));
    }

    // copies all negatives
    for(int i = 0; i < nNeg; ++i){
        // copies the sample
        for(int j = 0; j < nFeatures; ++j){
            bootstrap.setFeature(i + nPos, j, negatives->getFeature(i, j));
        }
        bootstrap.setTrueLabel(i + nPos, negatives->getTrueLabel(i));
    }

    return bootstrap;
}
