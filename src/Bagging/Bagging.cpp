#include "Bagging.hpp"
#include <iostream>

Bagging::Bagging(){
    this->combine = new Combinator;
}

Bagging::Bagging(Classifier *classifier, int nClassifiers, int bootstrapSize){
    this->nClassifiers = nClassifiers;
    this->bootstrapSize = bootstrapSize;
    this->classifiers = new Classifier*[nClassifiers];
    this->trainData = NULL;
    this->combine = new Combinator;

    // creates the list of classifiers
    for(int i = 0; i < nClassifiers; ++i){
        this->classifiers[i] = classifier->clone();
        if(!this->classifiers[i]->hasScore() && classifier->hasScore()) std::cout << "error\n";
    }
}

Bagging::Bagging(Data *trainData, Classifier *classifier, int nClassifiers, int bootstrapSize){
    train(trainData, classifier, nClassifiers, bootstrapSize);
    this->combine = new Combinator;
}

Bagging::Bagging(Classifier **classifiers, int nClassifiers, int bootstrapSize){
    this->nClassifiers = nClassifiers;
    this->bootstrapSize = bootstrapSize;
    this->classifiers = new Classifier*[nClassifiers];
    this->trainData = NULL;
    this->combine = new Combinator;

    // clones classifiers
    for(int i = 0; i < nClassifiers; ++i){
        this->classifiers[i] = classifiers[i]->clone();
    }
}

Bagging::Bagging(Data *trainData, Classifier **classifiers, int nClassifiers, int bootstrapSize){
    this->combine = new Combinator;
    train(trainData, classifiers, nClassifiers, bootstrapSize);
}

Bagging::~Bagging(){
    if(trainData != NULL){
        delete trainData;
    }

    // deletes classifiers
    for(int i = 0; i < nClassifiers; ++i){
        delete classifiers[i];
    }
    delete [] classifiers;

}

void Bagging::train(Data *trainData, Classifier **classifiers, int nClassifiers, int bootstrapSize){
    this->nClassifiers = nClassifiers;
    this->bootstrapSize = bootstrapSize;
    this->classifiers = new Classifier*[nClassifiers]; /// apagar vetor antigo (se houver)

    // clones classifiers
    for(int i = 0; i < nClassifiers; ++i){
        this->classifiers[i] = classifiers[i]->clone();
    }

    train(trainData);
}

void Bagging::train(Data *trainData, Classifier *classifier, int nClassifiers, int bootstrapSize){
    this->nClassifiers = nClassifiers;
    this->bootstrapSize = bootstrapSize;
    this->classifiers = new Classifier*[nClassifiers]; /// apagar vetor antigo (se houver)

    // clones classifiers
    for(int i = 0; i < nClassifiers; ++i){
        this->classifiers[i] = classifier->clone();
    }

    train(trainData);
}

void Bagging::train(Data *trainData){
    this->trainData = trainData->clone(); /// apagar conjunto antigo (se houver)

    srand(time(NULL));

    for(int i = 0; i < nClassifiers; ++i){
        Data bootstrap = generateBootstrap();

        classifiers[i]->train(&bootstrap);
    }
    combine->onTrain(classifiers, nClassifiers);
}

Data Bagging::generateBootstrap(){
    Data bootstrap(bootstrapSize, trainData->getNFeatures(), trainData->getNLabels());
    int randSample;

    for(int i = 0; i < bootstrapSize; ++i){
        // selects a random sample
        randSample = rand() % trainData->getNSamples();

        // copies the sample
        for(int j = 0; j < trainData->getNFeatures(); ++j){
            bootstrap.setFeature(i, j, trainData->getFeature(randSample, j));
        }
        bootstrap.setTrueLabel(i, trainData->getTrueLabel(randSample));
    }

    return bootstrap;
}

void Bagging::predict(Data *data){
    //int classification[nClassifiers][data->getNSamples()];
    int **classification = new int*[nClassifiers];
    for(int i = 0; i < nClassifiers; ++i) {
        classification[i] = new int[data->getNSamples()];
    }
    // classifies the data with all classifiers
    for(int i = 0; i < nClassifiers; ++i){
        classifiers[i]->predict(data);
        for(int j = 0; j < data->getNSamples(); ++j){
            classification[i][j] = data->getClassificationLabel(j);
        }
    }

    (*combine)(data, classification, classifiers, nClassifiers);
    for(int i = 0; i < nClassifiers; ++i) {
        delete[] classification[i];
    }
    delete[] classification;
}

void Bagging::predict(Data *data, double *double_fault, double *q_static, double *ir_agree,
                                double *desagree, double *correlation){
    //int classification[nClassifiers][data->getNSamples()];
    int **classification = new int*[nClassifiers];
    for(int i = 0; i < nClassifiers; ++i) {
        classification[i] = new int[data->getNSamples()];
    }
    // classifies the data with all classifiers
    for(int i = 0; i < nClassifiers; ++i){
        classifiers[i]->predict(data);

        for(int j = 0; j < data->getNSamples(); ++j){
            classification[i][j] = data->getClassificationLabel(j);
        } 
    }

    diversidade(classification, 2, data->getNSamples(), data, double_fault, q_static, ir_agree,
                desagree, correlation);

    (*combine)(data, classification, classifiers, nClassifiers);
    for(int i = 0; i < nClassifiers; ++i) {
        delete[] classification[i];
    }
    delete[] classification;
}

Bagging* Bagging::clone() const{ /// o clone nao fica igual porque o treinamento eh nao deterministico
    Bagging * bg;
    if(trainData != NULL){
        bg = new Bagging(trainData, classifiers, nClassifiers, bootstrapSize);
    }
    else bg = new Bagging(classifiers, nClassifiers, bootstrapSize);
    bg->setCombinator(combine);
    return bg;
}

void Bagging::setCombinator(Combinator * cb)
{
    if(combine) delete combine;
    combine = cb;
    if (trainData) combine->onTrain(classifiers, nClassifiers);
}
