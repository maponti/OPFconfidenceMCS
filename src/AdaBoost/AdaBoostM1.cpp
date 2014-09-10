#include "AdaBoostM1.hpp"

AdaBoostM1::AdaBoostM1(Data *trainData, Classifier **baseClassifiers, int nBaseClassifiers, int nIterations){
    train(trainData, baseClassifiers, nBaseClassifiers, nIterations);
}

AdaBoostM1::AdaBoostM1(Classifier **baseClassifiers, int nBaseClassifiers, int nIterations){
    this->nBaseClassifiers = nBaseClassifiers;
    this->baseClassifiers = new Classifier*[nBaseClassifiers];
    for(int i = 0; i < nBaseClassifiers; ++i){
        this->baseClassifiers[i] = baseClassifiers[i]->clone();
    }

    nFinalClassifiers = nIterations;

    setTrained(false);

    finalClassifiers = NULL;
    trainData = NULL;
    beta = NULL;
}

Data AdaBoostM1::generateSubSet(double *w, int subSetSize){
    Data subSet(subSetSize, trainData->getNFeatures(), trainData->getNLabels());
    default_random_engine generator;

    /// TENTAR OTIMIZAR
    vector<double> vect(trainData->getNSamples());
    for (size_t i = 0; i < vect.size(); ++i){
        vect.at(i) = w[i];
    }

    discrete_distribution<int> distribution(vect.begin(), vect.end());
    int randSample;

    for(int i = 0; i < subSetSize; ++i){
        // select a random sample
        randSample = distribution(generator);

        // copy sample
        for(int j = 0; j < trainData->getNFeatures(); ++j){
            subSet.setFeature(i, j, trainData->getFeature(randSample, j));
        }
        subSet.setTrueLabel(i, trainData->getTrueLabel(randSample));
    }

    return subSet;
}

void AdaBoostM1::train(Data *trainData, Classifier **baseClassifiers, int nBaseClassifiers, int nIterations){
    // set new base classifiers
    if(this->baseClassifiers != NULL){
        for(int i = 0; i < this->nBaseClassifiers; ++i){
            delete this->baseClassifiers[i];
        }
        delete [] this->baseClassifiers;
    }

    this->nBaseClassifiers = nBaseClassifiers;
    this->baseClassifiers = new Classifier*[nBaseClassifiers];
    for(int i = 0; i < nBaseClassifiers; ++i){
        this->baseClassifiers[i] = baseClassifiers[i]->clone();
    }

    // set the new final classifiers
    if(this->finalClassifiers != NULL){
        for(int i = 0; i < this->nFinalClassifiers; ++i){
            delete this->finalClassifiers[i];
        }
        delete [] this->finalClassifiers;
    }

    finalClassifiers = new Classifier*[nIterations];
    nFinalClassifiers = nIterations;

    if(beta != NULL){
        delete [] beta;
    }

    beta = new double[nIterations];

    train(trainData);
}

void AdaBoostM1::train(Data *trainData){
    // set the new final classifiers
    if(finalClassifiers == NULL){
        finalClassifiers = new Classifier*[nFinalClassifiers];
    }

    if(beta == NULL){
        beta = new double[nFinalClassifiers];
    }

    if(this->trainData != NULL){
        delete this->trainData;
    }

    this->trainData = trainData->clone();

    train();
}

void AdaBoostM1::train(){
    int nSamples = trainData->getNSamples();
    double w[nSamples];
    double e[nBaseClassifiers]; /// OTIMIZAR : fazer sem o vetor
    int min; /// menor erro
    double z;

    // initialize w
    for(int i = 0; i < nSamples; ++i){
        w[i] = 1.0 / nSamples;
    }

    Data subSet = generateSubSet(w, nSamples); /// qual o tamanho do subset ???

    for(int i = 0; i < nFinalClassifiers; ++i){
        //cerr << "it = " << i << endl;
        // error rate of classifiers
        for(int j = 0; j < nBaseClassifiers; ++j){
            baseClassifiers[j]->train(&subSet);
            baseClassifiers[j]->predict(trainData);

            // error rate e[j]
            e[j] = 0;
            for(int k = 0; k < nSamples; ++k){
                if(trainData->getClassificationLabel(k) != trainData->getTrueLabel(k)){
                    e[j] += w[k];
                }
            }
        }

        // find the classifier that minimizes e
        min = 0;
        for(int j = 1; j < nBaseClassifiers; ++j){
            if(e[j] < e[min]){
                min = j;
            }
        }

        ///if(e[min] == 0 || e[min] >= 0.5){ <<<< tirei o == 0
        if(e[min] >= 0.5){
            --i;

            // reset w
            for(int j = 0; j < nSamples; ++j){
                w[j] = 1.0 / nSamples;
            }

            continue;
        }

        beta[i] = e[min] / (1.0 - e[min]);

        /// OTIMIZAR : evitar refazer a previsao
        finalClassifiers[i] = baseClassifiers[min]->clone();

        finalClassifiers[i]->predict(trainData); /// <<< pode ser retirado se armazenar as classificacoes de todos os classificadores base

        // calculate the normalization factor z
        /// acho que pode nao calcular o z porque a distribuicao discreta ja normaliza
        z = 0;
        for(int k = 0; k < nSamples; ++k){
            if(trainData->getClassificationLabel(k) == trainData->getTrueLabel(k)){
                z += (w[k] * beta[i]);
            }else{
                z += w[k];
            }
        }

        for(int k = 0; k < nSamples; ++k){
            if(trainData->getClassificationLabel(k) == trainData->getTrueLabel(k)){
                w[k] = (w[k] * beta[i]) / z;
            }else{
                w[k] = w[k] / z;
            }
        }
    }
    setTrained(true);
}

void AdaBoostM1::predict(Data *data){
    int classification[nFinalClassifiers][data->getNSamples()];
    int classificationLabel;
    int nLabels = data->getNLabels();
    double support[nLabels];

    // classifies the data with all classifiers
    for(int i = 0; i < nFinalClassifiers; ++i){
        finalClassifiers[i]->predict(data);

        for(int j = 0; j < data->getNSamples(); ++j){
            classification[i][j] = data->getClassificationLabel(j);
        }
    }

    for(int j = 0; j < data->getNSamples(); ++j){
        for(int i = 0; i < nLabels; ++i){
            support[i] = 0;
        }

        for(int i = 0; i < nFinalClassifiers; ++i){
            support[classification[i][j]] += log(1.0 / beta[i]);
        }

        classificationLabel = 0;
        for(int i = 1; i < nLabels; ++i){
            if(support[i] > support[classificationLabel]){
                classificationLabel = i;
            }
        }

        // set classification
        data->setClassificationLabel(j, classificationLabel);
    }
}

void AdaBoostM1::predict(Data *data, double *double_fault, double *q_static, double *ir_agree,
                                double *desagree, double *correlation){
    //int classification[nFinalClassifiers][data->getNSamples()];
    int **classification = new int*[nFinalClassifiers];
    for(int i = 0; i < nFinalClassifiers; ++i) {
        classification[i] = new int[data->getNSamples()];
    }

    int classificationLabel;
    int nLabels = data->getNLabels();
    double support[nLabels];

    // classifies the data with all classifiers
    for(int i = 0; i < nFinalClassifiers; ++i){
        finalClassifiers[i]->predict(data);

        for(int j = 0; j < data->getNSamples(); ++j){
            classification[i][j] = data->getClassificationLabel(j);
        }
    }

    diversidade(classification, 2, data->getNSamples(), data, double_fault, q_static, ir_agree,
                desagree, correlation);

    for(int j = 0; j < data->getNSamples(); ++j){
        for(int i = 0; i < nLabels; ++i){
            support[i] = 0;
        }

        for(int i = 0; i < nFinalClassifiers; ++i){
            support[classification[i][j]] += log(1.0 / beta[i]);
        }

        classificationLabel = 0;
        for(int i = 1; i < nLabels; ++i){
            if(support[i] > support[classificationLabel]){
                classificationLabel = i;
            }
        }

        // set classification
        data->setClassificationLabel(j, classificationLabel);
    }
}


AdaBoostM1* AdaBoostM1::clone() const{
    if(trainData != NULL){
        return new AdaBoostM1(trainData, baseClassifiers, nBaseClassifiers, nFinalClassifiers);
    }
    return new AdaBoostM1(baseClassifiers, nBaseClassifiers, nFinalClassifiers);
}
