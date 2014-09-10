#include "AdaBoost.hpp"

AdaBoost::AdaBoost(Data *trainData, Classifier **classifiers, int nClassifiers, int nIterations){
    train(trainData, classifiers, nClassifiers, nIterations);
}

void AdaBoost::train(Data *trainData, Classifier **classifiers, int nClassifiers, int nIterations){
    int nSamples = trainData->getNSamples();
    double d[nSamples];
    double e[nClassifiers]; /// OTIMIZAR : fazer sem o vetor
    int max;
    int haux, yaux;
    double z;

    /// desalocar o antigo se houver
    alpha = new double[nIterations];
    this->classifiers = new Classifier*[nIterations];
    this->nClassifiers = nIterations;

    // initialize d
    for(int i = 0; i < nSamples; ++i){
        d[i] = 1.0 / nSamples;
    }

    for(int i = 0; i < nIterations; ++i){
        cout << "it = " << i << endl;
        // error rate of classifiers
        for(int j = 0; j < nClassifiers; ++j){
            classifiers[j]->train(trainData);
            classifiers[j]->predict(trainData);

            // error rate e[j]
            e[j] = 0;
            for(int k = 0; k < nSamples; ++k){
                if(trainData->getClassificationLabel(k) != trainData->getTrueLabel(k)){
                    e[j] += d[k];
                }
            }
            e[j] = abs(0.5 - e[j]);
        }

        // find the classifier that maximizes |0.5 - e[j]|
        max = 0;
        for(int j = 1; j < nClassifiers; ++j){
            cout << "j = " << j << ", e[j] = " << e[j] << ", max = " << max << ", e[max] = " << e[max] << endl;
            if(e[j] > e[max]){
                max = j;
            }
        }

        /// OTIMIZAR : evitar refazer a previsao
        this->classifiers[i] = classifiers[max]->clone();
        cout << "max = " << max << endl;
        alpha[i] = 0.5 * log((1.0 - e[max]) / e[max]);

        this->classifiers[i]->predict(trainData);

        // calculate the normalization factor Zt
        z = 0;
        for(int k = 0; k < nSamples; ++k){
            haux = (trainData->getClassificationLabel(k) == 0) ? -1 : 1;
            yaux = (trainData->getTrueLabel(k) == 0) ? -1 : 1;
            z += exp((-1.0) * alpha[i] * yaux * haux);
        }

        for(int k = 0; k < nSamples; ++k){
            haux = (trainData->getClassificationLabel(k) == 0) ? -1 : 1;
            yaux = (trainData->getTrueLabel(k) == 0) ? -1 : 1;
            d[k] = (d[k] * exp((-1.0) * alpha[i] * yaux * haux)) / z;
        }
    }
}

void AdaBoost::predict(Data *data){
    int classification[nClassifiers][data->getNSamples()];
    int classificationLabel;
    double out;

    // classifies the data with all classifiers
    for(int i = 0; i < nClassifiers; ++i){
        classifiers[i]->predict(data);

        for(int j = 0; j < data->getNSamples(); ++j){
            classification[i][j] = data->getClassificationLabel(j);
        }
    }

    for(int j = 0; j < data->getNSamples(); ++j){
        out = 0;
        for(int i = 0; i < nClassifiers; ++i){
            out += alpha[i] * classification[i][j];
        }

        classificationLabel = (out > 0.5) ? 1 : 0;

        // set classification
        data->setClassificationLabel(j, classificationLabel);
    }
}
