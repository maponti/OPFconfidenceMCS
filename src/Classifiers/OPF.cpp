#include "OPF.hpp"

OPF::OPF(){
    trainData = NULL;
    subgraph = NULL;
}

OPF::OPF(Data *trainData){
    this->trainData = NULL;
    subgraph = NULL;
	train(trainData);
}

OPF::~OPF(){
    if(trainData != NULL){
        delete trainData;
    }

    if(subgraph != NULL){
        DestroySubgraph(&subgraph);
    }
}

Subgraph* OPF::data2Subgraph(Data *data){
	int cntSamples,
		cntFeatures,
		nSamples = data->getNSamples(),
		nFeatures = data->getNFeatures();
	Subgraph *g = CreateSubgraph(nSamples);

	g->nlabels = data->getNLabels();
	g->nfeats = nFeatures;

	for(cntSamples = 0; cntSamples < nSamples; ++cntSamples){
		g->node[cntSamples].feat = AllocFloatArray(g->nfeats);
		g->node[cntSamples].position = cntSamples; /// esta certo?????
		g->node[cntSamples].truelabel = data->getTrueLabel(cntSamples) ;
		for(cntFeatures = 0; cntFeatures < nFeatures; ++cntFeatures){
			g->node[cntSamples].feat[cntFeatures] = data->getFeature(cntSamples, cntFeatures);
		}
	}

	return g;
}

void OPF::train(Data *trainData){
    if(this->trainData != NULL){
        delete this->trainData;
    }
    this->trainData = trainData->clone();

    if(subgraph != NULL){
        free(subgraph);
    }
	subgraph = data2Subgraph(this->trainData);

	opf_OPFTraining(subgraph);
}

void OPF::predict(Data *data){
	Subgraph *g = data2Subgraph(data);
	int	cntSamples,
		nSamples;

	opf_OPFClassifying(subgraph, g);

	for(cntSamples = 0, nSamples = data->getNSamples(); cntSamples < nSamples; ++cntSamples){
		data->setClassificationLabel(cntSamples, g->node[cntSamples].label);
	}
	DestroySubgraph(&g);
}

OPF* OPF::clone() const{
    if(trainData != NULL){
        return new OPF(trainData);
    }
    return new OPF();
}
