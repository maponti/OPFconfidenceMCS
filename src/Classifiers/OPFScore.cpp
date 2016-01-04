#include "Data.hpp"
#include "OPFScore.hpp"
#include <iostream>
#include <fstream>

static void opf_OPFClassifyingScore(Subgraph *sgtrain, Subgraph *sg, float **minLabel)
{
	int i, j, k, l, p, label = -1;
	float tmp, weight, minCost, sumCost;
	int nlabels = sgtrain->nlabels;
	//matrix to store scores for each object and class
	//float **minLabel = (float **) calloc(sg->nnodes, sizeof(float *));

	// for each test set node


	for (i = 0; i < sg->nnodes; i++)  
	{
		// allocate row for element i and fill with max float value
		//minLabel[i] = (float *) calloc(nlabels, sizeof(float));
		for (p=0; p < nlabels; p++) { minLabel[i][p]=FLT_MAX; }

		j = 0;
		k = sgtrain->ordered_list_of_nodes[j];

		if(!opf_PrecomputedDistance)
			weight = opf_ArcWeight(sgtrain->node[k].feat,sg->node[i].feat,sg->nfeats);
		else
			weight = opf_DistanceValue[sgtrain->node[k].position][sg->node[i].position];

		minCost = MAX(sgtrain->node[k].pathval, weight);
		label   = sgtrain->node[k].label;

		minLabel[i][label-1] = minCost;
		while ((j < sgtrain->nnodes-1) &&
			  (minCost > sgtrain->node[sgtrain->ordered_list_of_nodes[j+1]].pathval))
		{

			l  = sgtrain->ordered_list_of_nodes[j+1];

			if(!opf_PrecomputedDistance)
				weight = opf_ArcWeight(sgtrain->node[l].feat,sg->node[i].feat,sg->nfeats);
			else
				weight = opf_DistanceValue[sgtrain->node[l].position][sg->node[i].position];
			tmp = MAX(sgtrain->node[l].pathval, weight);

			if (tmp < minLabel[i][sgtrain->node[l].label-1]) 
			{       
				minLabel[i][sgtrain->node[l].label-1] = tmp;
			}

			if (tmp < minCost)
			{
				minCost = tmp;
				label = sgtrain->node[l].label;
			}
		
			j++;
			k  = l;
		}
		sumCost = 0.0;
		for (p=0; p < nlabels; p++) 
		{ 
			sumCost+= minLabel[i][p];  
		}
		for (p=0; p < nlabels; p++) 
		{ 
			minLabel[i][p] = ((1.0-(minLabel[i][p])/(float)sumCost)) / (nlabels-1);	
		}
		sg->node[i].label = label;
	}
}

OPFScore::OPFScore()
	: OPF(), scores(0), scoreSize(0), scoresNumClasses(0)
{

}

OPFScore::OPFScore(Data *data)
	: OPF(data), scores(0), scoreSize(0), scoresNumClasses(0)
{

}

OPFScore::~OPFScore()
{
	destroyScores();
}


void OPFScore::predict(Data *data)
{
	Subgraph *g = data2Subgraph(data);
	int	cntSamples,
	nSamples;
	destroyScores();

	scoreSize = data->getNSamples();
	scoresNumClasses = subgraph->nlabels;

	scores = new float*[scoreSize];
	for (int i = 0; i < scoreSize; i++)
		scores[i] = new float[scoresNumClasses]; 

	opf_OPFClassifyingScore(subgraph, g, scores);

	for(cntSamples = 0, nSamples = data->getNSamples(); cntSamples < nSamples; ++cntSamples){
		data->setClassificationLabel(cntSamples, g->node[cntSamples].label);
	}
	DestroySubgraph(&g);

	//this->predictedData = data->clone();
}

float OPFScore::getScore(int node, int cls) const 
{
	if (scores && node < scoreSize && node >= 0 && cls >= 0 && cls < scoresNumClasses)
		return scores[node][cls];
	return -1.0f;

}
void OPFScore::destroyScores()
{
	if(scores != 0 && scoreSize > 0)
	{
		for (int i = 0; i < scoreSize; i++)
		{
			if (scores[i]) 
			{
				delete[] scores[i];
				scores[i] = 0;
			}
		
		}
		delete[] scores;
		scores = 0;
		scoreSize = 0;
		scoresNumClasses = 0;
	}
}

bool OPFScore::hasScore() const 
{
	return true;
}

OPFScore * OPFScore::clone() const
{
	if(trainData != NULL){
        return new OPFScore(trainData);
    }
    return new OPFScore();
}

void OPFScore::writeScores()
{
	if (!hasScore()) return;

	ofstream filescores;
	filescores.open("scores_classification.txt");

	int p; // classes 
	int i; // instances
	for (i = 0; i < scoreSize; i++) {
		filescores << i << "\t";
		for (p = 0; p < scoresNumClasses; p++) 
		{ 
			filescores << scores[i][p] << " ";	
		}
		filescores << "\n";
	}
	filescores.close();
}
