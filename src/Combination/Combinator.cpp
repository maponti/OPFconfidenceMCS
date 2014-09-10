#include "Combinator.hpp"
#include "Data.hpp"
#include "Classifier.hpp"
#include "ConfusionMatrix.hpp"
#include <vector>

void Combinator::operator() (Data * dt, int ** classification, Classifier ** classifiers, int numClassifiers)
{
	int i;
	int chosen;
	std::vector<int> votes(dt->getNLabels());
	for (i = 0; i < dt->getNSamples(); i++)
	{
		for (int & v : votes)
			v = 0;
		for (int j = 0; j < numClassifiers; j++)
			++votes[classification[j][i] - 1];

		chosen = 0;
		for (int j = 1; j < dt->getNLabels(); j++)
			if (votes[j] > votes[chosen])
				chosen = j;
		
		dt->setClassificationLabel(i, chosen + 1);	

	}
}

void Average::operator() (Data * dt, int ** classification, Classifier ** classifiers, int numClassifiers)
{
	int i, j, k;
	int chosen;
	std::vector<float> averages(dt->getNLabels());
	for (i = 0; i < dt->getNSamples(); i++)
	{
		for (float & av : averages)
			av = 0.0f;
		for (j = 0; j < numClassifiers; j++)
			if(classifiers[j]->hasScore())
			{
				for (k = 0; k < dt->getNLabels(); k++)
					averages[k] += classifiers[j]->getScore(i, k);
			}
		chosen = 0;
		for (j = 0; j < dt->getNLabels(); j++)
		{
			if (averages[j] > averages[chosen])
				chosen = j;
		} 
		dt->setClassificationLabel(i, chosen + 1);

	}
}

WeightedAverage::WeightedAverage()
	: evaluationData(0)
{}

WeightedAverage::WeightedAverage(const WeightedAverage & cp)
{
	this->evaluationData = cp.evaluationData->clone();
}

WeightedAverage::WeightedAverage(Data * validationData)
{
	evaluationData = validationData->clone();	
}

WeightedAverage & WeightedAverage::operator=(const WeightedAverage & other)
{
	this->evaluationData = other.evaluationData->clone();
	return *this;
}

WeightedAverage::~WeightedAverage()
{
	if(evaluationData) delete evaluationData;
}

void WeightedAverage::operator()(Data * dt, int ** classification, Classifier ** classifiers, int numClassifiers)
{
	int i, j, k;
	int chosen;
	std::vector<float> averages(dt->getNLabels(), 0.0f);
	for (i = 0; i < dt->getNSamples(); i++)
	{
		for (float & av : averages)
			av = 0.0f;
		for (j = 0; j < numClassifiers; j++)
		{
			for (k = 0; k < dt->getNLabels(); k++)
			{
				if(classifiers[j]->hasScore())
				{
					averages[k] += classifiers[j]->getScore(i, k) * weight[j];  
				}
			}
		}

		chosen = 0;
		for (j = 0; j < dt->getNLabels(); j++)
		{
			if (averages[j] > averages[chosen])
				chosen = j;
		}
		dt->setClassificationLabel(i, chosen + 1);

	}
}

void WeightedAverage::onTrain(Classifier ** classifiers, int numClassifiers)
{
	if (!evaluationData) return;
	weight.clear();
	for (int i = 0; i < numClassifiers; i++)
	{
		double acc = 1.0f;
		// encontra a acuracia
		classifiers[i]->predict(evaluationData);
		ConfusionMatrix<MULTIPLE> cm(evaluationData);
		acc = cm.acc();

		weight.push_back(acc);
	}
}

void WeightedAverage::setEvaluationData(Data * evaluationData)
{
	if (this->evaluationData) delete this->evaluationData;
	this->evaluationData = evaluationData->clone();
}

WeightedVote::WeightedVote()
	: evaluationData(0)
{}

WeightedVote::WeightedVote(const WeightedVote & cp)
{
	this->evaluationData = cp.evaluationData->clone();
}

WeightedVote::WeightedVote(Data * validationData)
{
	evaluationData = validationData->clone();	
}

WeightedVote & WeightedVote::operator=(const WeightedVote & other)
{
	this->evaluationData = other.evaluationData->clone();
	return *this;
}

WeightedVote::~WeightedVote()
{
	if(evaluationData) delete evaluationData;
}

void WeightedVote::operator()(Data * dt, int ** classification, Classifier ** classifiers, int numClassifiers)
{
	int i;
	int chosen;
	std::vector<float> votes(dt->getNLabels());
	for (i = 0; i < dt->getNSamples(); i++)
	{
		for (auto & v : votes)
			v = 0.0f;
		for (int j = 0; j < numClassifiers; j++)
			votes[classification[j][i] - 1] += weight[j];

		chosen = 0;
		for (int j = 1; j < dt->getNLabels(); j++)
			if (votes[j] > votes[chosen])
				chosen = j;
		
		dt->setClassificationLabel(i, chosen + 1);	

	}
}

void WeightedVote::onTrain(Classifier ** classifiers, int numClassifiers)
{
	if (!evaluationData) return;
	weight.clear();
	for (int i = 0; i < numClassifiers; i++)
	{
		double acc = 1.0f;
		// encontra a acuracia
		classifiers[i]->predict(evaluationData);
		ConfusionMatrix<MULTIPLE> cm(evaluationData);
		acc = cm.acc();

		weight.push_back(acc);
	}
}

void WeightedVote::setEvaluationData(Data * evaluationData)
{
	if (this->evaluationData) delete this->evaluationData;
	this->evaluationData = evaluationData->clone();
}