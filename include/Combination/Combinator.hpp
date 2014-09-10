#ifndef H_COMBINATOR_LIBCOMB
#define H_COMBINATOR_LIBCOMB

#include <vector>

class Data;
class Classifier;



struct Combinator
{
	virtual ~Combinator() {}
	virtual void operator() (Data * dt, int ** classification, Classifier ** classifiers, int numClassifiers);
	virtual void onTrain(Classifier ** classifiers, int numClassifiers){}
};

struct Average : public Combinator
{
	void operator() (Data * dt, int ** classification, Classifier ** classifiers, int numClassifiers);
};

class WeightedAverage : public Combinator
{
	Data * evaluationData;
	std::vector<double> weight;	

	public:
		WeightedAverage();
		WeightedAverage(const WeightedAverage & cp);
		WeightedAverage(Data * validationData);
		WeightedAverage & operator=(const WeightedAverage & other);

		~WeightedAverage();

		void operator()(Data * dt, int ** classification, Classifier ** classifiers, int numClassifiers);
		void onTrain(Classifier ** classifiers, int numClassifiers);
		void setEvaluationData(Data * evaluationData);

};

class WeightedVote : public Combinator
{
	Data * evaluationData;
	std::vector<double> weight;	

	public:
		WeightedVote();
		WeightedVote(const WeightedVote & cp);
		WeightedVote(Data * validationData);
		WeightedVote & operator=(const WeightedVote & other);

		~WeightedVote();

		void operator()(Data * dt, int ** classification, Classifier ** classifiers, int numClassifiers);
		void onTrain(Classifier ** classifiers, int numClassifiers);
		void setEvaluationData(Data * evaluationData);

};




#endif