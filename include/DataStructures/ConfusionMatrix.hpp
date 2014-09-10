#ifndef H_CONFUSION_MATRIX_LIBCOMB
#define H_CONFUSION_MATRIX_LIBCOMB

#include <cstdio>
#include <iostream>
#include <iomanip>
#include <string>
#include <cstdlib>
#include <vector>
#include <iostream>

#include "Data.hpp"

using namespace std;

#define BINARY 0
#define MULTIPLE 1

template <int type = BINARY>
class ConfusionMatrix
{
	private:
		int sampleSize;
		std::vector<int> negatives;
		std::vector<int> number;
		std::vector<int> positives;

	public:

		ConfusionMatrix(Data *d)
		//	: positives (d->getNLabels(), 0), negatives(d->getNLabels(), 0), number(d->getNLabels(), 0), 
			: sampleSize(d->getNSamples())
		{
			negatives.reserve(d->getNLabels());
			number.reserve(d->getNLabels());
			positives.reserve(d->getNLabels());
			for (int i = 0; i < d->getNLabels(); ++i)
			{
				positives.push_back(0);
				negatives.push_back(0);
				number.push_back(0);
			}

			for (int i = 0; i < sampleSize; i++)
			{
				if (d->getClassificationLabel(i) != d->getTrueLabel(i))
				{

					negatives[d->getTrueLabel(i) -1]++;
					positives[d->getClassificationLabel(i) - 1]++;

				}
				number[d->getTrueLabel(i) - 1]++;
			}
		}

		double acc()
		{
			double sum = 0.0;
			for (int i = 0; i < positives.size(); i++)
			{
				double e1 = (double)positives[i] / (double)(sampleSize - number[i]);
				double e2 = (double)negatives[i] / (double)number[i];
				sum += e1 + e2;
			}
			return 1.0 - (sum/(double)(2*positives.size()));
		}

};

template<>
class ConfusionMatrix<BINARY>{
	private:
		int tp, fp, tn, fn;

	public:
        ConfusionMatrix(Data *d)
        {
		    /// Usar Ex
		    if(d->getNLabels() != 2){
		        cout << "ERROR!!! nLabels = " << d->getNLabels() << endl;
		        exit(EXIT_FAILURE);
		    }

		    int nSamples = d->getNSamples();
		    tp = fp = tn = fn = 0;

		    for(int i = 0; i < nSamples; ++i){
		        if(d->getClassificationLabel(i) == d->getTrueLabel(i)){
		            if(d->getClassificationLabel(i) == 1) ++tp;
		            else ++tn;
		        }else{
		            if(d->getClassificationLabel(i) == 1) ++fp;
		            else ++fn;
		        }
		    }
		}

        ConfusionMatrix(int tp, int fp, int tn, int fn)
        {
		    this->tp = tp;
		    this->fp = fp;
		    this->tn = tn;
		    this->fn = fn;
		}

        int getTP() { return tp; }
        int getTN() { return tn; }
        int getFP() { return fp; }
        int getFN() { return fn; }
        double auc()
        {
		    double tpRate = (double)tp / (double)(tp + fn);
		    double fpRate = (double)fp / (double)(fp + tn);

		    return((1 + tpRate - fpRate) / 2);
		}

        double acc()
       	{
       		return((double)(tp + tn) / (double)(tp + fn + fp + tn));
       	} 
};
#endif
