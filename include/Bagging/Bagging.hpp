#ifndef H_BAGGING_LIBCOMB
#define H_BAGGING_LIBCOMB

#include <cstdio>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <cstdlib>

#include "Classifier.hpp"
#include "Data.hpp"
#include "Combinator.hpp"

class Bagging: public Classifier{
	protected:
		Classifier **classifiers;
		int nClassifiers;
		int bootstrapSize;
		Data *trainData;
		Combinator * combine;

		Data generateBootstrap();
	        Bagging();

	public:
	        Bagging(Classifier *classifier, int nClassifiers, int bootstrapSize);
	        Bagging(Data *trainData, Classifier *classifier, int nClassifiers, int bootstrapSize);
		Bagging(Classifier **classifiers, int nClassifiers, int bootstrapSize);
		Bagging(Data *trainData, Classifier **classifiers, int nClassifiers, int bootstrapSize);
		~Bagging();

		Bagging* clone() const;

		void train(Data *trainData, Classifier **classifiers, int nClassifiers, int bootstrapSize);
		void train(Data *trainData, Classifier *classifier, int nClassifiers, int bootstrapSize);
		void train(Data *trainData);
		void predict(Data *data);

		void predict(Data *data, double *double_fault, double *q_static, double *ir_agree,
                                double *desagree, double *correlation);

		void setClassifiers(Classifier **classifiers, int nClassifiers);
		void setBootstrapSize(int bootstrapSize);

		int getBootstrapSize();

		void setCombinator(Combinator * cb);
};

#endif
