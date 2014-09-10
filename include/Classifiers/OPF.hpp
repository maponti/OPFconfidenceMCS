#ifndef H_OPF_LIBCOMB
#define H_OPF_LIBCOMB

#ifdef __cplusplus
extern "C"{
#include "OPF.h"
}
#endif

#include <cstdio>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <cstdlib>

#include "Data.hpp"
#include "Classifier.hpp"

class OPF: public Classifier{
	protected:
        Data *trainData;
		Subgraph* subgraph;
		Subgraph* data2Subgraph(Data *data);
	
	public:
        OPF();
		OPF(Data *data);
        ~OPF();

		void train(Data *trainData);
		void predict(Data *data);
		OPF* clone() const;
};

#endif
