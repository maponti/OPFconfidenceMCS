#ifndef H_PREDICTOR_LIBCOMB
#define H_PREDICTOR_LIBCOMB

#include <cstdio>
#include <iostream>
#include <fstream> // file stream
#include <iomanip>
#include <string>
#include <cstdlib>

#include "Data.hpp"

class Predictor{
	public:
		virtual void predict(Data *data) = 0;
};

#endif
