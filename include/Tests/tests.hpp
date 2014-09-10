#ifndef H_TESTS_LIBCOMB
#define H_TESTS_LIBCOMB

#include <cstdio>
#include <iostream>
#include <fstream> // file stream
#include <iomanip>
#include <string>
#include <cstdlib>
#include <cmath>

#include "Predictor.hpp"
#include "Data.hpp"

void confusionMatrix(Predictor *p, Data *d);
void confusionMatrix(Data *d);
void AUC(Predictor *p, Data *d);
void AUC(Data *d);

#endif
