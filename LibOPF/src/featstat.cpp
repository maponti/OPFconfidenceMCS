#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
using namespace std;

int main (int argc, char *argv[]) {

	FILE *f = fopen(argv[1], "rb");
	int M, N;
	fread(&N, 1, sizeof(int), f);
	fseek(f, sizeof(int), SEEK_CUR);
	fread(&M, 1, sizeof(int), f);
	int i,j;

	cout << "N = "<< N << " M = " << M << endl;

	float **Features = new float*[N];
	int *Labels = new int[N];
	float *fDiff = new float[M];
	float *KLd = new float[M];
	for (i = 0; i < N; ++i) {
		Features[i] = new float[M];
	}

	for (i = 0; i < N; i++){
		fseek(f, sizeof(int), SEEK_CUR);
		fread(&Labels[i], 1, sizeof(int), f);
		cout << Labels[i] << endl;
		for (j = 0; j < M; j++) {
			fread(&Features[i][j], sizeof(float), 1, f);
		}
	}

	double means[2];
	double vars[2];
	means[0] = 0, means[1] = 0;
	vars[0] = 0, vars[1] = 0;
	int ind[M];
	for (j = 0; j < M; j++) {	
		for (i = 0; i < N; i++) {
			means[Labels[i]-1] += Features[i][j];;
		}
		for (i = 0; i < N; i++) {
			vars[Labels[i]-1] += pow((Features[i][j] - means[Labels[i-1]]), 2);
		}
		vars[0] = sqrt(vars[0]);
		vars[1] = sqrt(vars[1]);
		fDiff[j] = fabs(means[0]-means[1]);
		KLd[j] = log(vars[1]/vars[0]) + ( (pow(vars[1],2) + pow(means[0]-means[1],2)) / (2*pow(vars[1],2)) ) - (1/2);
		ind[j] = j;
	}	

	for (i = 0; i < N; ++i) {
		delete[] Features[i];
	}
	delete[] Features;
	delete[] Labels;
	delete[] fDiff; 
	delete[] KLd;

	fclose(f);

	return 0;
}
