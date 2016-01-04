#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
using namespace std;

void sortInsertion(float *v, int *id, int N) {
	int i, j;
	// considera que existe uma lista ordenada de elementos
	// inicialmente v[0], por isso comeco comparando v[1]
	for (j = 1; j < N; j++) {
		int ind = id[j];
		float elem = v[j];
		i = j-1; // elemento a comparar
		while ((i >= 0) && (v[i] > elem)) {
			v[i+1] = v[i];
			id[i+1] = id[i];
			i--;
		}
		v[i+1]= elem;
		id[i+1] = ind;
	}
}


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

	int count[2];
	count[0] = 0; count[1] = 0;
	for (i = 0; i < N; i++){
		fseek(f, sizeof(int), SEEK_CUR);
		fread(&Labels[i], 1, sizeof(int), f);
		//cout << Labels[i] << endl;
		for (j = 0; j < M; j++) {
			fread(&Features[i][j], sizeof(float), 1, f);
			Features[i][j]+= 1;
		}
		count[Labels[i]-1]++; 
	}

	double means[2];
	double vars[2];
	means[0] = 0, means[1] = 0;
	vars[0] = 0, vars[1] = 0;
	int ind[M];
	int *BCdiff = new int[M];
	int *BCkl = new int[M];
	
	for (j = 0; j < M; j++) {	
		BCdiff[j] = 0;
		BCkl[j] = 0;

		for (i = 0; i < N; i++) {
			means[Labels[i]-1] += Features[i][j];;
		}
		means[0] /= (double)count[0];
		means[1] /= (double)count[1];
		for (i = 0; i < N; i++) {
			double v = pow((Features[i][j] - means[Labels[i]-1]), 2);
			vars[Labels[i]-1] += v;
		}
		vars[0] = vars[0]/(double)count[0];
		vars[1] = vars[1]/(double)count[1];

		fDiff[j] = pow(means[0]-means[1],2);

		if (vars[1] == 0 || vars[0] == 0) {
			KLd[j] = pow(means[0]-means[1], 2); cout << "\nnull var\n";
		} else {
			KLd[j] = log(sqrt(vars[1])/sqrt(vars[0])) + ( (vars[0] + pow(means[0]-means[1],2)) / (2.0*vars[1]) ) - (1/2.0);
		}
		ind[j] = j;
	}

	sortInsertion(fDiff, ind, M);
	cout << "Diff" << endl;
	for (j = 0; j < M; j++) {
		if (j <= 10) cout << j << " - " << ind[j] << "\t" << fDiff[j] << "\n";
		BCdiff[ind[j]] += j+1;
	}
	for (j = 0; j < M; j++) {
		cout << ind[j] << " " << BCdiff[ind[j]] << "\n";
	}

	sortInsertion(KLd, ind, M);
	cout << "KL" << endl;
	for (j = 0; j < M; j++) {
		if (j <= 10) cout << j << " - " << ind[j] << "\t" << KLd[j] << "\n";
		BCkl[ind[j]] += j+1; //M-j;
	}
	for (j = 0; j < M; j++) {
		cout << ind[j] << " " << BCkl[ind[j]] << "\n";
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
