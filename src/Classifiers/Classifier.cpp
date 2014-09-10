#include "Classifier.hpp"

//0: double fault
		//1: Q- statistic
		//2: Inter-rated agreement
		//3: The Disagree Measure
		//4: Correlation

//dentro da funcao combine
//o prototipos nao sao treinados entao nao cria matriz de output
void Classifier::diversidade(int **matrix_label, int numclas, int namostras, Data *d,
                                double *double_fault, double *q_static, double *ir_agree,
                                double *desagree, double *correlation){
	float **matrix_A;
	float *vetor_erros;
	float **matrix_D;
	matrix_A = (float**)calloc(numclas,sizeof(float*));//amostras classificadas corretamente entre os 2 classificadores (A)
	vetor_erros = (float *)calloc(numclas,sizeof(float));//amostras classificadas incorretamente por cada classificador (B)(C)
	matrix_D = (float**)calloc(numclas,sizeof(float*));//amostras classificadas incorretamente entre os 2 classificadores (D)
	int i,j,n;
	//float resD = 0.0f;
	//float resA = 0.0f;
	int num = 0;
	//resultados
	float result = 0.0f;//guarda o resultado da matriz
	float **div;
	div = (float **)calloc(numclas,sizeof(float*));

	//aloca as colunas da matriz
	for (i=0; i < numclas; i++){
	    matrix_A[i] = (float*)calloc(numclas,sizeof(float));
	    matrix_D[i] = (float*)calloc(numclas,sizeof(float));
	    div[i] = (float*)calloc(numclas,sizeof(float));
	}

	//acha a quantidade  de erro de cada classificador
	for(i=0; i<numclas; i++){
		for(n=0; n<namostras; n++){
			if((matrix_label[i][n] +1) != d->getTrueLabel(n)){
				vetor_erros[i] += 1.0;
			}
		}
	}

	//calcula matriz A e D
	for(i=0; i<numclas; i++){
		for(j=0; j<numclas; j++){
			if(j != i){
				for(n=0; n<namostras; n++){
					//printf("[n = %d]\n",n);
					if(matrix_label[i][n] == d->getTrueLabel(n) && matrix_label[j][n] == d->getTrueLabel(n))
						matrix_A[i][j] += 1.0;

					else if(matrix_label[i][n] != d->getTrueLabel(n) && matrix_label[j][n] != d->getTrueLabel(n))
						matrix_D[i][j] += 1.0;
				}
			}
		}
	}

	//Dividindo os elementos da matriz pelo total de amostras
	for(i=0; i<numclas; i++){
		vetor_erros[i] = vetor_erros[i] / namostras;
				for(j=0; j<numclas; j++){
					matrix_D[i][j] = matrix_D[i][j] / namostras;
					matrix_A[i][j] = matrix_A[i][j] / namostras;
				}
	}

	//tipos de diversidades
		//0: double fault
		//1: Q- statistic
		//2: Inter-rated agreement
		//3: The Disagree Measure
		//4: Correlation

		//calcular a média dessas matrizes
		//colocar eles em uma única matriz
		//usar a estatistica Q

		//0: double fault

  for(i=0; i<numclas; i++)
	  for(j=0; j<numclas; j++)
		  div[i][j] = matrix_D[i][j];

	//calculando a diagonal superior - so ta somando os diferentes...nao soma 0 e 0 etc...tem q ver se nao sao 0
	for(result = 0, num = 0, i=0; i<numclas-1; i++){
		for(j=i+1; j<numclas; j++){
			result = result + div[i][j];
			num++;
		}
	}

    (*double_fault) = result/num;

    //printf("double fault = %lf \n", (*double_fault));

  //1: Q- statistic ------------------------------------

  for(i=0; i<numclas; i++){
	  for(j=0; j<numclas; j++){
		  if(j != i){
			  for(n=0; n<namostras; n++)
				  div[i][j] = ((matrix_A[i][j]* matrix_D[i][j]) - (vetor_erros[i]*vetor_erros[j]))/ ((matrix_A[i][j]* matrix_D[i][j]) + (vetor_erros[i]*vetor_erros[j]));
		  }
	  }
  }

	//calculando a diagonal superior - so ta somando os diferentes...nao soma 0 e 0 etc...tem q ver se nao sao 0
	for(result = 0, num = 0, i=0; i<numclas-1; i++){
		for(j=i+1; j<numclas; j++){
			result = result + div[i][j];
			num++;
		}
	}

    (*q_static) = result/num;

	//printf("Q- statistic = %lf \n", (*q_static));

		//2: Inter-rated agreement ----------------

  for(i=0; i<numclas; i++){
	  for(j=0; j<numclas; j++){
		  if(j != i){
			  for(n=0; n<namostras; n++)
				  div[i][j] = (2*((matrix_A[i][j]* matrix_D[i][j])-(vetor_erros[i]*vetor_erros[j])))/(((matrix_A[i][j] + vetor_erros[j])*(vetor_erros[j]+ matrix_D[i][j]))+((matrix_A[i][j] + vetor_erros[i])*(vetor_erros[i] +matrix_D[i][j])));
		  }
	  }
  }

	//calculando a diagonal superior - so ta somando os diferentes...nao soma 0 e 0 etc...tem q ver se nao sao 0
	for(result = 0, num = 0, i=0; i<numclas-1; i++){
		for(j=i+1; j<numclas; j++){
			result = result + div[i][j];
			num++;
		}
	}

    (*ir_agree) = result/num;

  //printf("Inter-rated agreement = %lf \n", (*ir_agree));

  //3: The Disagree Measure -------------------

  for(i=0; i<numclas; i++){
	  for(j=0; j<numclas; j++){
		  if(j != i){
			  for(n=0; n<namostras; n++)
				  div[i][j] = vetor_erros[i]+vetor_erros[j];
		  }
	  }
  }

	//calculando a diagonal superior - so ta somando os diferentes...nao soma 0 e 0 etc...tem q ver se nao sao 0
	for(result = 0, num = 0, i=0; i<numclas-1; i++){
		for(j=i+1; j<numclas; j++){
			result = result + div[i][j];
			num++;
		}
	}


    (*desagree) = result/num;


  //printf("The Disagree Measure = %lf \n", (*desagree));

  //4: Correlation -----------------------------

  for(i=0; i<numclas; i++){
	  for(j=0; j<numclas; j++){
		  if(j != i){
			  for(n=0; n<namostras; n++)
				  div[i][j] = ((matrix_A[i][j]* matrix_D[i][j])-(vetor_erros[i]*vetor_erros[j]))/(sqrt((matrix_A[i][j] + vetor_erros[i])*(vetor_erros[j]+ matrix_D[i][j])*(matrix_A[i][j] + vetor_erros[j])*(vetor_erros[i]+matrix_D[i][j])));
		  }
	  }
  }

	//calculando a diagonal superior - so ta somando os diferentes...nao soma 0 e 0 etc...tem q ver se nao sao 0
	for(result = 0, num = 0, i=0; i<numclas-1; i++){
		for(j=i+1; j<numclas; j++){
			result = result + div[i][j];
			num++;
		}
	}

    (*correlation) = result/num;
	//printf("Correlation = %lf \n", (*correlation));
}


