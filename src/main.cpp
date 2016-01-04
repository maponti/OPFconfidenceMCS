#include "Data.hpp"
#include "KNN.hpp"
#include "NormalBayes.hpp"
#include "OPFScore.hpp"
#include "Bagging.hpp"
#include "DecisionTree.hpp"
#include "MLP.hpp"
#include "AdaBoostM1.hpp"
#include "ConfusionMatrix.hpp"
#include "Algoritmo1.hpp"
#include "Algoritmo2.hpp"
#include "IRBagging.hpp"
#include "RBBagging.hpp"
#include "EBBagging.hpp"
#include "SBagging.hpp"
#include "IRBaggingOver.hpp"
#include "Combinator.hpp"
#include <string>
#include <cmath>

void printCM(ConfusionMatrix<> cm){
    cout << "tp = " << cm.getTP() << ", fp = " << cm.getFP() << ", tn = " << cm.getTN() << ", fn = " << cm.getFN() << endl;
    cout << "ACC = " << cm.acc() << endl;
    cout << "AUC = " << cm.auc() << endl;
}

void kFoldCrossValidation(Classifier *c, Data **train, Data **test, int k){
    ConfusionMatrix<> *cm[k];
    double avgAUC = 0, avgAcc = 0, sAUC = 0, sAcc = 0;
    double double_fault[k], avg_double_fault = 0, s_double_fault = 0;
    double q_static[k], avg_q_static = 0, s_q_static = 0;
    double ir_agree[k], avg_ir_agree = 0, s_ir_agree = 0;
    double desagree[k], avg_desagree = 0, s_desagree = 0;
    double correlation[k], avg_correlation = 0, s_correlation = 0;

    for(int i = 0; i < k; ++i){
        c->train(train[i]);
        //c->predict(test[i], &double_fault[i], &q_static[i], &ir_agree[i], &desagree[i], &correlation[i]);
        c->predict(test[i]);

        cm[i] = new ConfusionMatrix<>(test[i]);

        avgAUC += cm[i]->auc();
        avgAcc += cm[i]->acc();

        avg_double_fault += double_fault[i];
        avg_q_static += q_static[i];
        avg_ir_agree += ir_agree[i];
        avg_desagree += desagree[i];
        avg_correlation += correlation[i];

        //cout << "K = " << i << endl;
        //printCM(cm);
        //cout << endl;
    }

    avgAUC /= k;
    avgAcc /= k;

    avg_double_fault /= k;
    avg_q_static /= k;
    avg_ir_agree /= k;
    avg_desagree /= k;
    avg_correlation /= k;

    for(int i = 0; i < k; ++i){
        sAcc += pow(cm[i]->acc() - avgAcc, 2);
        sAUC += pow(cm[i]->auc() - avgAUC, 2);

        s_double_fault += pow(double_fault[i] - avg_double_fault, 2);
        s_q_static += pow(q_static[i] - avg_q_static, 2);
        s_ir_agree += pow(ir_agree[i] - avg_ir_agree, 2);
        s_desagree += pow(desagree[i] - avg_desagree, 2);
        s_correlation += pow(correlation[i] - avg_correlation, 2);
    }
    sAUC /= k;
    sAcc /= k;

    s_double_fault /= k;
    s_q_static /= k;
    s_ir_agree /= k;
    s_desagree /= k;
    s_correlation /= k;

    sAUC = sqrt(sAUC);
    sAcc = sqrt(sAcc);

    s_double_fault = sqrt(s_double_fault);
    s_q_static = sqrt(s_q_static);
    s_ir_agree = sqrt(s_ir_agree);
    s_desagree = sqrt(s_desagree);
    s_correlation = sqrt(s_correlation);

//    cout << "AVG double fault = " << avg_double_fault << " +- " << s_double_fault << endl;
//    cout << "AVG q static = " << avg_q_static << " +- " << s_q_static << endl;
//    cout << "AVG ir agree = " << avg_ir_agree << " +- " << s_ir_agree << endl;
//    cout << "AVG desagree = " << avg_desagree << " +- " << s_desagree << endl;
//    cout << "AVG correlation = " << avg_correlation << " +- " << s_correlation << endl;

    //cout << "AVG AUC = " << avgAUC << " +- " << sAUC << endl;
    //cout << "AVG Acc = " << avgAcc << " +- " << sAcc << endl;

    printf("& $%.2lf\\%% \\pm %.2lf$ ", avgAUC * 100, sAUC * 100);

//    printf("& $%.3lf \\pm %.3lf$ ", avg_ir_agree, s_ir_agree);

    for(int i = 0; i < k; ++i){
        delete cm[i];
    }
}

bool OPFToData(const std::string & opf_filename, const std::string & train, const std::string & evaluation, const std::string & test ,const std::string & saux)
{
    std::ofstream otrain(train, std::ofstream::out | std::ofstream::trunc);
    std::ofstream oeval(evaluation, std::ofstream::out | std::ofstream::trunc);
    std::ofstream otest(test, std::ofstream::out | std::ofstream::trunc);
    std::ofstream oaux(saux, std::ofstream::out | std::ofstream::trunc);
    Subgraph * opf = ReadSubgraph(const_cast<char*>(opf_filename.c_str()));
    Subgraph *aux = 0, * dtrain = 0, *deval = 0, *dtest = 0;

    opf_NormalizeFeatures(opf);
    opf_SplitSubgraph(opf, &aux, &dtest, 0.5f );
    opf_SplitSubgraph(aux, &dtrain, &deval, 0.4f );


    if(!otrain.good() || !oeval.good() || !otest.good() || !oaux.good()) return false;

    oaux << aux->nnodes << " " << aux->nfeats << " " << aux->nlabels << "\n";
    for (int i = 0; i < aux->nnodes; ++i)
    {
        for (int j = 0; j < aux->nfeats; j++)
            oaux << aux->node[i].feat[j] << " ";
        oaux << (aux->node[i].truelabel) << "\n";
    }

    otrain << dtrain->nnodes << " " << dtrain->nfeats << " " << dtrain->nlabels << "\n";
    for (int i = 0; i < dtrain->nnodes; i++)
    {
        for (int j = 0; j < dtrain->nfeats; j++)
            otrain << dtrain->node[i].feat[j] << " ";
        otrain << (dtrain->node[i].truelabel) << "\n";
    }

    oeval << deval->nnodes << " " << deval->nfeats << " " << deval->nlabels << "\n";
    for (int i = 0; i < deval->nnodes; i++)
    {
        for (int j = 0; j < deval->nfeats; j++)
            oeval << deval->node[i].feat[j] << " ";
        oeval << (deval->node[i].truelabel) << "\n";
    }

    otest << dtest->nnodes << " " << dtest->nfeats << " " << dtest->nlabels << "\n";
    for (int i = 0; i < dtest->nnodes; i++)
    {
        for (int j = 0; j < dtest->nfeats; j++)
            otest << dtest->node[i].feat[j] << " ";
        otest << (dtest->node[i].truelabel) << "\n";
    }



    DestroySubgraph(&opf);
    DestroySubgraph(&aux);
    DestroySubgraph(&dtrain);
    DestroySubgraph(&deval);
    DestroySubgraph(&dtest);
    return true;
}
/*
Reais
=============
TropicalBIC
Corel75
Wine
NTL
SpamBase
parkinsons
*CTG (anexo em opf)
*Skin (anexo em opf)

Depois rodar com EBBagging
========================
NTL
SpamBase
parkinsons
CTG
Skin*/

void extract_average(const std::vector<double> & vec, double & av, double & dp)
{
    double sum = 0.0;
    for(const double ac : vec)
    {
        sum += ac;
    }
    av = sum/vec.size();
    sum = 0.0;
    for (const double & ac : vec)
    {
        sum += (ac - av) * (ac - av);
    }
    sum = sum/(vec.size() - 1);
    dp = std::sqrt(sum);
}

int main(int argc, char **argv)
{
    
    OPFScore classifier;
    OPF opf_classifier;
    
    std::vector<std::string> bases_full{ "skin.opf" };
    std::vector<std::string> bases_EB;
//    std::vector<double> acc_opf;
    
    float percEns = atof(argv[1]);
    
    for (int b = 2; b < argc; b++) {
	  bases_EB.push_back(argv[b]);
    }

    if (!OPFToData(bases_EB[0], "temp_train.dat", "temp_eval.dat", "temp_test.dat", "temp_trainf.dat"))
              exit(EXIT_FAILURE);
	
    Data train("temp_train.dat"), eval("temp_eval.dat"), test("temp_test.dat"), trainf("temp_trainf.dat");

//    std::vector<std::string> bases_EB{"CircleGaussian5.opf","DifficultTwoAnomalies.opf","DifficultTwoNormal.opf","masses1.opf", "GreenCoverage.opf"};
//    std::vector<std::string> bases_EB{"suborbital40.opf"};

    std::vector<double> acc_opf;
    std::vector<double> acc_average;
    std::vector<double> acc_waverage;
    std::vector<double> acc_waverages;
    std::vector<double> acc_vote;
    std::vector<double> acc_wvote;

    for (const auto & base : bases_EB)
    {
        std::cout << "Running dataset " << base << " with EBBagging\n";
	
        
        for (const auto & size : {10, 50, 100, 200})
        {
            acc_opf.clear();
            acc_average.clear();
            acc_waverage.clear();
            acc_waverages.clear();
            acc_vote.clear();
            acc_wvote.clear();

            std::cout << "Ensemble size " << size << "\n";

            for (int i = 0; i < 10; i++)
            {

                if (!OPFToData(base, "temp_train.dat", "temp_eval.dat", "temp_test.dat", "temp_trainf.dat"))
                    exit(EXIT_FAILURE);
		
                Data train("temp_train.dat"), eval("temp_eval.dat"), test("temp_test.dat"), trainf("temp_trainf.dat");
                
		float bite = ((float)train.getNSamples())*percEns; 
		
		test.writeData();

		OPFScore classifier_opf;
		classifier_opf.train(&trainf);
		classifier_opf.predict(&test);
		classifier_opf.writeScores();
		ConfusionMatrix<MULTIPLE> cm2(&test);
		acc_opf.push_back(cm2.acc());
		//std::cout << "Accuracy " << cm2.acc() << "\n";
		
		WeightedAverage wa(&eval);
                //EBBagging bag_wavg(&classifier, size);
                Bagging bag_wavg(&classifier, size, bite);
                bag_wavg.setCombinator(&wa);
                bag_wavg.train(&train);
                bag_wavg.predict(&test);
                ConfusionMatrix<MULTIPLE> cm(&test);
                acc_waverage.push_back(cm.acc());
		std::cout << "-";

		WeightedAverage was(&eval);
                //EBBagging bag_wavg(&classifier, size);
                SBagging bag_wavgs(&classifier, size);
                bag_wavgs.setCombinator(&was);
                bag_wavgs.train(&train);
                bag_wavgs.predict(&test);
                ConfusionMatrix<MULTIPLE> cm6(&test);
                acc_waverages.push_back(cm6.acc());
		std::cout << "-";

		WeightedVote wv(&eval);
                //EBBagging bag_wvote(&opf_classifier, size);
                Bagging bag_wvote(&opf_classifier, size, bite);
                bag_wvote.setCombinator(&wv);
                bag_wvote.train(&train);
                bag_wvote.predict(&test);
                ConfusionMatrix<MULTIPLE> cm3(&test);
                acc_wvote.push_back(cm3.acc());
 		std::cout << "-";
               
                Average av_comb;
                //EBBagging bag_avg(&classifier, size);
                Bagging bag_avg(&classifier, size, bite);
                bag_avg.setCombinator(&av_comb);
                bag_avg.train(&train);
                bag_avg.predict(&test);
                ConfusionMatrix<MULTIPLE> cm4(&test);
                acc_average.push_back(cm4.acc());
		std::cout << "-";

                //EBBagging bag_vote(&opf_classifier, size);
                Bagging bag_vote(&opf_classifier, size, bite);
                bag_vote.train(&train);
                bag_vote.predict(&test);
                ConfusionMatrix<MULTIPLE> cm5(&test);
                acc_vote.push_back(cm5.acc());
		std::cout << "-.";
		std::cout << flush;

          }
          double avg, dp;
          extract_average(acc_opf, avg, dp);
          std::cout << "\nOPF: " << avg << "+-" << dp << "\n";
            
          extract_average(acc_vote, avg, dp);
          std::cout << "Majority voting: " << avg << "+-" << dp << "\n";

	  extract_average(acc_average, avg, dp);
          std::cout << "Average: " << avg << "+-" << dp << "\n";
	    
          extract_average(acc_wvote, avg, dp);
          std::cout << "Weighted majority voting: " << avg << "+-" << dp << std::endl;
	  
	  extract_average(acc_waverage, avg, dp);
          std::cout << "Weighted average: " << avg << "+-" << dp << "\n";

	  extract_average(acc_waverages, avg, dp);
          std::cout << "Weighted average Smote: " << avg << "+-" << dp << "\n";
	  
        }
    }


    return 0;
}
