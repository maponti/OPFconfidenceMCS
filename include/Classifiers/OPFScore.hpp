#ifndef H_OPFSCORE_LIBCOMB
#define H_OPFSCORE_LIBCOMB


#include "OPF.hpp"



class OPFScore : public OPF
{
	float ** scores;
	int scoreSize;
	int scoresNumClasses;

	public:
		OPFScore();
		OPFScore(Data *data);
        ~OPFScore();

		void predict(Data *data);

		float getScore(int index, int label) const;
		bool hasScore() const;
		void destroyScores();
		OPFScore * clone() const;

};

#endif