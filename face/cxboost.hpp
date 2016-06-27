/**
*** Copyright (C) 1985-2011 Intel Corporation.  All rights reserved.
***
*** The information and source code contained herein is the exclusive
*** property of Intel Corporation and may not be disclosed, examined
*** or reproduced in whole or in part without explicit written authorization
*** from the company.
***
*** Embedded Application Lab, Intel Labs China.
**/

/*!
*	@file		cxboost.hpp
*	@brief		Head file for boosting recognition
*	@author		Jianguo Li, Intel Labs China
*	copyright reserved 2010, please do not remove this head
*/
#ifndef _CXBOOST_HPP
#define _CXBOOST_HPP

#include <vector>
#include <string>

#include "opencv/cv.h"
#include "fann.h"
#include "cxslidewinfea.hpp"
#include "cxrecognizerbase.hpp"

// note this file only include detector
typedef struct tagUsedWinA
{
	float minv[512];
	float maxv[512];
	float inv_diffv[512];
	int id;
	CvRect rc;	
}tagUsedWinA;

class CxBoostDetect : public CxRecognizerBase
{
public: 
	CxBoostDetect()
	{
		m_exfea = NULL;
		m_pFea  = NULL;
		m_ann   = NULL;

		m_defRound  = 0;
		m_defThreshold  = 0;
		m_nAlgo     = 0;
		m_fea_type  = 0;
		m_fea_dim   = 0;

		m_votebuff_idx = 0;
		for(int i =0; i < NVOTEBUFF; i++)
			m_votebuff_face_id[i] = -1;
	};

	~CxBoostDetect();

	// for load recognizer model
	virtual int    load( const char* path, const char* filename, int cutface_size = 64);
	
	// get recognizer property
	virtual CvSize getDefCutFaceSize() { return cvSize(m_imgw, m_imgh); } // recognizer's default aligned face size
	virtual int    getFeatureDim()     { return m_fea_dim; }              // recognizer's total feature  dimensition
	virtual int    getFeatureType()    { return m_fea_type; }             // recognizer's feature type
	virtual float  getDefThreshold()   { return m_defThreshold; }         // default recognizer threshold 
	virtual int    getDefRound()       { return m_defRound; }             // default recognizer weak classifier number 

	// recognize
	virtual int    predict(IplImage* pCutFace, float *prob = NULL);
	virtual int    predict(float *pFea,  float* retprob = NULL);
	virtual int    predictDiff(float *pFea1, float *pFea2, float* retprob = NULL);

	// extracte feature
	virtual void   extFeature(IplImage* pCutFace, float* pFea);

	//smoth recognizer output result
	int   voteLabel(int face_trackid, int label, int vote_threshold = 2, int smooth_len = 8);  

private:
	// recognizer classifier
	struct fann **m_ann;
	std::vector<tagUsedWinA> m_winList;

	// recognizer feature extractor
	CxSlideWinFeature* m_exfea;
	float* m_pFea;  // weak calssifier's fea buff

	// recognizer property
	char  m_prefix[64];
	int   m_nAlgo;
	int   m_fea_type;
	int   m_fea_dim;

	int   m_imgw;   // recognizer's default aligned face size
	int   m_imgh; 
	int   m_defRound;
	float m_defThreshold;

	// for vote label
	static const int NVOTEBUFF   		= 16;   // voting face label buff
	int		 m_votebuff_idx;
	int      m_votebuff_face_id[NVOTEBUFF];
	int      m_votebuff_face_label[NVOTEBUFF];
};

inline void icvMLP2Prob(float* prob)
{
	if( prob[0] >= prob[1] )
	{
		prob[0] = MIN(MAX(prob[0], 1e-16), 1.0 - 1e-16);
		prob[1] = 1.0 - prob[0];
		if( prob[0] < prob[1] ) //swap two values
		{
			float tmp = prob[0];
			prob[0] = prob[1];
			prob[1] = tmp;
		}
	}
	else
	{
		prob[1] = MIN(MAX(prob[1], 1e-16), 1.0 - 1e-16);
		prob[0] = 1.0 - prob[1];
		if( prob[0] > prob[1] ) //swap two values
		{
			float tmp = prob[0];
			prob[0] = prob[1];
			prob[1] = tmp;
		}
	}
}

#endif
