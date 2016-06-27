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
*	@file		cxmcboost.hpp
*	@brief		Head file for multi-class boosting recognition
*	@author		Jianguo Li, Intel Labs China
*	copyright reserved 2010, please do not remove this head
*/

#ifndef _MULTICLASS_BOOST_HPP
#define _MULTICLASS_BOOST_HPP

// multi-class boosting for weaker classifier
#include <vector>
#include <string>

#include "opencv/cv.h"
#include "opencv/ml.h"

#include "cxslidewinfea.hpp"
#include "cxrecognizerbase.hpp"

typedef struct tagUsedWinB
{
	int d;		// dimension
	int id;		// window-id
	CvRect rc;	// window
	float minv[256];	// min-max value in the window
	float maxv[256];
	float invdiff[256];
}tagUsedWinB;

typedef std::vector<int> iarray;
typedef std::vector<float> rarray;
typedef std::vector<double> darray;

//////////////////////////////////////////////////////////////////////////
class CxLogit: public CvStatModel
{
public:
	CxLogit():m_beta(NULL), m_dim(0), m_nC(0)
	{
	}
	~CxLogit();

public:
	bool isLoaded()
	{
		return (m_beta != NULL );
	}
	int predict(float* rvTest, float *ret = NULL);

	virtual void clear();
	virtual void read( CvFileStorage* storage, CvFileNode* node );

public:
	int		m_dim;
	int		m_nC;
	float*  m_beta;		// weight vector beta in logit model
};

//////////////////////////////////////////////////////////////////////////
class CxMCBoostDetect : public CxRecognizerBase
{
public: 
	CxMCBoostDetect()
	{
		m_exfea = NULL;
		m_pFea  = NULL;
		m_alphaList = NULL;
		m_theLogit = NULL;

		m_defRound = 0;
		m_nAlgo = 0;
		m_nC    = 0;
		m_imgw  = 0;

		m_fea_type = -1;
		m_fea_dim  = 0;
		m_fea_space = 0;
		m_fea_ng = 1;

		
		m_votebuff_idx = 0;
		for(int i =0; i < NVOTEBUFF; i++)
		{
			m_votebuff_face_id[i] = -1;
			m_votebuff_face_label[i].resize(NVOTEAGE);
		}
	};
	virtual ~CxMCBoostDetect();

	// for load recognizer model
	virtual int    load( const char* path, const char* filename, int cutface_size = 128);
	
	// get recognizer property
	virtual CvSize getDefCutFaceSize() { return cvSize(m_imgw, m_imgh); }  // recognizer's default aligned face size
	virtual int    getFeatureDim()     { return m_fea_dim;  } // recognizer's total feature  dimensition
	virtual int    getFeatureType()    { return m_fea_type; } // recognizer's feature type
	virtual float  getDefThreshold()   { return m_defThreshold; } // default recognizer threshold 
	virtual int    getDefRound()       { return m_defRound; } // default recognizer weak classifier number 

	// recognize
	virtual int    predict(IplImage* pCutFace, float *prob = NULL);
	virtual int    predict(float *pFea,  float* retprob = NULL);
	virtual int    predictDiff(float *pFea1, float *pFea2, float* retprob = NULL);

	// extracte feature
	virtual void   extFeature(IplImage* pCutFace, float* pFea);

	int voteLabel(int face_trackid, int label); 

private:
	// recognizer classifier
	CvMat *m_alphaList;	 // optional boosting alpha
	CxLogit* m_theLogit;
	std::vector<tagUsedWinB> m_winList;

	// recognizer feature extractor
	CxSlideWinFeature* m_exfea;
	float* m_pFea;  // weak calssifier's fea buff

	int   m_nAlgo;
	char  m_prefix[64];
	int   m_fea_type;
	int   m_fea_dim;
	int   m_fea_space;
	int   m_fea_ng;

	int   m_imgw;   // recognizer's default aligned face size
	int   m_imgh; 
	int   m_defRound;
	float m_defThreshold;
	int   m_nC;

	// for vote label
	static const int NVOTEBUFF   		= 16;   // voting face label buff
	static const int NVOTEAGE   		= 4;    // voting max age number
	int		 m_votebuff_idx;
	int      m_votebuff_face_id[16];//NVOTEBUFF];
	std::vector<int> m_votebuff_face_label[16];//NVOTEBUFF];
};

#endif
