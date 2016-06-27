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

#ifndef _CAS_BOOST_HPP
#define _CAS_BOOST_HPP

#include <vector>
#include <string>

#include "det_types.hpp"
#include "cxslidewinfea.hpp"
#include "cxrecognizerbase.hpp"

// boost training for global features
class CxBaseReg
{
public: 
	CxBaseReg():m_linearModel(NULL), m_rcweight(NULL)
	{
		m_verbose = 0;
		m_diff_type = 0;

		m_cur_round = 0;
		m_nRound = 0;
		m_nAlgo = 0;
		m_lr_lamda = 2.0f;

		m_threshold = 0.5f;
		m_lnth = 0;
		m_imgw = 0;
		m_imgh = 0;

		m_fea_type  = 0;
		m_fea_space = 0;
		m_fea_ng  = 1;
		m_d = 0;
	};
	virtual ~CxBaseReg();

	// clear memory
	virtual void clear();

	void train(char* posfea, char*negfea, int nAlgo, int maxRound, float lamda=1.0f, int verbose = 0);

public:
	void save( const char* filename);
	int  load( const char* filename);

	int initDetector(int diff_type);
	int predict(float* aFea, float* retprob = NULL);

	CvSize getDefCutFaceSize()    { return cvSize(m_imgw, m_imgh); }
	int    getFeatureDim()        { return m_d;   }
	int    getFeatureType()       { return m_fea_type;  }
	int    getDiffType()          { return m_diff_type; }
	float  getDefThreshold()      { return m_threshold; }

private:
	// different classifiers
	tagLinearModel* m_linearModel;
	tagUsedWinGL    m_winList;
	float m_scl[MAX_FEA_DIM];
	float* m_rcweight;

	int  m_nAlgo;
	char m_prefix[64];
	int  m_cur_round;
	int  m_nRound;
	int m_nRC;

	int  m_verbose;
	float m_lr_lamda;

	// training indicators
	float m_auc, m_eer, m_tpr, m_fpr;
	float m_threshold;	// threshold for the stage
	float m_lnth;

public:
	// features
	int m_fea_type;
	int m_fea_space;
	int m_fea_ng;
	int m_diff_type;

	int m_imgw;
	int m_imgh;
	int m_d;
};

// cascade detection
class CxCasDetect : public CxRecognizerBase
{
public: 
	CxCasDetect()
	{
		m_mu        = NULL;
		m_prjmat    = NULL;

		m_exfea     = NULL;
		m_pFea      = NULL;
		m_baseReg   = NULL;
		m_pImgNorm  = NULL;

		m_diff_type = 0;
		m_imgw      = 0;
		m_imgh      = 0;
		m_d			= 0;

		m_fea_type  = 0;
		m_defRound  = 0;
		m_defThreshold = 0.978236f;
	};

	~CxCasDetect();
	
	// for load recognizer model
	virtual int    load( const char* path, const char* filename, int cutface_size = 128);
	
	// get recognizer property
	virtual CvSize getDefCutFaceSize() { return cvSize(m_imgw, m_imgh); } // recognizer's default aligned face size
	virtual int    getFeatureDim()     { return m_d/sizeof(float); }              // recognizer's total feature  dimensition
	virtual int    getFeatureType()    { return m_fea_type; }             // recognizer's feature type
	virtual float  getDefThreshold()   { return m_defThreshold; }         // default recognizer threshold 
	virtual int    getDefRound()       { return m_defRound; }             // default recognizer weak classifier number 

	// recognize
	virtual int    predict(float *pDiffFea,  float* retprob = NULL);
	virtual int    predictDiff(float *pFea1, float *pFea2, float* retprob = NULL);

	virtual int    predict(IplImage* pCutFace, float *prob = NULL){ return 0; };

	// extract feature
	virtual void   extFeature(IplImage* pCutFace, float* pFea);

private:
	// recognizer classifier
	CxBaseReg* m_baseReg;

	// recognizer feature extractor
	CxSlideWinFeature* m_exfea;
	IplImage *m_pImgNorm;

	// recognizer property
	int   m_diff_type;
	int   m_fea_type;
	int   m_fea_space;
	int   m_fea_ng;
	int   m_d;
	
	int   m_imgw;   // recognizer's default aligned face size
	int   m_imgh; 
	int   m_defRound;
	float m_defThreshold;

private:
	int m_odim;
	int m_prjdim;
	int m_nRC;
	float* m_mu;
	float* m_prjmat;	// project matrix
	float* m_pFea;
	float* m_pFeaTmp;

private:
	void load8uPrjMat(const char* modname);
};

#endif
