// HvRealHaarClassifier.h: interface for the HvRealHaarClassifier class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_HVREALHAARCLASSIFIER_H__47484A8D_AAC3_44D6_A910_C26EFA71AD69__INCLUDED_)
#define AFX_HVREALHAARCLASSIFIER_H__47484A8D_AAC3_44D6_A910_C26EFA71AD69__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include "HvTrain.h"
#include "FD_Data.h"

#include "Const.h"

class HvRealHaarClassifier  
{
public:
	bool m_bTilted;
	int m_nFeatureNum;
	int m_nRectNum;
    int m_nCompIdx;
    HvFeature m_Feature;
    HvFastFeature m_Fastfeature;
	HidFeature m_HidFeature;
	
	// real Ada-boosting
	int  m_nBinNum;
    float m_fMin;
	float m_fBinWidth_Inv;
    float m_fVal[MAX_REAL_CLASSIFIER_BIN];

	int m_nWinWidth;
	int m_nWinHeight;
	
public:
	void SetRealBinNum(int nBinNum)
	{
		if((nBinNum>0) && (nBinNum<=MAX_REAL_CLASSIFIER_BIN))
			m_nBinNum = nBinNum;
	}
	void Flip_H(HvRealHaarClassifier *SrcClassifier);
	void GetFastFeature(double dScale = 1.0);
	void GetFastFeature(int x, int y, int step,double dScale);
	float Evaluate(int* sum, int* tilted,  float fNorm);
	
	bool LoadHaarClassifier(FILE* file);
	bool SaveHaarClassifier(FILE* file);

	void SaveHaarFeature(FILE* file, int nNo);
	bool LoadHaarFeature(FILE* file, int nNo);
public:
	HvRealHaarClassifier();
	virtual ~HvRealHaarClassifier();

};

#endif // !defined(AFX_HVREALHAARCLASSIFIER_H__47484A8D_AAC3_44D6_A910_C26EFA71AD69__INCLUDED_)
