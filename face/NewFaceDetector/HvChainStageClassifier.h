// HvChainStageClassifier.h: interface for the HvChainStageClassifier class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_HvChainStageClassifier_H__7B510B88_8CF0_45C3_AC2C_8055FEDD76E8__INCLUDED_)
#define AFX_HvChainStageClassifier_H__7B510B88_8CF0_45C3_AC2C_8055FEDD76E8__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000
#include <stdio.h>
#include "Const.h"
#include "HvRealHaarClassifier.h"

int MakeDir(char * filename);

class HvChainStageClassifier  
{
public:
	/// weak classifier number
	int m_nWeakNum;
	float m_fThreshold;
	HvRealHaarClassifier **m_RealWeakClassifier;

public:
	bool LoadStageFromRealSeq(CvSeq* seq);
	bool LoadStageClassifier(char * sFileName);
	bool SaveStageClassifier(char * sFileName);

public:
	void Flip_H(HvChainStageClassifier *SrcClassifier);
	void ReleaseWeakClassifier();
	bool CreateWeakClassifier();
	bool Evaluate(int *sum, int *tilted, 
		          float normfactor, float fPreviousScore, float *fReturnScore);
	HvChainStageClassifier();
	HvChainStageClassifier(HvChainStageClassifier & c):
		m_nWeakNum(c.m_nWeakNum),m_fThreshold(m_fThreshold)
	{
		CreateWeakClassifier();
	}
	virtual ~HvChainStageClassifier();
	bool LoadStageClassifier_FromArray(const unsigned short *Int_Array, const double * FloatArray, 
														  int*nIntRead, int *nFloatRead);
};

#endif // !defined(AFX_HvChainStageClassifier_H__7B510B88_8CF0_45C3_AC2C_8055FEDD76E8__INCLUDED_)
