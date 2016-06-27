// HvChainCascadeClassifier.h: interface for the HvChainCascadeClassifier class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_HvChainCascadeClassifier_H__37484300_7757_4FC2_8ADE_EA487FB2BB4B__INCLUDED_)
#define AFX_HvChainCascadeClassifier_H__37484300_7757_4FC2_8ADE_EA487FB2BB4B__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include "FD_Data.h"
#include "HvTrain.h"
#include "HvChainStageClassifier.h"


#define STAGE_CART_FILE_NAME "AdaBoostCARTHaarClassifier.txt"


class HvChainCascadeClassifier  
{
public:
	int m_nStageNum;
	int m_nWinHeight, m_nWinWidth;
	HvChainStageClassifier m_StageClassifier[MAX_STAGE_NUM];

	FdSize m_realWindowSize;
	FdSize m_sumsize;
	double scale, m_invWindowArea;
	int *m_sum_image;
	int *m_tilt_image;
        double *m_sqsum_image;
        double *m_pq0, *m_pq1, *m_pq2, *m_pq3;
        int *m_p0, *m_p1, *m_p2, *m_p3;
	
public:
	bool CalMeanNorm(int nStart_x, int nStart_y, float *fmean,float *Inv_fWinNorm);

	/// for real time face detection
	void SetImagesForClassifier(int nArea_Width, int nArea_Height,
								int nImageWidth, int nImageHeight,
		                        int *sum_image, int *tilt_image,
								double *sqsum_image);
	bool Fast_Evaluate(int nStart_x, int nStart_y,									
						float mean, float Inv_fNorm,
						float *fResult);

	void SetImageWinSize(int nAreaWidth, int nAreaHeight);
	float Evaluate(int* sum, int* tilted, float normfactor);
	bool Evaluate_bool(int *sum, int *tilted, float mean, float normfactor);
	/// load a classifier
	bool LoadStageClassifier(char * sDirectoryName, int nStageNo);
	void SaveStageClassifier(char * sDirectoryName, int nStageNo);

	/// load all classifier
	bool LoadClassifier(char * sDirectoryName);
	void SaveClassifier(char * sDirectoryName);

	HvChainCascadeClassifier(int nWinWidth, int nWinHeight);
	HvChainCascadeClassifier(HvChainCascadeClassifier &);
	virtual ~HvChainCascadeClassifier();

};

#endif // !defined(AFX_HvChainCascadeClassifier_H__37484300_7757_4FC2_8ADE_EA487FB2BB4B__INCLUDED_)
