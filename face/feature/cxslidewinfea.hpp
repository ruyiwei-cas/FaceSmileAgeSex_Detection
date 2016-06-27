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
*	@file		  slidewinfea.hpp
*	@brief		sliding feature extraction for facial attribute analysis
*	@author		Jianguo Li, Intel Labs China
*	copyright reserved 2010, please do not remove this head
*/

/** feature included
	LBP
	Dense SIFT/HoG
	Dense SURF
	Gabor
  boosting local LBP / dense interesting point feature for binary classification
**/

#ifndef _FEATUER_EXT_HPP
#define _FEATUER_EXT_HPP

#include "opencv/cv.h"

enum FEA_TYPE{
	FEA_LBP256	= 0,
	FEA_LBP59	= 1,
	FEA_HIST10	= 2,
	FEA_SIFT128	= 3,
	FEA_SURF64	= 4,
	FEA_HAAR144	= 5,
	FEA_GABOR_SIFT256= 6,
	FEA_GABOR_SURF128= 7,
	FEA_GABOR120     = 8,
	FEA_GABOR160     = 9,
	FEA_GABOR240     = 10,
	FEA_GABOR320     = 11,
	FEA_SURF128   	 = 12,
	FEA_SIFTC128   	 = 13,
	FEA_GABOR_HAAR192= 14,
	FEA_CSLBP_8U     = 15
};

typedef unsigned char	BYTE;

class CxSlideWinFeature
{
public:
	CxSlideWinFeature()
	{
		m_pData = NULL;
		m_pFDat = NULL;
		m_hLBP = NULL;
		m_nw = 0;
		m_nh = 0;

		m_fea_type = FEA_LBP256;
		m_r = 1;
		m_nFeaDim = 256;
		
		strcpy(m_szFeaName, "FEA_LBP256");
	};

	~CxSlideWinFeature()
	{
		freeMemory();
	}

	void freeMemory();

	void setFeaParam(int fea_type)
	{
		//if( m_fea_type < 2 )
		//	m_r = MIN(MAX(r, 1), 2);
		//else
		//	m_r = MAX(r, 0);

		m_r = 0;
		m_fea_type = fea_type;

		if( m_fea_type == FEA_LBP256 )	// LBPH-Full
		{
			m_r = 1;
			m_nFeaDim = 256;
			strcpy(m_szFeaName, "FEA_LBP256");
		}
		else if( m_fea_type == FEA_LBP59 )	// LBP-Uniform
		{
			m_r = 1;
			m_nFeaDim = 59;
			strcpy(m_szFeaName, "FEA_LBP59");
		}
		else if( m_fea_type == FEA_SIFT128 )	// DenseSIFT
		{
			m_nFeaDim = 128;
			strcpy(m_szFeaName, "FEA_SIFT128");
		}
		else if(m_fea_type == FEA_SURF64)
		{
			m_nFeaDim = 64;
			strcpy(m_szFeaName, "FEA_SURF64");
		}
		else if(m_fea_type == FEA_HAAR144)
		{
			m_nFeaDim = 144;
			strcpy(m_szFeaName, "FEA_HAAR144");
		}
		else if(m_fea_type == FEA_GABOR_SIFT256)
		{
			m_nFeaDim = 256;
			strcpy(m_szFeaName, "FEA_GABOR_SIFT256");
		}
		else if(m_fea_type == FEA_GABOR_SURF128)
		{
			m_nFeaDim = 128;
			strcpy(m_szFeaName, "FEA_GABOR_SURF128");
		}
		else if(m_fea_type == FEA_GABOR120)
		{
			m_nFeaDim = 120;
			strcpy(m_szFeaName, "FEA_GABOR120");
		}
		else if(m_fea_type == FEA_GABOR160)
		{
			m_nFeaDim = 160;
			strcpy(m_szFeaName, "FEA_GABOR160");
		}
		else if(m_fea_type == FEA_GABOR240)
		{
			m_nFeaDim = 240;
			strcpy(m_szFeaName, "FEA_GABOR240");
		}
		else if(m_fea_type == FEA_GABOR320)
		{
			m_nFeaDim = 320;
			strcpy(m_szFeaName, "FEA_GABOR320");
		}
		else if(m_fea_type == FEA_SURF128)
		{
			m_nFeaDim = 128;
			strcpy(m_szFeaName, "FEA_SURF128");
		}
		else if(m_fea_type == FEA_SIFTC128)
		{
			m_nFeaDim = 128;
			strcpy(m_szFeaName, "FEA_SIFT128");
		}
		else if(m_fea_type == FEA_GABOR_HAAR192)
		{
			m_nFeaDim = 192;
			strcpy(m_szFeaName, "FEA_GABOR_HAAR192");
		}
		else if(m_fea_type == FEA_CSLBP_8U)
		{
			m_nFeaDim = 128;
			m_r = 2;
			strcpy(m_szFeaName, "FEA_CSLBP_8U");
		}
		else
		{
			//printf("Unsupported or unmatched feature-type = %d!\n", m_fea_type);
			exit(0);
		}
	}
	int getFeaDim() { return m_nFeaDim; }
	int getFeaType()     { return m_fea_type; };

	int preproc(IplImage *pSrcImg);
	int extFeature(CvRect rc, float *pFeature);

private:
	int LBPHFeature(float* img, CvRect rc, const int nBin, float *pFeature);

private:
	char m_szFeaName[256];	// name of the feature
	int m_nFeaDim;			// feature dimension
	int m_fea_type;     	// feature type
	int m_r;	            // radius for LBP, or boundary for DSIFT
	int m_nw, m_nh, m_sz;	// image size
	int m_fea_ng;

private:
	uchar *m_pData;
	float *m_pFDat;	
	float *m_hLBP;
};

// create SIFT descriptor for a certain point
// src_img: single channel, IPL_DEPTH_32F format
// !!Tao, Stop use the old denseSIFT code since the sigma parameter is not adjusted by Rx and Ry.
void icvDenseSIFTFeaturesConst( IplImage*  src_img, float* feature_vec, 
						   int bin_num, int ori_bin_num, float coeff_sigma, float threshold,
						   CvPoint pt, float sigma, float angle, int Rx, int Ry );

void interp_hist_entry(float* hist, float xbin, float ybin, float obin, float mag, 
					   int nXBin = 4, int nYBin = 4, int nOBin = 8);

int icvDenseSIFTFeatures(CvMat* src_img, float* pFea, CvRect rc, bool bQuantize =false,
					   int nx =4, int ny =4, int no =8);

// dense surf feature extraction from given rect "rc" with sub-region grid "ng"
// output 64-dim feature to "pFea" must be outside allocated
int icvDenseSURF(CvMat* src_img, float* pFea, CvRect rc, int ng =4, int extend=0);


void LBP8ImageFast(CvMat* srcImg, CvMat* destImg, int r =1, int LBPtype =0);

void LBP8ImageFast(IplImage* srcImg, CvMat* destImg, int r =1, int LBPtype =0);
#endif
