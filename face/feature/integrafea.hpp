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
*	@file		IntegraFea.hpp
*	@brief		Head file for integral feature extraction
*	@author		Jianguo Li, Intel Labs China
*	copyright reserved 2010, please do not remove this head
*/
#ifndef _INTEGRAL_FEA_HPP
#define _INTEGRAL_FEA_HPP

#include <stdio.h>
#include "opencv/cv.h"
#include "det_types.hpp"

enum icxImgSpaceList{SPS_GREY_256=0, SPS_LBP_256};
enum icxFeaTypeList{HOG_8BIN=0, CSLBP_U8, SURF_8BIN, SURF_8BIN_T2, SURF_8BIN_D2};

const char gImgSpaceName[][64]={"GRAY256", "LBP256"};
const char gFeaTypeName[][64]={"HoG8", "CSLBP_U8", "SURF8", "SURF8_T2", "SURF8_D2"};

using namespace std;

class CxIntFeature
{
public:
	CxIntFeature():m_sumtab(NULL),m_gradx(NULL), m_grady(NULL),m_pImg(NULL)
	{
		m_nw = 0;
		m_nh = 0;
		m_sz = 0;
		m_reserve_sz = 0;
		m_nTab = 0;
		m_nBin = 0;

		m_fea_type = 0;
		m_fea_ng = 2;
		m_fea_space = SPS_GREY_256;
	};
	~CxIntFeature();
	void freeMemory();

	// pre-process inputing data
	int preproc(IplImage *pImg);
	int preproc(IplImage *pImg, CvRect roi);
	int preproc(uchar *pImage, int nw, int nh, int nChannels);

	void extFeature(const CvRect& rc, int feang, float *pFea);

	inline float getAbsSumGradient(const CvRect& rc);

	void setFeaParam(int fea_type, int ngrid = 2, int fea_space = SPS_GREY_256)
	{
		m_fea_type = fea_type;
		m_fea_space = MIN(MAX(fea_space, SPS_GREY_256), SPS_LBP_256);
		m_fea_ng = MAX(ngrid, 1);
		m_inv_ng = 1.0f/m_fea_ng;

		m_nBin = 8;
		m_nTab = max(m_nBin/4, 1);
		int dim = m_nBin * m_fea_ng * m_fea_ng;
		if( dim < 1 || dim > 128 )
		{
			//printf("not supported feature parameters\n");
			exit(0);
		}
	}
	int getFeaDim()
	{
		return (m_fea_ng*m_fea_ng * m_nBin);
	}
	int getNumBin()
	{
		return m_nBin;
	}

private:
	void SURFImage(IplImage* src);
	void SURFImage(IplImage* src, CvRect roi);

private:
	int m_nw, m_nh, m_sz;	// image size
	int m_reserve_sz;		// reserve size, if size smaller than this, no reallocate
	int m_fea_type;			// feature type, maybe dsurf4, dsurf8 or dhog8
	int m_fea_space;		// feature in which image space

	unsigned int m_fea_ng;	// #grids in x/y directions	
	float m_inv_ng;

	int m_nBin;
	int m_nTab;

private:
	// may be 8bin HoG or extended surf
	F128Dat** m_sumtab;

	// gradient memory
	float* m_gradx;
	float* m_grady;
	float* m_pImg;
};

inline void icxRect2pt4(const CvRect& ptrc, int w, int& p1, int& p2, int& p3, int& p4)
{
	int xx1 = ptrc.x;
	int xx2 = ptrc.x + ptrc.width-1;
	int yy1 = ptrc.y * w;
	int yy2 = (ptrc.y +ptrc.height-1)*w;
	p1 = yy1 + xx1;
	p2 = yy1 + xx2;
	p4 = yy2 + xx1;
	p3 = yy2 + xx2;
}

inline void icxRectFeature(F128Dat* src, float* pFea, 
						   int p1, int p2, int p3, int p4)
{
#ifndef _SSE2_
	float* a1 = (src[p1].f);
	float* a2 = (src[p2].f);
	float* a3 = (src[p3].f);
	float* a4 = (src[p4].f);

	pFea[0] = (a1[0] - a2[0] + a3[0] - a4[0]);
	pFea[1] = (a1[1] - a2[1] + a3[1] - a4[1]);
	pFea[2] = (a1[2] - a2[2] + a3[2] - a4[2]);
	pFea[3] = (a1[3] - a2[3] + a3[3] - a4[3]);
#else
	_mm_store_ps(pFea, _mm_sub_ps(
		_mm_add_ps(src[p1].xmm_f, src[p3].xmm_f), 
		_mm_add_ps(src[p2].xmm_f, src[p4].xmm_f)) );
#endif
}

inline float CxIntFeature::getAbsSumGradient(const CvRect& rc)
{
	int p1, p2, p3, p4;
	icxRect2pt4(rc, m_nw, p1, p2, p3, p4);

	float CX_DECL_ALIGNED(16) pFea[32] = {0};
	icxRectFeature(m_sumtab[0], pFea, p1, p2, p3, p4);

	return (fabsf(pFea[0]) + fabsf(pFea[1]) + fabsf(pFea[2]) + fabsf(pFea[3]));
}

#endif
