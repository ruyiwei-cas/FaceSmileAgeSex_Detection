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
*	@file		IntegraFea.cpp
*	@brief		C++ implementation for integral feature extraction
*	@author		Jianguo Li, Intel Labs China
*	copyright reserved 2010, please do not remove this head
*/

#include <memory.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <opencv/highgui.h>

#include "integrafea.hpp"
#include "vecmath.hpp"

// SUM(X,Y) = sum(x<X,y<Y)I(x,y)
// based on viola-jones paper, the 2MN addition recursion algorithm
// cache-efficient but not parallel
inline void icxIntegralSURF(CvMat *srcx, CvMat *srcy, F128Dat* dst, int st=0)
{
	int i, j;
	int height = srcx->height;
	int width = srcx->width;

	// first row only
	float rsx = 0.0f, rsy = 0.0f;
	for(j=0; j<width; j++) 
	{
		rsx += srcx->data.fl[j];
		dst[j].f[st] = rsx;	// dx

		rsy += srcy->data.fl[j];
		dst[j].f[st+1] = rsy;	// dy
	}
	// remaining cells are sum above and to the left
	// recursion based implementation, the 2MN addition algorithm
	F128Dat* dstrowim = dst;
	F128Dat* dstrowi = dst + width;	
	float* srcxrowi = (srcx->data.fl + width);
	float* srcyrowi = (srcy->data.fl + width);
	for(i=1; i<height; ++i)
	{
		// recursion based implementation, the 2MN addition algorithm
		rsx = 0.0f;
		rsy = 0.0f;
		for(j=0; j<width; j++)
		{
			rsx += srcxrowi[j];
			dstrowi[j].f[st] = rsx + dstrowim[j].f[st];

			rsy += srcyrowi[j];
			dstrowi[j].f[st+1] = rsy + dstrowim[j].f[st+1];
		}
		dstrowim = dstrowi;
		dstrowi += width;
		srcxrowi += width;
		srcyrowi += width;
	}
}

inline void icxIntegralSURF(CvMat *srcx, CvMat *srcy, F128Dat* dst, CvRect roi, int st=0)
{
	int i, j;
	int height = srcx->height;
	int width = srcx->width;

	// first row only
	float rsx = 0.0f, rsy = 0.0f;
	F128Dat* dstrow = dst + roi.y * width;
	float* srcxrow = srcx->data.fl + roi.y * width;
	float* srcyrow = srcy->data.fl + roi.y * width;
	for(j=roi.x; j<MIN(roi.x + roi.width, width); j++) 
	{
		rsx += srcxrow[j];
		dstrow[j].f[st] = rsx;	// dx

		rsy += srcyrow[j];
		dstrow[j].f[st+1] = rsy;	// dy
	}
	// remaining cells are sum above and to the left
	// recursion based implementation, the 2MN addition algorithm
	F128Dat* dstrowim = dstrow;
	F128Dat* dstrowi = dstrow + width;
	float* srcxrowi = (srcxrow + width);
	float* srcyrowi = (srcyrow + width);
	for(i=roi.y+1; i<MIN(roi.y+roi.height, height); ++i)
	{
		// recursion based implementation, the 2MN addition algorithm
		rsx = 0.0f;
		rsy = 0.0f;
		for(j=roi.x; j<MIN(roi.x + roi.width, width); j++) 
		{
			rsx += srcxrowi[j];
			dstrowi[j].f[st] = rsx + dstrowim[j].f[st];

			rsy += srcyrowi[j];
			dstrowi[j].f[st+1] = rsy + dstrowim[j].f[st+1];
		}
		dstrowim = dstrowi;
		dstrowi += width;
		srcxrowi += width;
		srcyrowi += width;
	}
}

//////////////////////////////////////////////////////////////////////////
CxIntFeature::~CxIntFeature()
{
	freeMemory();
}

void CxIntFeature::freeMemory()
{
	if( m_sumtab != NULL )
	{
		for(int i=0; i<m_nTab; ++i)
			cvFree( &m_sumtab[i] );
		delete [] m_sumtab;
		m_sumtab = NULL;
	}
	cvFree(&m_gradx);
	cvFree(&m_grady);
	cvFree(&m_pImg);
}

inline void CxIntFeature::SURFImage(IplImage *src)
{
	int y;
	int width = src->width;
	int height = src->height;
	CvMat gradx = cvMat(height, width, CV_32F, m_gradx);
	CvMat grady = cvMat(height, width, CV_32F, m_grady);

	{
		//// below is another implementation without resize
		CvMat pImg = cvMat(height, width, CV_32F, m_pImg);
		cvScale(src, &pImg);
		int width_sse = ((width-2)>>2) <<2;

		for(y=1; y<height-1; y++)
		{
			int x=1, row = y * width;
			float* pRowy1m = pImg.data.fl + row - width;
			float* pRowy = pImg.data.fl + row;
			float* pRowy1p = pImg.data.fl + row + width;
			float* pGradx = gradx.data.fl + row;
			float* pGrady = grady.data.fl + row;
			for(; x<width-1; x++)
			{
				float a = (pRowy[x+1] - pRowy[x-1]);
				float b = (pRowy1p[x] - pRowy1m[x]);
				pGradx[x] = (a>=0) ? b : 0;
				pGrady[x] = (a< 0) ? b : 0;
			}
		}
		icxIntegralSURF(&gradx, &grady, m_sumtab[0], 0);
		icxAbs2(gradx.data.fl, grady.data.fl, m_sz);
		icxIntegralSURF(&gradx, &grady, m_sumtab[0], 2);

		for(y=1; y<height-1; y++)
		{
			int x=1, row = y * width;
			float* pRowy1m = pImg.data.fl + row - width;
			float* pRowy = pImg.data.fl + row;
			float* pRowy1p = pImg.data.fl + row + width;
			float* pGradx = gradx.data.fl + row;
			float* pGrady = grady.data.fl + row;
			for(; x<width-1; x++)
			{
				float a = (pRowy[x+1] - pRowy[x-1]);
				float b = (pRowy1p[x] - pRowy1m[x]);
				pGradx[x] = (b>=0) ? a : 0;
				pGrady[x] = (b< 0) ? a : 0;
			}
		}
		icxIntegralSURF(&gradx, &grady, m_sumtab[1], 0);
		icxAbs2(gradx.data.fl, grady.data.fl, m_sz);
		icxIntegralSURF(&gradx, &grady, m_sumtab[1], 2);
	}
}

inline void CxIntFeature::SURFImage(IplImage *src, CvRect roi)
{
	int y;
	int width = src->width;
	int height = src->height;
	CvMat gradx = cvMat(height, width, CV_32F, m_gradx);
	CvMat grady = cvMat(height, width, CV_32F, m_grady);

	int xstart = roi.x + 1;
	int ystart = roi.y + 1;
	int xend = roi.x + roi.width - 1;
	int yend = roi.y + roi.height - 1;

	{
		//// below is another implementation without resize
		CvMat pImg = cvMat(height, width, CV_32F, m_pImg);
		cvScale(src, &pImg);
		int width_sse = ((yend-ystart-2)>>2) <<2;

		for(y=ystart; y<yend; y++)
		{
			int x=xstart, row = y * width;
			float* pRowy1m = pImg.data.fl + row - width;
			float* pRowy = pImg.data.fl + row;
			float* pRowy1p = pImg.data.fl + row + width;
			float* pGradx = gradx.data.fl + row;
			float* pGrady = grady.data.fl + row;
			for(; x<xend; x++)
			{
				float a = (pRowy[x+1] - pRowy[x-1]);
				float b = (pRowy1p[x] - pRowy1m[x]);
				pGradx[x] = (a>=0) ? b : 0;
				pGrady[x] = (a< 0) ? b : 0;
			}
		}
		icxIntegralSURF(&gradx, &grady, m_sumtab[0], roi, 0);
		for(y=roi.y; y<MIN(roi.y +roi.height, height); y++)
		{
			for(int x=roi.x; x<MIN(roi.x +roi.width, width); ++x)
			{
				CV_MAT_ELEM(gradx, float, y, x) = fabsf(CV_MAT_ELEM(gradx, float, y, x));
				CV_MAT_ELEM(grady, float, y, x) = fabsf(CV_MAT_ELEM(grady, float, y, x));
			}
		}
		icxIntegralSURF(&gradx, &grady, m_sumtab[0], roi, 2);

		for(y=ystart; y<yend; y++)
		{
			int x=xstart, row = y * width;
			float* pRowy1m = pImg.data.fl + row - width;
			float* pRowy = pImg.data.fl + row;
			float* pRowy1p = pImg.data.fl + row + width;
			float* pGradx = gradx.data.fl + row;
			float* pGrady = grady.data.fl + row;
			for(; x<xend; x++)
			{
				float a = (pRowy[x+1] - pRowy[x-1]);
				float b = (pRowy1p[x] - pRowy1m[x]);
				pGradx[x] = (b>=0) ? a : 0;
				pGrady[x] = (b< 0) ? a : 0;
			}
		}
		icxIntegralSURF(&gradx, &grady, m_sumtab[1], roi, 0);
		for(y=roi.y; y<MIN(roi.y +roi.height, height); y++)
		{
			for(int x=roi.x; x<MIN(roi.x +roi.width, width); ++x)
			{
				CV_MAT_ELEM(gradx, float, y, x) = fabsf(CV_MAT_ELEM(gradx, float, y, x));
				CV_MAT_ELEM(grady, float, y, x) = fabsf(CV_MAT_ELEM(grady, float, y, x));
			}
		}
		icxIntegralSURF(&gradx, &grady, m_sumtab[1], roi, 2);
	}
}

int CxIntFeature::preproc(uchar *pImage, int nw, int nh, int nChannels)
{
	if( pImage == NULL )
		return -1;

	int sz = nw*nh;
	m_nw = nw;
	m_nh = nh;
	m_sz = sz;
	if( sz > m_reserve_sz )
	{
		freeMemory();

		m_reserve_sz = sz;

		m_gradx = (float*)cvAlloc(sizeof(float)* m_sz);
		m_grady = (float*)cvAlloc(sizeof(float)* m_sz);
		m_pImg = (float*)cvAlloc(sizeof(float)* (m_sz+40004) );

		m_sumtab = new F128Dat *[m_nTab];
		for(int i=0; i<m_nTab; ++i)
			m_sumtab[i] = (F128Dat *)cvAlloc(sizeof(F128Dat) * m_sz);		
	}
	memset(m_gradx, 0, sizeof(float)* m_sz);
	memset(m_grady, 0, sizeof(float)* m_sz);

	IplImage* pSrcImg = cvCreateImageHeader(cvSize(nw, nh), IPL_DEPTH_8U, nChannels);
	cxData2Image(pImage, pSrcImg);

	if( m_fea_type <= SURF_8BIN )
		SURFImage(pSrcImg);

	cvReleaseImageHeader(&pSrcImg);

	return 1;
}

int CxIntFeature::preproc(IplImage *pSrcImg)
{
	assert(pSrcImg->depth == IPL_DEPTH_8U );

	int nw = pSrcImg->width;
	int nh = pSrcImg->height;
	int sz = nw*nh;
	m_nw = nw;
	m_nh = nh;
	m_sz = sz;
	if( sz > m_reserve_sz )
	{
		freeMemory();

		m_reserve_sz = sz;

		m_gradx = (float*)cvAlloc(sizeof(float)* m_sz);
		m_grady = (float*)cvAlloc(sizeof(float)* m_sz);
		m_pImg = (float*)cvAlloc(sizeof(float)* (m_sz+40004) );

		m_sumtab = new F128Dat *[m_nTab];
		for(int i=0; i<m_nTab; ++i)
			m_sumtab[i] = (F128Dat *)cvAlloc(sizeof(F128Dat) * m_sz);
	}
	memset(m_gradx, 0, sizeof(float)* m_sz);
	memset(m_grady, 0, sizeof(float)* m_sz);

	if( m_fea_type <= SURF_8BIN )
		SURFImage(pSrcImg);

	return 1;
}

int CxIntFeature::preproc(IplImage *pSrcImg, CvRect roi)
{
	assert(pSrcImg->depth == IPL_DEPTH_8U );

	int nw = pSrcImg->width;
	int nh = pSrcImg->height;
	int sz = nw*nh;
	m_nw = nw;
	m_nh = nh;
	m_sz = sz;
	if( sz > m_reserve_sz )
	{
		freeMemory();

		m_reserve_sz = sz;
		
		m_gradx = (float*)cvAlloc(sizeof(float)* m_sz);
		m_grady = (float*)cvAlloc(sizeof(float)* m_sz);
		m_pImg = (float*)cvAlloc(sizeof(float)* (m_sz+40004) );

		m_sumtab = new F128Dat *[m_nTab];
		for(int i=0; i<m_nTab; ++i)
			m_sumtab[i] = (F128Dat *)cvAlloc(sizeof(F128Dat) * m_sz);
	}
	memset(m_gradx, 0, sizeof(float)* m_sz);
	memset(m_grady, 0, sizeof(float)* m_sz);

	if( m_fea_type <= SURF_8BIN )
		SURFImage(pSrcImg, roi);

	return 1;
}

void CxIntFeature::extFeature(const CvRect& rc, int feang, float *pFea)
{
	CvRect ptrc;
	int p1, p2, p3, p4;

	int idx = 0;
	int ng2 = feang * feang;
	int dim = ng2* m_nBin;
	if( m_fea_ng == 2 )
	{
		int rcwidth = (rc.width >> 1);
		int rcheight = (rc.height >> 1);
		int istepx = rcwidth;
		int istepy = rcheight;
		// this may be not the goodway, interpolation is the best to maintain precision
		// int i2ndhalfx = rc.width - istepx;
		// int i2ndhalfy = rc.height - istepy;
		int i2ndhalfx = rcwidth;
		int i2ndhalfy = rcheight;

		if( m_nBin == 8 ) // 8-bin
		{
			dim = ng2*8;

			ptrc = cvRect(rc.x, rc.y, istepx, istepy);
			icxRect2pt4(ptrc, m_nw, p1, p2, p3, p4);
			icxRectFeature(m_sumtab[0], pFea, p1, p2, p3, p4);
			icxRectFeature(m_sumtab[1], pFea+4, p1, p2, p3, p4);

			ptrc = cvRect(rc.x, rc.y +istepy, istepx, i2ndhalfy);
			icxRect2pt4(ptrc, m_nw, p1, p2, p3, p4);
			icxRectFeature(m_sumtab[0], pFea+8, p1, p2, p3, p4);
			icxRectFeature(m_sumtab[1], pFea+12, p1, p2, p3, p4);

			ptrc = cvRect(rc.x + istepx, rc.y, i2ndhalfx, istepy);
			icxRect2pt4(ptrc, m_nw, p1, p2, p3, p4);
			icxRectFeature(m_sumtab[0], pFea+16, p1, p2, p3, p4);
			icxRectFeature(m_sumtab[1], pFea+20, p1, p2, p3, p4);

			ptrc = cvRect(rc.x + istepx, rc.y +istepy, i2ndhalfx, i2ndhalfy);
			icxRect2pt4(ptrc, m_nw, p1, p2, p3, p4);
			icxRectFeature(m_sumtab[0], pFea+24, p1, p2, p3, p4);
			icxRectFeature(m_sumtab[1], pFea+28, p1, p2, p3, p4);
		}
		else
		{
			// m_nTab * 16
			float* pFeature = pFea;
			for(int i=0; i<m_nTab; ++i)
			{
				ptrc = cvRect(rc.x, rc.y, istepx, istepy);
				icxRect2pt4(ptrc, m_nw, p1, p2, p3, p4);
				icxRectFeature(m_sumtab[i], pFeature, p1, p2, p3, p4);

				ptrc = cvRect(rc.x, rc.y +istepy, istepx, i2ndhalfy);
				icxRect2pt4(ptrc, m_nw, p1, p2, p3, p4);
				icxRectFeature(m_sumtab[i], pFeature+4, p1, p2, p3, p4);

				ptrc = cvRect(rc.x + istepx, rc.y, i2ndhalfx, istepy);
				icxRect2pt4(ptrc, m_nw, p1, p2, p3, p4);
				icxRectFeature(m_sumtab[i], pFeature+8, p1, p2, p3, p4);

				ptrc = cvRect(rc.x + istepx, rc.y +istepy, i2ndhalfx, i2ndhalfy);
				icxRect2pt4(ptrc, m_nw, p1, p2, p3, p4);
				icxRectFeature(m_sumtab[i], pFeature+12, p1, p2, p3, p4);

				pFeature += 16;
			}
		}
	}
	else
	{
		// 1x1 grid
		int idx = 0;
		ptrc = rc;
		icxRect2pt4(ptrc, m_nw, p1, p2, p3, p4);

		// 4 * m_nTab
		for(int i=0; i<m_nTab; ++i)
		{
			icxRectFeature(m_sumtab[i], pFea + idx, p1, p2, p3, p4);
			idx += 4;
		}
		dim = m_nTab * 4;
	}
	// d = 16, 32, 64
	float peakthresh = 0.3536f; // sqrtf(4.0f/dim);	
	if( dim == 16 )
		peakthresh = 0.4f;
	else if( dim == 32 )
		peakthresh = 0.3536f;
	else if( dim == 64 )
		peakthresh = 0.25f;

	// L2HYS_NORM is generally the best
	normalizeVector(pFea, dim, NORM_L2HYS, peakthresh);

	return ;
}
