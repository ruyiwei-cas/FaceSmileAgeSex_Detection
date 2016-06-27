/*!
*	@file		LBP.hpp
*	@brief		LBP image transform and histogram utils
*	@author		Jianguo Li, Intel Labs China
*	copyright reserved 2010, please do not remove this head
*/

#ifndef _LBPLTP_HPP
#define _LBPLTP_HPP

#include <stdio.h>
#include <math.h>

#include "opencv/cxcore.h"
#include "vecmath.hpp"

//////////////////////////////////////////////////////////////////////////
// LBP8U2 LUT for texture classification
static int gLBP8U2[256] = {0,1,2,3,4,58,5,6,7,58,58,58,8,58,9,10,11,58,58,58,58,58,58,58,12,58,58,58,13,58,14,\
		15,16,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,17,58,58,58,58,58,58,58,18,58,58,58,19,58,20,21,\
		22,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,23,\
		58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,24,58,58,58,58,58,58,58,25,58,58,58,26,58,27,28,29,30,\
		58,31,58,58,58,32,58,58,58,58,58,58,58,33,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,34,58,58,58,\
		58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,35,36,37,58,38,\
		58,58,58,39,58,58,58,58,58,58,58,40,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,41,42,43,58,44,58,\
		58,58,45,58,58,58,58,58,58,58,46,47,48,58,49,58,58,58,50,51,52,58,53,54,55,56,57};

inline int LBPMapping(int idx, int LBPtype =0)
{
	if( LBPtype == 0 )
		return idx;
	else
		return gLBP8U2[idx];
}

inline void LBP8ImageFast(CvArr* srcImg, IplImage* dstImg, int r /* =1 */, int LBPtype /* =0 */)
{
	int width, height;
	if ( CV_IS_IMAGE(srcImg) )
	{
		width = ((IplImage*)srcImg)->width;
		height = ((IplImage*)srcImg)->height;
		assert(((IplImage*)srcImg)->nChannels == 1);
	}
	else if( CV_IS_MAT(srcImg) )
	{
		width = ((CvMat*)srcImg)->width;
		height = ((CvMat*)srcImg)->height;
	}
	else
		return;

	CvMat* lbpImg = cvCreateMat( height-2*r, width-2*r, CV_8UC1 );
	CvMat* cmpImg = cvCreateMat( height-2*r, width-2*r, CV_8UC1 );

	CvRect rcROI = cvRect(r, r, width-2*r, height-2*r);
	CvMat centerROI;
	cvGetSubRect(srcImg, &centerROI, rcROI);

	// LBP image
	cvSetZero(lbpImg);
	int dx[8] = { -r,  0,  r, r,  r, 0,  -r,  -r};	
	int dy[8] = { -r, -r, -r,  0,  r,  r,  r,  0};
	int scl[8] = {1, 2, 4, 8, 16, 32, 64, 128};	
	for(int k=0; k<8; k++)
	{
		// neighbor from shift
		CvRect rcShift = rcROI;
		rcShift.x += dx[k];
		rcShift.y += dy[k];

		CvMat shiftNeighbor;
		cvGetSubRect(srcImg, &shiftNeighbor, rcShift);
		cvCmp(&shiftNeighbor, &centerROI, cmpImg, CV_CMP_GT ); // CV_CMP_GE is better

		cvAddS(lbpImg, cvScalar(scl[k]), lbpImg, cmpImg);
	}
	cvSetZero(dstImg);
	for(int y=r; y<height-r; ++y)
	{
		for(int x=r; x<width-r; ++x)
		{
			int val = CV_MAT_ELEM(*lbpImg, uchar, y-r, x-r);
			CV_IMAGE_ELEM(dstImg, uchar, y, x) = LBPMapping(val, LBPtype);
		}
	}
	cvReleaseMat(&lbpImg);
	cvReleaseMat(&cmpImg);
}

static int gCSLBP8U[16] = {0, 1, 2, 2, 5, 3, 3, 3, 4, 4, 4, 2, 5, 5, 6, 7};
inline void CSLBPImage(CvArr* srcImg, IplImage* dstImg, int r=2, int uniform=1)
{
	int width, height;
	if ( CV_IS_IMAGE(srcImg) )
	{
		width = ((IplImage*)srcImg)->width;
		height = ((IplImage*)srcImg)->height;
		assert(((IplImage*)srcImg)->nChannels == 1);
	}
	else if( CV_IS_MAT(srcImg) )
	{
		width = ((CvMat*)srcImg)->width;
		height = ((CvMat*)srcImg)->height;
	}
	else
		return;

	CvMat* cmpImg = cvCreateMat(height-2*r, width-2*r, CV_8UC1);
	CvMat* cslbpImg = cvCreateMat(height-2*r, width-2*r, CV_8UC1 );

	CvMat* pImg = cvCreateMat( height, width, CV_32F);
	cvScale(srcImg, pImg);

	cvSetZero(cslbpImg);
	int dx[8] = { r, -r, r, -r, 0, 0, -r, r};
	int dy[8] = { 0, 0, r, -r,  r, -r, r, -r};
	int scl[4] = {1, 2, 4, 8};
	for(int k=0; k<4; k++)
	{
		CvMat s1, s2;
		CvRect r1, r2;
		r1.x = r + dx[2*k+0];
		r1.y = r + dy[2*k+0];
		r2.x = r + dx[2*k+1];
		r2.y = r + dy[2*k+1];
		r1.width = width-2*r;
		r1.height = height-2*r;
		r2.width = width-2*r;
		r2.height = height-2*r;

		cvGetSubRect(pImg, &s1, r1);
		cvGetSubRect(pImg, &s2, r2);

		// I(+d) - T> I(-d)  // not allow equal here
		cvCmp(&s1, &s2, cmpImg, CV_CMP_GT );
		cvAddS(cslbpImg, cvScalar(scl[k]), cslbpImg, cmpImg);
	}
	cvSetZero(dstImg);
	if( uniform == 1 )
	{
		for(int y=r; y<height-r; ++y)
		{
			for(int x=r; x<width-r; ++x)
			{
				int val = CV_MAT_ELEM(*cslbpImg, uchar, y-r, x-r);
				CV_IMAGE_ELEM(dstImg, uchar, y, x) = gCSLBP8U[val];
			}
		}
	}
	else
	{
		for(int y=r; y<height-r; ++y)
		{
			for(int x=r; x<width-r; ++x)
			{
				int val = CV_MAT_ELEM(*cslbpImg, uchar, y-r, x-r);
				CV_IMAGE_ELEM(dstImg, uchar, y, x) = val;
			}
		}
	}
	cvReleaseMat(&cslbpImg);
	cvReleaseMat(&cmpImg);
	cvReleaseMat(&pImg);
}

inline void computLBPWeight(CvArr* srcImg, CvMat* weightImg, int r=1, int wtype=3)
{
	if( srcImg == NULL )
		return;

	int nw = 0, nh = 0;
	if ( CV_IS_IMAGE(srcImg) )
	{
		nw = ((IplImage*)srcImg)->width;
		nh = ((IplImage*)srcImg)->height;
		assert(((IplImage*)srcImg)->nChannels == 1);
	}
	else if( CV_IS_MAT(srcImg) )
	{
		nw = ((CvMat*)srcImg)->width;
		nh = ((CvMat*)srcImg)->height;
	}
	else
		return;

	CvMat* pImg = cvCreateMat(nh, nw, CV_32F);
	cvScale(srcImg, pImg);

	cvSetZero(weightImg);
	if( wtype == 0 )
	{
		// magnitude of gradient: sqrt((dx)^2 + (dy)^2)
		int x, y;
		int sz = nw * nh;

		CvMat* DX = cvCreateMat(nh, nw, CV_32F);
		CvMat* DY = cvCreateMat(nh, nw, CV_32F);
		cvSetZero(DX);
		cvSetZero(DY);

		int width_sse = ((nw-2)>>2) <<2;
		for(y=1; y<nh-1; ++y)
		{
			int row = y * nw;
			x=1;
			float* pRowy1m = pImg->data.fl + row - nw;
			float* pRowy = pImg->data.fl + row;
			float* pRowy1p = pImg->data.fl + row + nw;
			float* gx = DX->data.fl + row;
			float* gy = DY->data.fl + row;
#ifdef _SSE_
			for(x=1; x<width_sse; x+=4)
			{
				__m128 xmm_a = _mm_sub_ps(_mm_loadu_ps(pRowy +x+1), _mm_loadu_ps(pRowy +x-1));
				_mm_storeu_ps(&gx[x], xmm_a);

				__m128 xmm_b = _mm_sub_ps(_mm_loadu_ps(pRowy1p +x), _mm_loadu_ps(pRowy1m +x));
				_mm_storeu_ps(&gy[x], xmm_b);
			}
#endif
			for(; x<nw-1; x++)
			{
				gx[x] = pRowy[x+1] - pRowy[x-1];
				gy[x] = pRowy1p[x] - pRowy1m[x];
			}
		}
		icxMag2(DX->data.fl, DY->data.fl, weightImg->data.fl, sz);

		cvReleaseMat(&DX);
		cvReleaseMat(&DY);
	}
	else
	{
		// weighted by Weber function
		int dx[8] = { -r,  0,  r, r,  r, 0,  -r,  -r};
		int dy[8] = { -r, -r, -r,  0,  r,  r,  r,  0};
		for(int k=0; k<8; k++)
		{
			for(int iy=r; iy<nh-r; ++iy)
			{
				for(int ix=r; ix<nw-r; ++ix)
				{
					int x = ix + dx[k];
					int y = iy + dy[k];
					float val = CV_MAT_ELEM(*pImg, float, y, x);
					float center = CV_MAT_ELEM(*pImg, float, iy, ix);
					CV_MAT_ELEM(*weightImg, float, iy, ix) += fabsf(val-center)/8.0f;
				}
			}
		}
	}
	cvReleaseMat(&pImg);

	// normalize the weight with sum = sqrt(nw*nh)
	float sum = 1.0f;
	for(int y=r; y<nh-r; ++y)
	{
		for(int x=r; x<nw-r; ++x)
		{
			sum += CV_MAT_ELEM(*weightImg, float, y, x);
		}
	}
	// here need SIMD to accelerate
	float invsum = sqrtf(nw*nh)/sum;
	for(int y=r; y<nh-r; ++y)
	{
		for(int x=r; x<nw-r; ++x)
		{
			float v = CV_MAT_ELEM(*weightImg, float, y, x);
			CV_MAT_ELEM(*weightImg, float, y, x) = (v +0.1f)*invsum;
		}
	}
	return ;
}

//////////////////////////////////////////////////////////////////////////
inline void computeHist(CvArr* srcImg, float* hist)
{
	int nw = 0, nh = 0;
	if ( CV_IS_IMAGE(srcImg) )
	{
		nw = ((IplImage*)srcImg)->width;
		nh = ((IplImage*)srcImg)->height;
		assert(((IplImage*)srcImg)->nChannels == 1);
		assert(((IplImage*)srcImg)->depth == IPL_DEPTH_8U);
	}
	else if( CV_IS_MAT(srcImg) )
	{
		nw = ((CvMat*)srcImg)->width;
		nh = ((CvMat*)srcImg)->height;
		assert(((CvMat*)srcImg)->type == CV_32F);
	}
	else
		return;

	if ( CV_IS_IMAGE(srcImg) )
	{
		IplImage* pImg = (IplImage*)srcImg;
		for(int y=0; y<nh; ++y)
		{
			for(int x=0; x<nw; ++x)
			{
				int idx = (int)(CV_IMAGE_ELEM(pImg, uchar, y, x));
				hist[idx] += 1.0f;
			}
		}
	}
	else
	{
		CvMat* pImg = (CvMat*)srcImg;
		for(int y=0; y<nh; ++y)
		{
			for(int x=0; x<nw; ++x)
			{
				int idx = CV_MAT_ELEM(*pImg, float, y, x);
				hist[idx] += 1.0f;
			}
		}
	}
}

inline void computeHist(CvArr* srcImg, CvRect rc, float* hist)
{
	if ( CV_IS_IMAGE(srcImg) )
	{
		assert(((IplImage*)srcImg)->nChannels == 1);
		assert(((IplImage*)srcImg)->depth == IPL_DEPTH_8U);

		IplImage* pImg = (IplImage*)srcImg;
		for(int y=rc.y; y<rc.y + rc.height; y++)
		{
			for(int x=rc.x; x<rc.x + rc.width; x++)
			{
				int idx = (int)(CV_IMAGE_ELEM(pImg, uchar, y, x));
				hist[idx] += 1.0f;
			}
		}
	}
	else if( CV_IS_MAT(srcImg) )
	{
		//assert(((CvMat*)srcImg)->type == CV_32F);

		CvMat* pImg = (CvMat*)srcImg;
		for(int y=rc.y; y<rc.y + rc.height; y++)
		{
			for(int x=rc.x; x<rc.x + rc.width; x++)
			{
				int idx = (int)(CV_MAT_ELEM(*pImg, float, y, x));
				hist[idx] += 1.0f;
			}
		}
	}
	else
		return;
}

inline void computeWeightHist(CvArr* srcImg, CvMat weight, CvRect rc, float* hist)
{
	if ( CV_IS_IMAGE(srcImg) )
	{
		assert(((IplImage*)srcImg)->nChannels == 1);
		assert(((IplImage*)srcImg)->depth == IPL_DEPTH_8U);

		IplImage* pImg = (IplImage*)srcImg;
		for(int y=rc.y; y<rc.y + rc.height; y++)
		{
			for(int x=rc.x; x<rc.x + rc.width; x++)
			{
				int idx = (int)(CV_IMAGE_ELEM(pImg, uchar, y, x));
				hist[idx] += CV_MAT_ELEM(weight, float, y, x);
			}
		}
	}
	else if( CV_IS_MAT(srcImg) )
	{
		//assert(((CvMat*)srcImg)->type == CV_32F);

		CvMat* pImg = (CvMat*)srcImg;
		for(int y=rc.y; y<rc.y + rc.height; y++)
		{
			for(int x=rc.x; x<rc.x + rc.width; x++)
			{
				int idx = (int)(CV_MAT_ELEM(*pImg, float, y, x));
				hist[idx] += CV_MAT_ELEM(weight, float, y, x);
			}
		}
	}
	else
		return;
}

#endif
