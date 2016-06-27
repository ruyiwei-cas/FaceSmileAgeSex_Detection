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

#ifndef _DE_LIGHT_HPP
#define _DE_LIGHT_HPP

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <algorithm>

#include "opencv/cv.h"
#include "opencv/highgui.h"

/*!
*	@file		delighting.hpp
*	@brief		Head file for face image delighting
*	@author		Jianguo Li, Intel Labs China
*	copyright reserved 2011, please do not remove this head
*/

// illumination normalization algorithms
// 0: no-normalize at all
// 1: hist-eq
// 2: DCT
// 3: MSR(multi-scale-retinex
// 4: DoGTT (Tan & Triggs way)
// 5: DoG (Simple DoG)

//////////////////////////////////////////////////////////////////////////
enum LIGHT_NORMALIZE_TYPE{LIGHT_NORM_NONE=0, LIGHT_NORM_HISTEQ, LIGHT_NORM_DCT, LIGHT_NORM_MSR, LIGHT_NORM_TTDOG, LIGHT_NORM_DOG};

// and the combination of them
inline float icvAvg(CvMat* pSrc)
{
	int w, h, x, y;
	w = pSrc->width;
	h = pSrc->height;

	// adjust the dynamic range to the 8-bit interval
	double ave = 0;
	for(y=0; y<h; ++y)
	{
		for(x=0; x<w; ++x)
		{
			float v = cvmGet(pSrc, y, x);
			ave += v;
		}
	}
	return (ave/(w*h));
}

inline void icvHistTrunc(CvMat* Xs, float low_per = 0.1f, float up_per = 0.1f)
{
	// scale to [0, 1]
	double minv, maxv;
	cvMinMaxLoc(Xs, &minv, &maxv);

	float invscl = (maxv - minv);
	if( invscl > FLT_EPSILON )
		invscl = 1.0f/invscl;
	else
		invscl = 0;

	int w, h, x, y;
	w = Xs->width;
	h = Xs->height;

	int m = w*h;
	std::vector<float> sv(m+2);
	sv[0] = 0;
	sv[m+1] = 1.0f;
	for(y=0; y<h; ++y)
	{
		for(x=0; x<w; ++x)
		{
			float v = cvmGet(Xs, y, x);
			v = (v - minv) * invscl;	// to the range [0, 1]		
			cvmSet(Xs, y, x, v);
			sv[y*w+x +1] = v;
		}
	}
	std::stable_sort(sv.begin()+1, sv.end()-1);
	int i1 = cvRound(m*0.01*low_per);
	int i2 = cvRound(m*(1.0f - 0.01*up_per));
	i1 = (std::max)(i1, 1);
	i2 = min(i2, m-2);
	float loth = sv[i1];
	float hith = sv[i2];
	invscl = (hith - loth);
	if( invscl > FLT_EPSILON )
		invscl = 1.0f/invscl;
	else
		invscl = 0;

	// image-adjust according to the threshold
	for(y=0; y<h; ++y)
	{
		for(x=0; x<w; ++x)
		{
			float v = cvmGet(Xs, y, x);
			if( v <= loth )
				v = 0;
			else if( v >= hith )
				v = 255;
			else v = (std::min)((std::max)(cvRound( (v - loth) * invscl * 255), 0), 255);
			cvmSet(Xs, y, x, v);
		}
	}
}

//////////////////////////////////////////////////////////////////////////
// produces the zigzag coordinates for DCT
inline void icvZigzag(CvMat* X, CvMat*& output)
{
	int h = 1;
	int v = 1;
	int vmin = 1;
	int hmin = 1;
	int vmax = X->rows;	// vert or rows
	int hmax = X->cols;	// horz or cols
	int i = 0;

	output = cvCreateMat(2, vmax*hmax, CV_16S);
	cvSetZero(output);

	// do the zigzag
	while( (v <= vmax) && (h <= hmax) )
	{
		if( (h+v)%2 == 0 )
		{
			if( v == vmin )
			{
				CV_MAT_ELEM(*output, short, 0, i) = v-1;
				CV_MAT_ELEM(*output, short, 1, i) = h-1;

				if (h == hmax)
					v = v + 1;
				else
					h = h + 1;
				i = i + 1;
			}
			else if ((h == hmax) && (v < vmax))
			{
				CV_MAT_ELEM(*output, short, 0, i) = v-1;
				CV_MAT_ELEM(*output, short, 1, i) = h-1;

				v = v + 1;
				i = i + 1;
			}
			else if ((v > vmin) && (h < hmax))
			{
				CV_MAT_ELEM(*output, short, 0, i) = v-1;
				CV_MAT_ELEM(*output, short, 1, i) = h-1;

				v = v - 1;
				h = h + 1;
				i = i + 1;
			}
		}
		else
		{
			if ((v == vmax) && (h <= hmax))
			{
				CV_MAT_ELEM(*output, short, 0, i) = v-1;
				CV_MAT_ELEM(*output, short, 1, i) = h-1;

				h = h + 1;
				i = i + 1;
			}
			else if (h == hmin)
			{
				CV_MAT_ELEM(*output, short, 0, i) = v-1;
				CV_MAT_ELEM(*output, short, 1, i) = h-1;

				if (v == vmax)
					h = h + 1;
				else
					v = v + 1;
				i = i + 1;
			}
			else if ((v < vmax) && (h > hmin))
			{
				CV_MAT_ELEM(*output, short, 0, i) = v-1;
				CV_MAT_ELEM(*output, short, 1, i) = h-1;

				v = v + 1;
				h = h - 1;
				i = i + 1;
			}
		}
		if ((v == vmax) && (h == hmax))
		{
			CV_MAT_ELEM(*output, short, 0, i) = v-1;
			CV_MAT_ELEM(*output, short, 1, i) = h-1;

			break;
		}
	}
}

// DCT based photometric normalization, good range of numb is 5~20 (5, 9, 14, 20)
inline void icvDCTLightNormalize(IplImage* pSrc, IplImage* Y, int numb = 14)
{
	/*
	The function performs photometric normalization of image X using DCT-based normalization method. 
	The technique sets a pre-defined number of DCT coefficients to 0 and hence removes some of the 
	low-frequency information in the images, which is considered to be susceptible to illumination changes.
	The function performs some kind of normalization.

	The function is intended for use in face recognition experiments.
	the default parameter is set in a way that a "good" normalization is achieved for 128 x 128 images.
	Of course the term "good" is relative and tunable.

	This function is an implementation of the paper
	W. Chen, M.J. Er, and S. Wu, ¡°Illumination Compensation and normalization
	for Robust Face Recognition Using Discrete Cosine Transform in Logarithmic Domain,¡±
	IEEE T on SMC-B, 36(2), pp. 458-466, April 2006.

	INPUTS:
	pSrc      - a grey-scale image of arbitrary size
	numb      - a scalar value determining the number of DCT coefficients to set to zero,
	default "numb=14", good range of numb is 5~20 (5, 9, 14, 20)
	OUTPUTS:
	Y         - a grey-scale image processed with the DCT-based normalization method
	*/

	// S0: Init. operations
	int w = pSrc->width;
	int h = pSrc->height;

	CvMat* X = cvCreateMat(h, w, CV_32F);
	cvConvertScale(pSrc, X);

	float means = icvAvg(X);
	means += 10; // chose a mean near the true mean (the value +10 can be changed)
	CvMat* coors = NULL;
	icvZigzag(X, coors);

	// S1: Transform to logarithm and frequency domains
	cvAddS(X, cvScalar(1.0), X);
	cvLog(X, X);

	CvMat* Dc = cvCreateMat(h, w, CV_32F);
	cvDCT(X, Dc, CV_DXT_FORWARD);

	// S2: apply the normalization
	float c_00 = logf(means) * sqrtf(w*h*1.0f);
	cvmSet(Dc, 0, 0, c_00);	

	for(int i=1; i<numb+1; ++i)
	{
		int v = CV_MAT_ELEM(*coors, short, 0, i);
		int h = CV_MAT_ELEM(*coors, short, 1, i);
		// according to the paper, set top-left part (in zig-zag way) to zero 
		float val = cvmGet(Dc, v, h);
		// val /= 10;
		val = 0;
		cvmSet(Dc, v, h, val);
	}
	// according to the paper, directly use the dct-log image for recognition, no inverse-log
	cvDCT(Dc, X, CV_DXT_INVERSE);

	// S3: Do some post-processing (or not)
	float percent = 0.1f;
	icvHistTrunc(X, percent, percent);

	// output
	assert(Y != NULL );
	cvScale(X, Y);

	cvReleaseMat(&X);
	cvReleaseMat(&Dc);
	cvReleaseMat(&coors);
}

/// creates a 2D gaussian filter
inline int icvGaussian2D(CvMat*& filter, float sigma)
{
	// fullsize = 4sigma=>95%;  5sigma=>99%;  6sigma=>99.9%
	// in this case: fullsize = 2*[2.5*sigma] + 1
	int i, j;
	int halfsize = (int)floorf(2.5f* sigma);
	int fullsize = 2*halfsize + 1;
	sigma = halfsize/2.5f;
	double c2 = 0.5f/(sigma*sigma);
	double sum = 0;

	filter = cvCreateMat(fullsize, fullsize, CV_32F);
	for(i = 0; i < fullsize; i++ ) // y
	{
		for( j = 0; j < fullsize; j++ ) // x
		{
			double x = j - halfsize;
			double y = i - halfsize;
			double val = exp(-(x*x + y*y)*c2);

			CV_MAT_ELEM(*filter, float, i, j) = val;
			sum += val;
		}
	}
	sum /= (fullsize * fullsize);
	if( sum > FLT_EPSILON )
		sum = 1.0/sum;
	else
		sum = 0;

	for(i = 0; i < fullsize; i++ ) // y
	{
		for( j = 0; j < fullsize; j++ ) // x
		{
			double val = CV_MAT_ELEM(*filter, float, i, j);
			val *= sum;

			CV_MAT_ELEM(*filter, float, i, j) = val;
		}
	}
	return halfsize;
}

//////////////////////////////////////////////////////////////////////////
// multi-scale retinex algorithm: very good algorithm
inline void icvMSRNormalize(IplImage* pSrc, IplImage* pDst, int nscale = 4) 
{
	int w = pSrc->width;
	int h = pSrc->height;

	CvMat* X = cvCreateMat(h, w, CV_32F);
	CvMat* Y = cvCreateMat(h, w, CV_32F);
	CvMat* Z = cvCreateMat(h, w, CV_32F);
	CvMat* LOGX = cvCreateMat(h, w, CV_32F);
	cvConvertScale(pSrc, X);
	cvAddS(X, cvScalar(0.001f), X);	
	cvLog(X, LOGX);
	cvSetZero(Z);

	int filterker[] = {3, 5, 9, 15};  // old-case
	float halfsize[] = {1.01f, 2.01f, 3.01f, 5.01f, 7.01f};
	for(int k=0; k<nscale; ++k)
	{
		float sigma = halfsize[k]/2.5f;
		CvMat* kernel = NULL;
		icvGaussian2D(kernel, sigma);

		cvFilter2D(X, Y, kernel);
		cvAddS(Y, cvScalar(0.001f), Y);

		cvLog(Y, Y);
		cvSub(LOGX, Y, Y);

		cvAdd(Z, Y, Z);
		cvReleaseMat(&kernel);
	}
	float percent = 0.1f;
	icvHistTrunc(Z, percent, percent);

	// output
	assert(pDst != NULL );
	cvScale(Z, pDst);

	cvReleaseMat(&X);
	cvReleaseMat(&Y);
	cvReleaseMat(&Z);
	cvReleaseMat(&LOGX);
}

//////////////////////////////////////////////////////////////////////////
// based on Tan & Triggs work in the LTP paper
inline void icvTTDoGNormalize(IplImage* pSrc, IplImage* Y) 
{
	int w = pSrc->width;
	int h = pSrc->height;

	CvMat* X = cvCreateMat(h, w, CV_32F);
	cvConvertScale(pSrc, X);

	// s1: gamma-correction
	float gamma = 0.2f;
	float gammaMap[256];
	int x, y;
	for(x=0; x<256; ++x)
		gammaMap[x] = powf(x+0.0f, gamma);

	for(y=0; y<h; ++y)
	{
		for(x=0; x<w; ++x)
		{
			int gv = (int)CV_MAT_ELEM(*X, float, y, x);
			CV_MAT_ELEM(*X, float, y, x) = gammaMap[gv];
		}
	}

	// s2: DoG filtering
	CvMat* X0 = cvCreateMat(h, w, CV_32F);
	CvMat* X1 = cvCreateMat(h, w, CV_32F);

	if( 0 )
	{
		// OpenCV's cvSmooth is very poor in accuracy

		// sigma = 1.0 => filter kernel size 5
		cvSmooth(X, X0, CV_GAUSSIAN, 5, 5);
		// sigma = 2.0 => filter kernel size 11
		cvSmooth(X, X1, CV_GAUSSIAN, 11, 11);
	}
	else
	{
		float sigma1 = 1.0f;
		CvMat* ker1 = NULL;
		icvGaussian2D(ker1, sigma1);
		cvFilter2D(X, X0, ker1);

		float sigma2 = 2.0f;
		CvMat* ker2 = NULL;
		icvGaussian2D(ker2, sigma2);
		cvFilter2D(X, X1, ker2);

		cvReleaseMat(&ker1);
		cvReleaseMat(&ker2);
	}
	cvSub(X0, X1, X);

	// S3: contrast normalization
	float a = 0.1f;
	float trim = 10.0f;
	// first pass
	cvAbs(X, X0);
	cvPow(X0, X1, a);
	float avg = icvAvg(X1);
	float scl1 = powf(avg, 1.0f/a);
	if( scl1 > FLT_EPSILON )
		cvScale(X, X, 1.0/scl1);
	else
		cvSetZero(X);

	// second pass
	cvAbs(X, X0);
	double sum = 0;
	for(y=0; y<h; ++y)
	{
		for(x=0; x<w; ++x)
		{
			float v = CV_MAT_ELEM(*X0, float, y, x);
			v = powf(min(trim, v), a);
			sum += v;
		}
	}
	float scl2 = powf(sum/(w*h), 1.0f/a);
	if( scl2 > FLT_EPSILON )
		cvScale(X, X, 1.0f/scl2);
	else
		cvSetZero(X);

	// trim/squash any large outliers (e.g. specularities)
	if( 1 )
	{
		for(y=0; y<h; ++y)
		{
			for(x=0; x<w; ++x)
			{
				double v = CV_MAT_ELEM(*X, float, y, x);
				CV_MAT_ELEM(*X, float, y, x) = trim *tanhf(v/trim);
			}
		}
	}

	// S3: Do some post-processing (or not)
	float percent = 0.1f;
	icvHistTrunc(X, percent, percent);

	// output
	assert(Y != NULL );
	cvScale(X, Y);

	cvReleaseMat(&X);
	cvReleaseMat(&X0);
	cvReleaseMat(&X1);
}

inline void icvDoGNormalize(IplImage* pSrc, IplImage* Y) 
{
	int w = pSrc->width;
	int h = pSrc->height;

	CvMat* X = cvCreateMat(h, w, CV_32F);
	cvConvertScale(pSrc, X);

	// DoG filtering
	CvMat* X0 = cvCreateMat(h, w, CV_32F);
	CvMat* X1 = cvCreateMat(h, w, CV_32F);

	if( 0 )
	{
		// OpenCV's cvSmooth is poor in accuracy

		// sigma = 1.0 => filter kernel size 5
		cvSmooth(X, X0, CV_GAUSSIAN, 5, 5);
		// sigma = 2.0 => filter kernel size 11
		cvSmooth(X, X1, CV_GAUSSIAN, 11, 11);
	}
	else
	{
		float sigma1 = 1.0f;
		CvMat* ker1 = NULL;
		icvGaussian2D(ker1, sigma1);
		cvFilter2D(X, X0, ker1);

		float sigma2 = 2.0f;
		CvMat* ker2 = NULL;
		icvGaussian2D(ker2, sigma2);
		cvFilter2D(X, X1, ker2);

		cvReleaseMat(&ker1);
		cvReleaseMat(&ker2);
	}
	cvSub(X0, X1, X);

	// S3: Do some post-processing (or not)
	float percent = 0.1f;
	icvHistTrunc(X, percent, percent);

	// output
	assert(Y != NULL );
	cvScale(X, Y);

	cvReleaseMat(&X);
	cvReleaseMat(&X0);
	cvReleaseMat(&X1);
}

//////////////////////////////////////////////////////////////////////////
inline void cvLightNormalize(IplImage* pSrc, IplImage* pDst, int method)
{
	if( pSrc == NULL || pDst == NULL )
		return;

	if( method == LIGHT_NORM_HISTEQ ) // hist-eq
	{
		cvEqualizeHist(pSrc, pDst);
	}
	else if( method == LIGHT_NORM_DCT ) // DCT-normalize
	{
		icvDCTLightNormalize(pSrc, pDst);
	}
	else if( method == LIGHT_NORM_MSR ) // MSR_NORMALIZE
	{
		icvMSRNormalize(pSrc, pDst);
	}
	else if( method == LIGHT_NORM_TTDOG ) // TT-DoG-normalize
	{
		icvTTDoGNormalize(pSrc, pDst);
	}
	else if( method == LIGHT_NORM_DOG )	// Simple DoG-normalize
	{
		icvDoGNormalize(pSrc, pDst);
	}
	else // no-normalization
	{
		cvCopy(pSrc, pDst);
	}
}

#endif
