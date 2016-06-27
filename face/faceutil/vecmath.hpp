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
*	@file		vecmath.hpp
*	@brief		vector math function like atan2, expf
*	@author		Jianguo Li, Intel Labs China
*	copyright reserved 2010, please do not remove this head
*/

#ifndef _VEC_MATH_HPP
#define _VEC_MATH_HPP

#include <float.h>
#include <opencv/cxcore.h>

#include "det_types.hpp"

#if defined(__INTEL_COMPILER) && (__INTEL_COMPILER >= 1100)
	#include "ia32intrin.h"
#endif

	inline void icxFastAtan2(float *X, float *Y, float *angle, int n, float quanscale)
	{
		int nsse = (n/4)*4;
		int i = 0;

#if defined(__INTEL_COMPILER) && (__INTEL_COMPILER >= 1100)
		Cv32suf iabsmask; iabsmask.i = 0x7fffffff;
		__m128 eps = _mm_set1_ps((float)DBL_EPSILON), absmask = _mm_set1_ps(iabsmask.f);
		__m128 _90 = _mm_set1_ps((float)(CV_PI*0.5)), _180 = _mm_set1_ps((float)CV_PI), _360 = _mm_set1_ps((float)(CV_PI*2));
		__m128 zero = _mm_setzero_ps(), _0_28 = _mm_set1_ps(0.28f), scale4 = _mm_set1_ps(quanscale);
		for(; i <nsse; i += 4)
		{
			__m128 x4 = _mm_loadu_ps(X + i), y4 = _mm_loadu_ps(Y + i);
			__m128 xq4 = _mm_mul_ps(x4, x4), yq4 = _mm_mul_ps(y4, y4);
			__m128 xly = _mm_cmplt_ps(xq4, yq4);
			__m128 z4 = _mm_div_ps(_mm_mul_ps(x4, y4), _mm_add_ps(_mm_add_ps(_mm_max_ps(xq4, yq4),
				_mm_mul_ps(_mm_min_ps(xq4, yq4), _0_28)), eps));

			// a4 <- x < y ? 90 : 0;
			__m128 a4 = _mm_and_ps(xly, _90);
			// a4 <- (y < 0 ? 360 - a4 : a4) == ((x < y ? y < 0 ? 270 : 90) : (y < 0 ? 360 : 0))
			__m128 mask = _mm_cmplt_ps(y4, zero);
			a4 = _mm_or_ps(_mm_and_ps(_mm_sub_ps(_360, a4), mask), _mm_andnot_ps(mask, a4));
			// a4 <- (x < 0 && !(x < y) ? 180 : a4)
			mask = _mm_andnot_ps(xly, _mm_cmplt_ps(x4, zero));
			a4 = _mm_or_ps(_mm_and_ps(_180, mask), _mm_andnot_ps(mask, a4));

			// a4 <- (x < y ? a4 - z4 : a4 + z4)
			a4 = _mm_mul_ps(_mm_add_ps(_mm_xor_ps(z4, _mm_andnot_ps(absmask, xly)), a4), scale4);
			_mm_storeu_ps(angle + i, a4);
		}
#endif
		for( ; i <n; i++ )
		{
			float x = X[i], y = Y[i];
			float a, x2 = x*x, y2 = y*y;
			if( y2 <= x2 )
				a = x*y/(x2 + 0.28f*y2 + (float)DBL_EPSILON) + (float)(x < 0 ? CV_PI : y >= 0 ? 0 : CV_PI*2);
			else
				a = (float)(y >= 0 ? CV_PI*0.5 : CV_PI*1.5) - x*y/(y2 + 0.28f*x2 + (float)DBL_EPSILON);
			angle[i] = a*quanscale;
		}
	}

	inline void icxMag2(float *x, float *y, float *mag, int len, float scale = 1.0f)
	{
		int nsse = (len/4)*4;
		int i = 0;

#ifdef __SSE2__
		__m128 xmm_scl4 = _mm_set1_ps(scale);
		for( ; i <nsse; i += 4)
		{
			__m128 x0 = _mm_loadu_ps(x + i);
			__m128 y0 = _mm_loadu_ps(y + i);
			x0 = _mm_add_ps(_mm_mul_ps(x0, x0), _mm_mul_ps(y0, y0));
			x0 = _mm_mul_ps(_mm_sqrt_ps(x0), xmm_scl4);
			_mm_storeu_ps(mag + i, x0);
		}
#endif
		for( ; i < len; i++)
		{
			float x0 = x[i], y0 = y[i];
			mag[i] = sqrtf(x0*x0 + y0*y0)*scale;
		}
	}

	inline void icxExp32f(float *x, float *y, int n )
	{
		int nsse = (n/4)*4;
		int i = 0;
#if defined(__INTEL_COMPILER) && (__INTEL_COMPILER >= 1100)
		for(i=0; i<nsse; i+=4)
		{
			_mm_store_ps(y +i, _mm_exp_ps(_mm_load_ps(x +i)) );
		}
#endif
		for(; i<n; ++i)
		{
			y[i] = expf(x[i]);
		}
	}

	inline void icxAbs2(float* pSrcA, float* pSrcB, int sz)
	{
		int i=0;

#ifdef __SSE2__
		int sz_sse = (sz >> 2) << 2;

		for(i=0; i<sz_sse; i+=4)
		{
			__m128 xmm_a = _mm_and_ps(_mm_loadu_ps( (pSrcA+i) ), g_xmm_absmask);
			_mm_storeu_ps((pSrcA+i), xmm_a);

			__m128 xmm_b = _mm_and_ps(_mm_loadu_ps( (pSrcB+i) ), g_xmm_absmask);
			_mm_storeu_ps((pSrcB+i), xmm_b);
		}
#endif
		for(; i<sz; ++i)
		{
			pSrcA[i] = fabsf(pSrcA[i]);
			pSrcB[i] = fabsf(pSrcB[i]);
		}
	}

	inline void icxAbs(float* pSrcA, int sz)
	{
		int i=0;

#ifdef __SSE2__
		int sz_sse = (sz >> 2) << 2;
		for(i=0; i<sz_sse; i+=4)
		{
			__m128 xmm_a = _mm_and_ps(_mm_loadu_ps( (pSrcA+i) ), g_xmm_absmask);
			_mm_storeu_ps((pSrcA+i), xmm_a);
		}
#endif
		for(; i<sz; ++i)
		{
			pSrcA[i] = fabsf(pSrcA[i]);
		}
	}

	//////////////////////////////////////////////////////////////////////////
	// SSE code for linear-model computing
inline float icxLinearWX(tagLinearModel* logit, float* rvTest)
	{
		int i = 0;
		float betax = logit->bias;
		int d_sse = (logit->d/4)*4;

#ifdef __SSE2__
		F128 xmm_s;
		xmm_s.pack = g_xmm_zero;
		for(i=0; i<d_sse; i+=4)
		{
			__m128 xmm_w = _mm_mul_ps(_mm_load_ps(&logit->beta[i]), _mm_loadu_ps(&rvTest[i]));
			xmm_s.pack = _mm_add_ps(xmm_w, xmm_s.pack);
		}
		betax += (xmm_s.f[0] + xmm_s.f[1] + xmm_s.f[2] + xmm_s.f[3]);
#endif

		for(; i<logit->d; ++i)
			betax += (logit->beta[i] * rvTest[i]);

		return betax;
	}

	// SSE code for logit-model computing
	inline float icxLogitWX(tagLogitModel* logit, float* rvTest)
	{
		int i = 0;
		float betax = logit->bias;
		int d_sse = (logit->d/4)*4;

#ifdef __SSE2__
		F128 xmm_s;
		xmm_s.pack = g_xmm_zero;
		for(i=0; i<d_sse; i+=4)
		{
			__m128 xmm_w = _mm_mul_ps(_mm_load_ps(&logit->beta[i]), _mm_loadu_ps(&rvTest[i]));
			xmm_s.pack = _mm_add_ps(xmm_w, xmm_s.pack);
		}
		betax += (xmm_s.f[0] + xmm_s.f[1] + xmm_s.f[2] + xmm_s.f[3]);
#endif

		for(; i<logit->d; ++i)
			betax += (logit->beta[i] * rvTest[i]);

		return betax;
	}

	// direct write-back to input histogram
	// quantization using linear-scaling with peak-value constraints
	inline void quantizeHist(float* hist, int d, int quantBit, float peakval = 0, bool nonNeg = true)
	{
		if( quantBit <= 1 )
			return;

		if( nonNeg )
		{
			// for non-negative vectors
			int maxVal = (1<<quantBit);
			if( fabsf(peakval) < FLT_EPSILON )
				peakval = sqrtf(4.0f/d);

			// map 1.6*peakval => maxVal
			float scale = (1.0f*maxVal)/(1.6f*peakval);
			for(int i=0; i<d; ++i)
			{
				int val = (int)(scale * hist[i]);
				if( val < 0 ) val = 0;
				if( val > maxVal-1) val = maxVal-1;
				hist[i] = val;
			}
		}
		else
		{
			// for vectors have both positive and negative values
			int maxVal = (1<<(quantBit-1));
			if( fabsf(peakval) < FLT_EPSILON )
				peakval = 1.6f/sqrtf(1.0f*d);

			// map 1.6*peakval => maxVal
			float scale = (1.0f*maxVal)/(1.6f*peakval);
			for(int i=0; i<d; ++i)
			{
				int val = (int)(scale * hist[i]);
				if( val < -maxVal ) val = -maxVal;
				if( val > maxVal-1) val = maxVal-1;
				hist[i] = val;
			}
		}
	}

	inline float icxVectSigmoid(float* pExpval, int round)
	{
		int i = 0;
		int round_sse = (round >> 2 ) << 2;
		float prob = 0;

#ifdef __SSE2__
		F128 xmm_s;
		xmm_s.pack = g_xmm_zero;
		for(i=0; i<round_sse; i+=4)
		{
			__m128 xmm_a = _mm_load_ps(&pExpval[i]);
			xmm_s.pack = _mm_add_ps(_mm_div_ps(xmm_a, _mm_add_ps(g_xmm_one, xmm_a)), xmm_s.pack);
		}
		prob = (xmm_s.f[0] + xmm_s.f[1] + xmm_s.f[2] + xmm_s.f[3]);
#endif

		for(; i<round; ++i)
		{
			prob += pExpval[i]/(1.0f + pExpval[i]);
		}
		prob /= round;
		return prob;
	}

// for robustness, L2HYS > L1SQRT > L2 > L1
	enum normType{NORM_L1 = 0, NORM_L1SQRT, NORM_L2, NORM_L2HYS};

	inline void normalizeVector(float* hist, int len, int type = NORM_L2, float peak_threshold = 0.2f)
	{
		float sum = 0;
		int dsse = (len/4)*4;
		int i=0;

		if( type == NORM_L1 ) // L1-norm
		{
#ifdef __SSE2__
			F128 xmm_sum;
			xmm_sum.pack = _mm_set1_ps(0);
			for(i=0; i<dsse; i+=4)
			{
				xmm_sum.pack = _mm_add_ps(xmm_sum.pack, _mm_and_ps(_mm_loadu_ps(hist + i), g_xmm_absmask));
			}
			// sum = xmm_sum.f[0]+ xmm_sum.f[1]+ xmm_sum.f[2]+ xmm_sum.f[3];
			xmm_sum.pack = _mm_hadd_ps(xmm_sum.pack, xmm_sum.pack);
			sum = xmm_sum.f[0]+ xmm_sum.f[1];
#endif
			for(; i<len; ++i)
				sum += fabsf(hist[i]);
			if( sum > FLT_EPSILON )
				sum = 1.0f/sum;
			else
				sum = 0.0f;

			i=0;
#ifdef __SSE2__
			__m128 xmm_s = _mm_set1_ps(sum);
			for(i=0; i<dsse; i+=4)
			{
				_mm_storeu_ps(hist + i, _mm_mul_ps(_mm_loadu_ps(hist + i), xmm_s));
			}
#endif
			for(; i<len; ++i)
				hist[i] *= sum;
		}
		else if( type == NORM_L1SQRT ) // L1-SQRT
		{
#ifdef __SSE2__
			F128 xmm_sum;
			xmm_sum.pack = _mm_set1_ps(0);
			for(i=0; i<dsse; i+=4)
			{
				xmm_sum.pack = _mm_add_ps(xmm_sum.pack, _mm_and_ps(_mm_loadu_ps(hist + i), g_xmm_absmask));
			}
			//sum = xmm_sum.f[0]+ xmm_sum.f[1]+ xmm_sum.f[2]+ xmm_sum.f[3];
			xmm_sum.pack = _mm_hadd_ps(xmm_sum.pack, xmm_sum.pack);
			sum = xmm_sum.f[0]+ xmm_sum.f[1];
#endif
			for(; i<len; ++i)
				sum +=  fabsf(hist[i]);
			if( sum > FLT_EPSILON )
				sum = 1.0f/sum;
			else
				sum = 0.0f;

			// note: the peak_threshold for NORM_L1SQRT is better following the rule
			// peak * d = 2, peak = 2/d
			for(i = 0; i<len; ++i)
			{
				float val = sqrtf( fabsf(hist[i] * sum) );
				// also support negative-value
				if( val > peak_threshold )
					hist[i] = peak_threshold;
				else if( val < -peak_threshold )
					hist[i] = -peak_threshold;
				else 
					hist[i] = val;
			}
		}
		else if( type == NORM_L2 ) // L2-norm
		{
#ifdef __SSE2__
			F128 xmm_sum;
			xmm_sum.pack = _mm_set1_ps(0);
			for(i=0; i<dsse; i+=4)
			{
				__m128 xmm_a = _mm_loadu_ps(hist + i);
				xmm_sum.pack = _mm_add_ps(xmm_sum.pack, _mm_mul_ps(xmm_a, xmm_a) );
			}
			sum = xmm_sum.f[0]+ xmm_sum.f[1]+ xmm_sum.f[2]+ xmm_sum.f[3];
#endif
			for(; i<len; ++i)
				sum += (hist[i] * hist[i]);
			sum = sqrt(sum);
			if( sum > FLT_EPSILON )
				sum = 1.0f/sum;
			else
				sum = 0.0f;

			i = 0;
#ifdef __SSE2__
			__m128 xmm_s = _mm_set1_ps(sum);
			for(i=0; i<dsse; i+=4)
			{
				_mm_storeu_ps(hist + i, _mm_mul_ps(_mm_loadu_ps(hist + i), xmm_s));
			}
#endif
			for(; i<len; ++i)
				hist[i] *= sum;
		}
		else if( type == NORM_L2HYS ) // L2-Hys
		{
#ifdef __SSE2__
			F128 xmm_sum;
			xmm_sum.pack = _mm_set1_ps(0);
			for(i=0; i<dsse; i+=4)
			{
				__m128 xmm_a = _mm_loadu_ps(hist + i);
				xmm_sum.pack = _mm_add_ps(xmm_sum.pack, _mm_mul_ps(xmm_a, xmm_a) );
			}
			//sum = xmm_sum.f[0]+ xmm_sum.f[1]+ xmm_sum.f[2]+ xmm_sum.f[3];
			xmm_sum.pack = _mm_hadd_ps(xmm_sum.pack, xmm_sum.pack);
			sum = xmm_sum.f[0]+ xmm_sum.f[1];
#endif
			for(; i<len; ++i)
				sum += (hist[i] * hist[i]);
			sum = sqrt(sum);
			if( sum > FLT_EPSILON )
				sum = 1.0f/sum;
			else
				sum = 0.0f;

			i = 0;
#ifdef __SSE2__
			__m128 xmm_s = _mm_set1_ps(sum);
			for(i=0; i<dsse; i+=4)
			{
				_mm_storeu_ps(hist + i, _mm_mul_ps(_mm_loadu_ps(hist + i), xmm_s));
			}
#endif
			for(; i<len; ++i)
				hist[i] *= sum;

			double sum2 = 0;
			// +-sigma=68%, +-1.5sigma=86%, +-1.7sigma=90%, +-2*sigma=>95%;  +-2.5*sigma=>98.7%;  +-3*sigma=>99.7%
			// note: the peak_threshold for L2HYS_NORM is better following the rule
			// sqrt(peak^2 * d) = 2 => peak = sqrt(4/d) for histogram features
			// for feature with both positive and negative values, the peak_threshold can be a little lower
			for(i=0; i<len; ++i)
			{
				// also support negative-value
				if( hist[i] > peak_threshold )
					hist[i] = peak_threshold;
				if( hist[i] < -peak_threshold )
					hist[i] = -peak_threshold;

				sum2 += (hist[i] * hist[i]);
			}
			sum2 = sqrt(sum2);
			if( sum2 > FLT_EPSILON )
				sum2 = 1.0f/sum2;
			else
				sum2 = 0.0f;

			i = 0;
#ifdef __SSE2__
			xmm_s = _mm_set1_ps(sum2);
			for(i=0; i<dsse; i+=4)
			{
				_mm_storeu_ps(hist + i, _mm_mul_ps(_mm_loadu_ps(hist + i), xmm_s));
			}
#endif
			for(; i<len; ++i)
				hist[i] *= sum2;
		}
	}

	inline void icxScale(float* va, float denorm, int d)
	{
		int i = 0;
		int d_sse = (d/4)*4;
		if( denorm > 0 )
		{
#ifdef __SSE2__
			__m128 xmm_scl = _mm_set1_ps(denorm);
			for(i=0; i<d_sse; i+=4)
			{
				_mm_store_ps(va+i, _mm_mul_ps(_mm_load_ps(va+i), xmm_scl));
			}
#endif
			for(; i<d; ++i)
				va[i] *= denorm;
		}
		else
			normalizeVector(va, d, NORM_L2);
	}

	//////////////////////////////////////////////////////////////////////////
	// feature-difference: sqrt-L1 ~ L1  >> chord-L1 ~ Chi2 >> others
	inline void pairwise_feature_diff(float* a, float* b, float* c, int len, int type = 2)
	{
		int i = 0;
		int dsse = (len/4)*4;

		if( type == 0 )
		{
			// dot-product  or cosine
			for(i=0; i<len; ++i)
			{
				c[i] = fabsf(a[i] * b[i]);
				c[i] = sqrtf(c[i]);
			}
		}
		else if( type == 1 ) 
		{
			// L1
#ifdef __SSE2__
			for(i=0; i<dsse; i+=4)
			{
				__m128 xmm_c = _mm_sub_ps(_mm_loadu_ps(a + i), _mm_loadu_ps(b + i));
				_mm_storeu_ps(c+i, _mm_and_ps(xmm_c, g_xmm_absmask));
			}
#endif
			for(; i<len; ++i)
			{
				c[i] = fabsf(a[i] - b[i]);
			}
		}
		else if( type == 2 ) 
		{
			// L2
#ifdef __SSE2__
			for(i=0; i<dsse; i+=4)
			{
				__m128 xmm_c = _mm_sub_ps(_mm_loadu_ps(a + i), _mm_loadu_ps(b + i));
				_mm_storeu_ps(c+i, _mm_mul_ps(xmm_c, xmm_c));
			}
#endif
			for(; i<len; ++i)
			{
				c[i] = (a[i] - b[i])*(a[i] - b[i]);
			}
		}
		else if( type == 3 )
		{
			// sqrt(L1)
#ifdef __SSE2__
			for(i=0; i<dsse; i+=4)
			{
				__m128 xmm_c = _mm_sub_ps(_mm_loadu_ps(a + i), _mm_loadu_ps(b + i));
				_mm_storeu_ps(c+i, _mm_sqrt_ps(_mm_and_ps(xmm_c, g_xmm_absmask)) );
			}
#endif
			for(; i<len; ++i)
			{
				c[i] = sqrtf(fabsf(a[i] - b[i]));
			}
		}
		else
		{
			// hamming distance
			for(int i=0; i<len; ++i)
			{
				unsigned int x = (unsigned int)a[i];
				unsigned int y = (unsigned int)b[i];

				unsigned int val = x ^ y;
				// Count the number of set bits
				unsigned int count = 0;
#ifdef __SSE2__
				count = _mm_popcnt_u32(val);
#else
				while(val)
				{
					++count;
					val &= val - 1;
				}
#endif
				c[i] = count;
			}
		}
	}

	inline void cxData2Image(uchar* pDat, IplImage* pImg)
	{
		int w = pImg->width;
		int h = pImg->height;
		int nCh = pImg->nChannels;
		if( pImg->widthStep == w*nCh )
		{
			memcpy(pImg->imageData, pDat, sizeof(uchar)* w*h*nCh);
		}
		else
		{
			for(int y=0; y<h; ++y)
			{
				uchar* pImgPtr = (uchar*)(pImg->imageData + pImg->widthStep * y);
				memcpy(pImgPtr, pDat + y *w*nCh, sizeof(uchar)* w*nCh);
			}
		}
	}

	inline void cxImage2Data(IplImage* pImg, uchar* pDat)
	{
		int w = pImg->width;
		int h = pImg->height;
		int nCh = pImg->nChannels;

		if( pImg->widthStep == w*nCh )
		{
			memcpy(pDat, pImg->imageData,  sizeof(uchar)* w*h*nCh);
		}
		else
		{
			for(int y=0; y<h; ++y)
			{
				uchar* pImgPtr = (uchar*)(pImg->imageData + pImg->widthStep * y);
				memcpy(pDat + y *w*nCh, pImgPtr, sizeof(uchar)* w*nCh);
			}
		}
	}

#endif