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
*	@file		det_types.hpp
*	@brief		struct, macro and inline function definition for object detection
*	@author		Jianguo Li, Intel Labs China
*	copyright reserved 2011, please do not remove this head
*/
#ifndef _DET_TYPE_HPP
#define _DET_TYPE_HPP

#include "opencv/cxcore.h"

// for min/max
#ifdef _MSC_VER
	#ifndef NOMINMAX
		#include <minmax.h>
	#endif
#else
	// for linux
	#ifndef max
		#define max(a, b) ((a)>(b) ? (a): (b))
	#endif
	#ifndef min
		#define min(a, b) ((a)<(b) ? (a): (b))
	#endif
#endif

// Squared
#ifndef SQR
	#define SQR(x) ((x)*(x))
#endif

// alignment
#ifdef __GNUC__
	#define CX_DECL_ALIGNED(x) __attribute__ ((aligned (x)))
#elif defined _MSC_VER
	#define CX_DECL_ALIGNED(x) __declspec(align(x))
#else
	#define CX_DECL_ALIGNED(x)
#endif

// SSE
#ifdef __SSE2__
	#include <xmmintrin.h> // MMX
	#include <emmintrin.h> // SSE2
	#include <pmmintrin.h> // SSE3
	#include <nmmintrin.h> // SSE4

	#define SIMD_WIDTH  4

typedef union F128{
	__m128 pack;
	float f[4];
}F128;

typedef union I128{
	__m128i pack;
	int i[4];
	CvRect rc;
}I128;

#endif

#ifdef __SSE2__
	static int CX_DECL_ALIGNED(16) v32f_absmask[] = {0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff};
	static const __m128 g_xmm_zero = _mm_setzero_ps();
	static const __m128 g_xmm_one = _mm_set1_ps(1.0f);
	static const __m128 g_xmm_quad = _mm_set1_ps(0.25f);
	static const __m128 g_xmm_01 = _mm_set1_ps(0.1f);
	const __m128 g_xmm_absmask = *(const __m128 *)(v32f_absmask);
#endif

// cache performance: F128Dat is much than F256Dat
typedef struct F128Dat
{
	union{
		float f[4];	 // 4bin hog or (dx, dy, |dx|, |dy| + du, dv, |du|, |dv|)
#ifdef __SSE2__
		__m128 xmm_f;
#endif
	};
}F128Dat;

// rect structures
typedef struct CxRect
{
	unsigned short x;
	unsigned short y;
	unsigned short width;
	unsigned short height;
}CxRect;

// multi-view output
typedef struct CvMVRect
{
	CvRect rc;		// region
	float  prob;	// probability
	int	   vid;		// view-id
	int	   angle;	// roll-angle
	int	   stage;	// which stage the rectange in

	CvMVRect():prob(0), vid(0), stage(0), angle(0)
	{};
}CvMVRect;

typedef struct CvAvgRect
{
	CvRect rc;		// region
	float  prob;	// probability
	int	   vid;		// view-id
	int	   angle;	// roll-angle
	int    stage;	// which stage the rectange in
	int	   neighbor;	// number of neighbor supports
}CvAvgRect;

/// CxRect defined in ExtSURF.h
inline CxRect cxRect(int x, int y, int width, int height )
{
	CxRect r;
	r.x = x;
	r.y = y;
	r.width = width;
	r.height = height;
	return r;
}
inline CxRect cxRect(CvRect rc)
{
	CxRect r = cxRect(rc.x, rc.y, rc.width, rc.height);
	return r;
}
inline CvRect cvRect(CxRect rc)
{
	CvRect r = cvRect(rc.x, rc.y, rc.width, rc.height);
	return r;
}

// point structures
typedef struct CxPoint
{
	short x, y;
}CxPoint;

inline CxPoint cxPoint(int x, int y)
{
	CxPoint pt;
	pt.x = x;
	pt.y = y;
	return pt;
}

typedef struct CxPoint3D
{
	short x, y, z;
}CxPoint3D;

inline CxPoint3D cxPoint3D(int x, int y, int z)
{
	CxPoint3D pt;
	pt.x = x;
	pt.y = y;
	pt.z = z;
	return pt;
}

//////////////////////////////////////////////////////////////////////////
// structure for used local-window
typedef struct tagUsedWin
{
	CvRect rc;	// window
	int d;		// dimension
	int id;		// window-id
	int pad[2];	// padding to x16 byte alignment
}tagUsedWin;

// structure for local logit-model
typedef struct tagLogitModel
{
	float beta[128];  // 2x2*4/8
	float bias;		  // bias term
	float threshold;
	int nC;
	int d;
}tagLogitModel;

//////////////////////////////////////////////////////////////////////////
#define MAX_FEA_DIM  40960

// structure for used local-window
typedef struct tagUsedWinGL
{
	CvRect rc;	// window
	int d;		// dimension
	int id;		// window-id
	int pad[2];	// padding to x16 byte alignment

	// added for face recognition
	float minv[MAX_FEA_DIM];	// min-max value in the window
	float maxv[MAX_FEA_DIM];
}tagUsedWinGL;


// structure for local linear-model
typedef struct tagLinearModel
{
	float beta[MAX_FEA_DIM];  	  // 2x2*4/8
	float sigmoid[4];		  // sigmoid coeff
	float bias;			  // bias term
	float threshold;		  // threshold for the model
	int nC;
	int d;
}tagLinearModel;

#endif
