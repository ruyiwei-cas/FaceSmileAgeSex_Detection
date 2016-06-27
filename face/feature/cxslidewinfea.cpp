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

#include <memory.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <opencv/highgui.h>

#include "cxslidewinfea.hpp"
#include "cxfaceutil.hpp"

#include "vecmath.hpp"
#include "lbp.hpp"

#ifndef SIMD_WIDTH
#define SIMD_WIDTH 4
#endif


#define		SQR(x)		((x)*(x))
#define		VL_PI		3.141592653589793
#define		VL_2_PI		(2 * VL_PI)
#define		VL_PI_4		(VL_PI/4)
#define		VL_3PI_4	(3*VL_PI/4)


#ifdef __SSE2__
#include "xmmintrin.h"

//typedef union F128{
//	__m128 pack;
//	float f[4];
//}F128;

#define _mm_sqr_ps(a)       (_mm_mul_ps((a), (a)))

#ifndef __ICL
__m128 _mm_exp_ps(__m128 x)
{
	F128 y;
	y.pack = x;
	y.f[0] = exp(y.f[0]);
	y.f[1] = exp(y.f[1]);
	y.f[2] = exp(y.f[2]);
	y.f[3] = exp(y.f[3]); 
	return y.pack;
}
		
__m128 _mm_atan2_ps(__m128 y, __m128 x)
{
	F128 yy, xx;
	yy.pack = y;
	xx.pack = x;
	yy.f[0] = atan2 (yy.f[0], xx.f[0]);
	yy.f[1] = atan2 (yy.f[1], xx.f[1]);
	yy.f[2] = atan2 (yy.f[2], xx.f[2]);
	yy.f[3] = atan2 (yy.f[3], xx.f[3]); 
	return yy.pack;
}
#endif	//__ICL
#endif

#if defined(__GNUC__) && !defined(_MM_ALIGN16)
#define _MM_ALIGN16 __attribute__ ((aligned(16)))
#endif

//////////////////////////////////////////////////////////////////////////
void CxSlideWinFeature::freeMemory()
{
	if( m_pData != NULL ) cvFree( &m_pData );
	if( m_pFDat != NULL ) cvFree( &m_pFDat );
	if( m_hLBP != NULL ) cvFree( &m_hLBP );
//	m_theGabor.destroyPlan();
}

int CxSlideWinFeature::preproc(IplImage *pSrcImg)
{
	assert(pSrcImg->depth == IPL_DEPTH_8U );

	int nw = pSrcImg->width;
	int nh = pSrcImg->height;
	int sz = nw*nh;

	if( nw != m_nw || nh != m_nh || sz != m_sz )
	{
		freeMemory();

		m_nw = nw;
		m_nh = nh;
		m_sz = nw*nh;

		m_pData = (BYTE *)cvAlloc(sizeof(BYTE) * m_sz);
		m_pFDat = (float *)cvAlloc(sizeof(float) * m_sz *2);

		if( m_fea_type < 2 )
			m_hLBP = (float *)cvAlloc(sizeof(float)*(nw-2*m_r) * (nh-2*m_r));
	}
	memset(m_pFDat, 0, sizeof(float)*m_sz*2);
	
	IplImage* pImgGray = NULL;
	if( pSrcImg->nChannels > 1 )
	{
		pImgGray = cvCreateImage(cvGetSize(pSrcImg), IPL_DEPTH_8U, 1);
		cvCvtColor(pSrcImg, pImgGray, CV_BGR2GRAY);
	}
	else
		pImgGray = pSrcImg;
	
	for(int y=0; y<nh; y++)
	{
		for(int x=0; x<nw; x++)
		{
			uchar val = CV_IMAGE_ELEM(pImgGray, uchar, y, x);
			int idx = y*nw + x;
			m_pData[idx] = val;
			m_pFDat[idx] = val;
		}
	}

	//////////////////////////////////////////////////////////////////////////
	// LBP image on gray channel
	// m_LBPtype, 0=>256, 1=>59
	//if( m_fea_type == FEA_LBP59 ||  m_fea_type == FEA_LBP256 || m_fea_type == FEA_CSLBP_8U)
	if( m_fea_type == FEA_CSLBP_8U)
	{
		IplImage* dstImg = cvCreateImage(cvSize(nw, nh), IPL_DEPTH_8U, 1);
		if( m_fea_type == FEA_LBP256 || m_fea_type == FEA_LBP59 )
			LBP8ImageFast(pImgGray, dstImg, m_r, m_fea_type);
		else if( m_fea_type == FEA_CSLBP_8U )
			CSLBPImage(pImgGray, dstImg, m_r);

		for(int y=0; y<nh; ++y)
		{
			for(int x=0; x<nw; ++x)
				m_pFDat[y*nw + x] = CV_IMAGE_ELEM(dstImg, uchar, y, x);
		}
		CvMat magImg = cvMat(m_nh, m_nw, CV_32F, m_pFDat + m_sz);
		computLBPWeight(pImgGray, &magImg, m_r, 3);

		cvReleaseImage(&dstImg);
	}

	// LBP image on gray channel
	// m_LBPtype, 0=>256, 1=>59, 2=>10
	if( m_fea_type == FEA_LBP59 ||  m_fea_type == FEA_LBP256)
	{
		CvMat srcImgHead = cvMat(nh, nw, CV_32F, m_pFDat);
		CvMat destImgHead = cvMat(nh-2*m_r, nw-2*m_r, CV_32F, m_hLBP);

		LBP8ImageFast(&srcImgHead, &destImgHead, m_r, m_fea_type);
	}
	/*
	// Gabor feature init executPlan
	if( m_fea_type == FEA_GABOR120 || m_fea_type == FEA_GABOR160 ||
		m_fea_type == FEA_GABOR240 || m_fea_type == FEA_GABOR320 ||
		m_fea_type == FEA_GABOR_SIFT256 || m_fea_type == FEA_GABOR_SURF128 ||
		m_fea_type == FEA_GABOR_HAAR192 )
	{
		if(m_bGaborInit == false)
		{
			mlcLogGaborParam gabor_param;
			setLogGaborDefaultParam(gabor_param, m_nFeaDim); //default params: 4 scale, 8 orientations
			m_theGabor.createPlan(m_nw, m_nh, &gabor_param);

			m_bGaborInit = true;
		}
		
		IplImage* pImgHeader = cvCreateImageHeader(cvSize(m_nw, m_nh), IPL_DEPTH_8U, 1);
		cvSetData(pImgHeader, m_pData, m_nw*sizeof(uchar));
		m_theGabor.executPlan(pImgHeader);
		cvReleaseImageHeader(&pImgHeader);
	}
	*/
	if( pSrcImg->nChannels > 1 )
		cvReleaseImage(&pImgGray);

	return 0;
}

int CxSlideWinFeature::LBPHFeature(float* img, CvRect rc, const int nBin, float *pFeature)
{
	int r = m_r;
	float* LBPHist = (float*)pFeature;
	int nFeaSZ = nBin;
	int idx, x, y, sz;
	sz = rc.height * rc.width;
	float unitVal = 1.0f/sz;
	CvMat destImg = cvMat(m_nh-2*r, m_nw-2*r, CV_32F, img);

	memset(LBPHist, 0, sizeof(float) * nBin);
	for(y=rc.y; y<rc.y + rc.height; y++)
	{
		for(x=rc.x; x<rc.x + rc.width; x++)
		{
			idx = (int)(cvmGet(&destImg, y, x));
			LBPHist[idx] += unitVal;
		}
	}

	return 0;
}

int CxSlideWinFeature::extFeature(CvRect rc, float *pFeature)
{
	if(pFeature == NULL) return -1;
	memset(pFeature, 0, sizeof(float)*m_nFeaDim);

	if( m_fea_type < 2 ) //FEA_LBP256 or FEA_LBP59
	{
		// LBP feature
		// LBP feature
		LBPHFeature(m_hLBP, rc, m_nFeaDim, pFeature);

		//CvMat dstImg = cvMat(m_nh, m_nw, CV_32F, m_pFDat);
		//computeHist(&dstImg, rc, pFeature);
		//normalizeVector(pFeature, m_nFeaDim, NORM_L1);
	}
	else if( m_fea_type == FEA_SIFT128 )
	{
		// DenseSIFT feature/HoG feature		
		// new SIFT with variant sigma 
		CvMat src_img = cvMat( m_nh,  m_nw, CV_32FC1, m_pFDat);
		icvDenseSIFTFeatures(&src_img, pFeature, rc);
	}
	else if( m_fea_type == FEA_SIFTC128 )
	{
		// DenseSIFT feature/HoG feature		
		//old SIFT with constnat sigma 
		IplImage* pImgHeader = cvCreateImageHeader(cvSize(m_nw, m_nh), IPL_DEPTH_32F, 1);
		cvSetData(pImgHeader, m_pFDat, m_nw*sizeof(float));

		icvDenseSIFTFeaturesConst(pImgHeader, pFeature,  4, 8, 3, 0.2, 
			cvPoint(rc.x+rc.width/2, rc.y+rc.height/2),  1.5, 0, rc.width/2, rc.height/2);
		cvReleaseImageHeader(&pImgHeader);
	}
	/*
	else if(m_fea_type == FEA_GABOR120 || m_fea_type == FEA_GABOR160 ||
		    m_fea_type == FEA_GABOR240 || m_fea_type == FEA_GABOR320)
	{
		// Gabor Feature
		IplImage* pImgHeader = cvCreateImageHeader(cvSize(m_nw, m_nh), IPL_DEPTH_8U, 1);
		cvSetData(pImgHeader, m_pData, m_nw*sizeof(uchar));

		icvGaborFeatures(&m_theGabor, pImgHeader, pFeature, &rc);

		cvReleaseImageHeader(&pImgHeader);
	}
	*/
	else if(m_fea_type == FEA_SURF64)
	{
		m_fea_ng =4;
		CvMat src_img = cvMat( m_nh,  m_nw, CV_32FC1, m_pFDat);
		icvDenseSURF(&src_img, pFeature, rc, m_fea_ng, 1);
	}
	else if(m_fea_type == FEA_SURF128)
	{
		m_fea_ng =4;
		CvMat src_img = cvMat( m_nh,  m_nw, CV_32FC1, m_pFDat);
		icvDenseSURF(&src_img, pFeature, rc, m_fea_ng);
	}
	/*
	else if(m_fea_type == FEA_GABOR_SIFT256)
	{
		// Gabor Feature
		float** pMag = new float* [8];
		m_theGabor.outputGaborMagnitue(pMag);

		for(int i=0; i<8; ++i)
		{
			CvMat matHead = cvMat(m_nh, m_nw, CV_32F, pMag[i]);

			float *pFea = pFeature + i* (2 * 2 * 8);
			icvDenseSIFTFeatures(&matHead, pFea, rc, false, 2, 2);
		}
		delete [] pMag;
	}
	*/
	else if(m_fea_type == FEA_CSLBP_8U)
	{
		CvMat lowImg = cvMat(m_nh, m_nw, CV_32F, m_pFDat);
		CvMat weightImg = cvMat(m_nh, m_nw, CV_32F, m_pFDat + m_sz);
		CvMat pImg, wImg;
		cvGetSubRect(&lowImg, &pImg, rc);
		cvGetSubRect(&weightImg, &wImg, rc);

		float peak_threshold = sqrtf(4.0f/m_nFeaDim);
		CvRect subrc[16];
		{
			int rcw = rc.width;
			int rch = rc.height;
			subrc[0] = cvRect(0, 0, rcw/4, rch/4);	subrc[1] = cvRect(rcw/4, 0, rcw/4, rch/4);
			subrc[2] = cvRect(rcw/2, 0, rcw/4, rch/4); subrc[3] = cvRect(3*rcw/4, 0, rcw/4, rch/4);

			subrc[4] = cvRect(0, rch/4, rcw/4, rch/4); subrc[5] = cvRect(rcw/4, rch/4, rcw/4, rch/4);
			subrc[6] = cvRect(rcw/2, rch/4, rcw/4, rch/4); subrc[7] = cvRect(3*rcw/4, rch/4, rcw/4, rch/4);

			subrc[8] = cvRect(0, rch/2, rcw/4, rch/4);	subrc[9] = cvRect(rcw/4,  rch/2, rcw/4, rch/4);
			subrc[10] = cvRect(rcw/2, rch/2, rcw/4, rch/4); subrc[11] = cvRect(3*rcw/4, rch/2, rcw/4, rch/4);

			subrc[12] = cvRect(0, 3*rch/4, rcw/4, rch/4); subrc[13] = cvRect(rcw/4, 3*rch/4, rcw/4, rch/4);
			subrc[14] = cvRect(rcw/2, 3*rch/4, rcw/4, rch/4); subrc[15] = cvRect(3*rcw/4, 3*rch/4, rcw/4, rch/4);
		}
		for(int i=0; i<16; ++i)
		{
			computeWeightHist(&pImg, wImg, subrc[i], pFeature + i*8);
		}
		normalizeVector(pFeature, m_nFeaDim, NORM_L2HYS, peak_threshold);
		normalizeVector(pFeature, m_nFeaDim, NORM_L2HYS, peak_threshold);
	}
	else
	{
		// do nothing
	}

	return 0;
}

//////////////////////////////////////////////////////////////////////////
// LBP8U2 LUT for texture classification
void LBP8ImageFast(CvMat* srcImg, CvMat* destImg, int r /* =1 */, int LBPtype /* =0 */)
{
	CvSize sz = cvGetSize( srcImg );
	cv::Mat lbpImg( sz.height-2*r, sz.width-2*r, CV_8UC1 );
	cv::Mat cmpImg( sz.height-2*r, sz.width-2*r, CV_8UC1 );

	CvRect rcROI = cvRect(r, r, sz.width-2*r, sz.height-2*r);
	CvMat centerROI;
	cvGetSubRect(srcImg, &centerROI, rcROI);

	// LBP image
	cvSetZero(&(CvMat)lbpImg);	//Modified by Nianzu
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
		cvCmp(&shiftNeighbor, &centerROI, &(CvMat)cmpImg, CV_CMP_GE );	//Modified by Nianzu

		cvAddS(&(CvMat)lbpImg, cvScalar(scl[k]), &(CvMat)lbpImg, &(CvMat)cmpImg);	//Modified by Nianzu
	}
	if( LBPtype == 0 ) // full-LBP
	{
		for(int y=0; y<sz.height-2*r; ++y)
		{
			for(int x=0; x<sz.width-2*r; ++x)
			{
				CvMat* cv_lbpImg = &CvMat(lbpImg);
				CV_MAT_ELEM(*destImg, float, y, x) = CV_MAT_ELEM(*cv_lbpImg, uchar, y, x); //Modified by Nianzu
				//cvReleaseMat(&cv_lbpImg);
			}
		}
	}
	else if( LBPtype == 1 ) // uniform-LBP
	{
		for(int y=0; y<sz.height-2*r; ++y)
		{
			for(int x=0; x<sz.width-2*r; ++x)
			{
				CvMat* cv_lbpImg = &CvMat(lbpImg);
				int val = CV_MAT_ELEM(*cv_lbpImg, uchar, y, x); //Modified by Nianzu
				CV_MAT_ELEM(*destImg, float, y, x) = gLBP8U2[val];
			}
		}
	}
}

void LBP8ImageFast(IplImage* srcImg, CvMat* destImg, int r /* =1 */, int LBPtype /* =0 */)
{
	LBP8ImageFast((CvMat*)srcImg, destImg, r, LBPtype);
}

// Gabor features
//void icvGaborFeatures( CxLogGabor *pLogGabor, IplImage* src_img, float* feature_vec, CvRect* rect /*= NULL*/)
/*
{
	CvRect rc;
	if(rect == NULL)
		rc = cvRect(0,0,src_img->width, src_img->height);
	else
		rc = *rect;

	rarray vecfea;
	std::vector <CvRect> vRect(1);
	vRect[0] = rc;

	pLogGabor->haarHistFeature(vecfea, vRect);

	for(int i = 0; i < vecfea.size(); i++)
		feature_vec[i] = vecfea[i];
}
*/
//////////////////////////////////////////////////////////////////////////
int icvDenseSURF(CvMat* src_img, float* pFea, CvRect rc, int ng /*=4*/, int extend/*=0*/)
{
	int nGrid = MAX(ng, 1);
	int ratio = MAX((rc.width + rc.height)/(2*nGrid*5), 1);
	int r = 5 *ratio;	// minimal diameter in each grid
	int PATCH_SZ = nGrid* r;

	// gaussian weight table
	cv::Mat DW(PATCH_SZ, PATCH_SZ, CV_32F);
	const double sigma = PATCH_SZ/6.0f;	// PATCH_SZ = 6sigma
	double c2 = 1./(sigma*sigma*2), gs = 0;
	double s = rc.width*1.2f/9.0f;
	int i, j, kk;
	for(i = 0; i < PATCH_SZ; i++ ) // y
	{
		for( j = 0; j < PATCH_SZ; j++ ) // x
		{
			double x = j - PATCH_SZ*0.5;
			double y = i - PATCH_SZ*0.5;
			double val = exp(-(x*x+y*y)*c2);
			cvmSet(&(CvMat)DW, i, j, val); //Modified by Nianzu
			gs += val;
		}
	}
	// zoom weight for numerical precision consideration
	double zoom = 100.0f;
	cvScale(&(CvMat)DW, &(CvMat)DW, zoom/gs );	//Modified by Nianzu

	// copy the target window
	float angle = 0;
	float cos_dir = cosf(angle);	// 1
	float sin_dir = sinf(angle);	// 0
	CvPoint2D32f pt = cvPoint2D32f(rc.x+ (rc.width-1)/2.0f, rc.y + (rc.height-1)/2.0f);

	CvMat* win = cvCreateMat(rc.height, rc.width, CV_32F);
	cvSetZero(win);
	// Nearest neighbor version to get the target rectangle
	{
		float rx = -(float)(rc.width-1)/2.0f;
		float ry = -(float)(rc.height-1)/2.0f;
		float start_x = pt.x + rx*cos_dir + ry*sin_dir;
		float start_y = pt.y - rx*sin_dir + ry*cos_dir;

		for( i = 0; i < rc.height; i++) // row, y
		{
			float pixel_x = start_x;
			float pixel_y = start_y;
			for( j = 0; j < rc.width; j++) // column, x
			{
				int x = MIN(MAX(cvRound(pixel_x), 0), src_img->cols-1);
				int y = MIN(MAX(cvRound(pixel_y), 0), src_img->rows-1);
				float val = cvmGet(src_img, y, x);
				cvmSet(win, i, j, val);

				pixel_x += cos_dir;
				pixel_y -= sin_dir;
			}
			start_x += sin_dir;
			start_y += cos_dir;
		}
	}
	// obtain the patch window
	cv::Mat patch(PATCH_SZ+2, PATCH_SZ+2, CV_32F);
	if( rc.width == PATCH_SZ+2 && rc.height == PATCH_SZ+2 )
		cvCopy(win, &(CvMat)patch);	//Modified by Nianzu
	else
		cvResize(win, &(CvMat)patch, CV_INTER_NN);	//Modified by Nianzu
	// increase, using default CV_INTER_LINEAR; decrease using CV_INTER_AREA
	cvReleaseMat(&win);

	// gradient
	cv::Mat DX(PATCH_SZ, PATCH_SZ, CV_32F);
	cv::Mat DY(PATCH_SZ, PATCH_SZ, CV_32F);

	CvMat* p = &(CvMat)patch;	//Modified by Nianzu
	for( i = 1; i < PATCH_SZ+1; i++ )	// y
	{
		for( j = 1; j < PATCH_SZ+1; j++ ) // x
		{
			double dw = cvmGet(&(CvMat)DW, i-1, j-1);	//Modified by Nianzu
			double dx = (cvmGet(p,i,j+1) - cvmGet(p,i,j-1) )*dw;
			double dy = (cvmGet(p,i+1,j) - cvmGet(p,i-1,j) )*dw;

			cvmSet(&(CvMat)DX, i-1, j-1, dx);	//Modified by Nianzu
			cvmSet(&(CvMat)DY, i-1, j-1, dy);	//Modified by Nianzu
		}
	}
	patch.release();
	DW.release();

	// 64-bin descriptor
	float* vec = pFea;
	double square_mag = 0;
	int d = 1;
	if(extend==0)
	{
		d = nGrid*nGrid*4;
		memset(pFea, 0, d*sizeof(float));
		for( i = 0; i < nGrid; i++ )
		{
			for( j = 0; j < nGrid; j++ )
			{
				for(int y = i*r; y < i*r +r; y++ )
				{
					for(int x = j*r; x < j*r +r; x++ )
					{
						float tx = cvmGet(&(CvMat)DX, y, x);	//Modified by Nianzu
						float ty = cvmGet(&(CvMat)DY, y, x);	//Modified by Nianzu
						vec[0] += tx; 
						vec[1] += ty;
						vec[2] += fabsf(tx); 
						vec[3] += fabsf(ty);
					}
				}					
				for( kk = 0; kk < 4; kk++ )
					square_mag += vec[kk]*vec[kk];

				vec+=4;
			}
		}
	}
	else
	{
		d = nGrid*nGrid*4*2;
		memset(pFea, 0, d*sizeof(float));

		// extended 128 dim
		for( i = 0; i < nGrid; i++ )
		{
			for( j = 0; j < nGrid; j++ )
			{
				for(int y = i*r; y < i*r +r; y++ )
				{
					for(int x = j*r; x < j*r +r; x++ )
					{
						float tx = cvmGet(&(CvMat)DX, y, x);	//Modified by Nianzu
						float ty = cvmGet(&(CvMat)DY, y, x);	//Modified by Nianzu
						if( ty >= 0 )
						{
							vec[0] += tx;
							vec[1] += fabsf(tx);
						} 
						else {
							vec[2] += tx;
							vec[3] += fabsf(tx);
						}
						if ( tx >= 0 )
						{
							vec[4] += ty;
							vec[5] += fabsf(ty);
						} 
						else {
							vec[6] += ty;
							vec[7] += fabsf(ty);
						}
					}
				}
				for( kk = 0; kk < 8; kk++ )
					square_mag += vec[kk]*vec[kk];
				vec += 8;
			}
		}
	}
	double scale = 1./(sqrt(square_mag) + DBL_EPSILON);
	for(kk = 0; kk < d; kk++)
		pFea[kk] = (float)(pFea[kk]*scale);

	return 1;
}

//////////////////////////////////////////////////////////////////////////
#define  GET_IMG_STEP(img)(img->widthStep*8/img->depth)
#if 1       // SIMD optimized code
void icvDenseSIFTFeaturesConst( IplImage*  src_img,
						  float* feature_vec, int bin_num, int ori_bin_num, float coeff_sigma, float threshold,
						  CvPoint pt, float sigma, float angle, int Rx, int Ry )
{
	CvRect rect = cvGetImageROI(src_img);
	int step = GET_IMG_STEP(src_img);
	float* data = (float*)src_img->imageData+rect.y*step+rect.x;
	memset(feature_vec, 0, bin_num*bin_num*ori_bin_num*sizeof(float));
	float bin_size = sigma*coeff_sigma; //size of feature bin (in pixels)

	float cosang=cos(angle)/bin_size;
	float sinang=sin(angle)/bin_size;

	int t1 = MIN(rect.height-2, pt.y + Ry);
	int t2 = MIN(rect.width-2, pt.x + Rx);
	int t3 = MAX(1, pt.y - Ry);
	int t4 = MAX(1, pt.x - Rx);

	float f1 = (bin_num - 1)*0.5f;
	float f2 = -2.0f/(bin_num*bin_num);
	float f3 = ori_bin_num/(2*CV_PI);
	float f6 = 2*CV_PI;

	int sz = ((t1-t3+1)*(t2-t4+1)/4+1)*4;
	int nThreads = 1;
#ifdef __SSE2__
	float *buf = (float *)_mm_malloc(sizeof(float)*(sz*8*nThreads), 16);
#else
	float *buf = (float *)malloc(sizeof(float)*(sz*8*nThreads));
#endif

	float *new_angles = buf;
	float *grad_vals = new_angles+sz;
	float *gradXs = grad_vals+sz;
	float *gradYs = gradXs+sz;
	float *x_bins = gradYs+sz;
	float *y_bins = x_bins+sz;
	float *x_bin_indxs = y_bins + sz;
	float *y_bin_indxs = x_bin_indxs + sz;

	sz = 0;
	//explore keypoint location and fill feature vector
	for( int yy = t3; yy <= t1; yy++ )
	{
		int y = yy - pt.y;
		float f4 = sinang*y;
		float f5 = cosang*y;
		for( int xx = t4; xx <= t2; xx++ )
		{
			int x = xx - pt.x;
			//rotate pixel 
			float x_bin_offset = (cosang*x-f4); //X offset from keypoint location (in bins)
			float y_bin_offset = (sinang*x+f5); //Y offset (in bins)
			float x_bin = x_bin_offset + f1; // current bin position (in bins)
			float y_bin = y_bin_offset + f1; //  

			if( x_bin > -1 && x_bin < bin_num && y_bin > -1 && y_bin < bin_num ){ 
				int t;
				t = sz++;
				y_bins[t] = y_bin;
				x_bins[t] = x_bin;
				gradXs[t] = data[yy*step+xx+1]-data[yy*step+xx-1];
				gradYs[t] = data[(yy+1)*step+xx]-data[(yy-1)*step+xx];
			}
		}
	}
#ifndef __SSE2__
	for(int i=0;i<sz;i++)
	{
		float gradX = gradXs[i];
		float gradY = gradYs[i];
		float x_bin_offset = x_bins[i]-f1;//(cosang*x-f4); //X offset from keypoint location (in bins)
		float y_bin_offset = y_bins[i]-f1;//(sinang*x+f5);
		grad_vals[i] = (float)(sqrt(gradX*gradX+gradY*gradY)*
			exp((x_bin_offset*x_bin_offset+y_bin_offset*y_bin_offset)*f2));
		float new_angle = atan2(gradY, gradX)-angle;
		if( new_angle < 0 ) new_angle += (float)(2*CV_PI);
		new_angles[i] = new_angle*f3;
		float x_bin = x_bins[i];
		float y_bin = y_bins[i];
		x_bin_indxs[i] = ((x_bin >= 0) ? x_bin : x_bin - 1);
		y_bin_indxs[i] = ((y_bin >= 0) ? y_bin : y_bin - 1);
	}
#else
	__m128 vf1    = _mm_set1_ps(f1);
	__m128 vf2    = _mm_set1_ps(f2);
	__m128 vf3    = _mm_set1_ps(f3);
	__m128 vangle = _mm_set1_ps(angle);
	__m128 v0     = _mm_setzero_ps();
	__m128 v1     = _mm_set1_ps(1.0f);
	__m128 v2Pi   = _mm_set1_ps(2*CV_PI);

	for(int i=0;i<sz-3;i+=4)
	{
		__m128 x = _mm_load_ps(x_bins+i);
		__m128 y = _mm_load_ps(y_bins+i);
		x = _mm_sub_ps(x, vf1);
		y = _mm_sub_ps(y, vf1);
		x = _mm_mul_ps(x, x);
		y = _mm_mul_ps(y, y);
		x = _mm_add_ps(x, y);
		x = _mm_mul_ps(x, vf2);
		_mm_store_ps(grad_vals+i, x);
	}

	// ippsExp_32f_I(grad_vals, sz);
	// ippsAtan2_32f_A11 (gradYs, gradXs, new_angles, sz);

	// ICC did a good job for array "exp" and "atan2", even faster than IPP code
	for(int i=0;i<sz;i++)
		grad_vals[i] = exp(grad_vals[i]);
	for(int i=0;i<sz;i++)
		new_angles[i] = atan2(gradYs[i], gradXs[i]);

	for(int i=0;i<sz-3;i+=4)
	{
		__m128 x = _mm_load_ps(gradXs+i);
		__m128 y = _mm_load_ps(gradYs+i);
		x = _mm_mul_ps(x, x);
		y = _mm_mul_ps(y, y);
		x = _mm_add_ps(x, y);
		y = _mm_load_ps(grad_vals+i);
		x = _mm_sqrt_ps(x);
		x = _mm_mul_ps(x, y);
		_mm_store_ps(grad_vals+i, x);
		__m128 ang = _mm_load_ps(new_angles+i);
		ang = _mm_sub_ps(ang, vangle);
		x   = _mm_cmplt_ps(ang, v0);
		y   = _mm_and_ps(x, v2Pi);
		ang = _mm_add_ps(ang, y);
		ang = _mm_mul_ps(ang, vf3);
		_mm_store_ps(new_angles+i,ang);

		x = _mm_load_ps(x_bins+i);
		y = _mm_load_ps(y_bins+i);
		__m128 m = _mm_cmplt_ps(x, v0);
		m = _mm_and_ps(m, v1);
		x = _mm_sub_ps(x, m);
		_mm_store_ps(x_bin_indxs+i, x);
		m = _mm_cmplt_ps(y, v0);
		m = _mm_and_ps(m, v1);
		y = _mm_sub_ps(y, m);
		_mm_store_ps(y_bin_indxs+i,y);
	}

	for(int i=sz-(sz%4); i<sz; i++)
	{
		float gradX = gradXs[i];
		float gradY = gradYs[i];
		float x_bin_offset = x_bins[i]-f1;//(cosang*x-f4); //X offset from keypoint location (in bins)
		float y_bin_offset = y_bins[i]-f1;//(sinang*x+f5);
		grad_vals[i] = (float)(sqrt(gradX*gradX+gradY*gradY)*
			exp((x_bin_offset*x_bin_offset+y_bin_offset*y_bin_offset)*f2));
		float new_angle = atan2(gradY, gradX)-angle;
		if( new_angle < 0 ) new_angle += (float)(2*CV_PI);
		new_angles[i] = new_angle*f3;
		float x_bin = x_bins[i];
		float y_bin = y_bins[i];
		x_bin_indxs[i] = ((x_bin >= 0) ? x_bin : x_bin - 1);
		y_bin_indxs[i] = ((y_bin >= 0) ? y_bin : y_bin - 1);
	}
#endif

	for(int i=0;i<sz;i++)
	{
		{
			{ 
				float ori_bin = new_angles[i];//(float)(new_angle*f3);
				float x_bin = x_bins[i];// = x_bin_offset + f1; // current bin position (in bins)
				float y_bin = y_bins[i];// = y_bin_offset + f1; // 

				int ori_bin_indx = (int)(ori_bin);
				int x_bin_indx = (int)x_bin_indxs[i];
				int y_bin_indx = (int)y_bin_indxs[i];

				float xoff = x_bin-x_bin_indx;
				float yoff = y_bin-y_bin_indx;
				float orioff = ori_bin - ori_bin_indx;

				float grad_val = grad_vals[i];
				float valY = grad_val*(1-yoff);

				int t3=MIN(y_bin_indx+2, bin_num);
				for(int indxY = MAX(y_bin_indx, 0); indxY < t3; indxY++ ){
					float valX = valY*(1-xoff);
					int t4 = MIN(x_bin_indx+2, bin_num);
					for(int indxX = MAX(x_bin_indx, 0); indxX < t4; indxX++ ){
						float val = valX*(1-orioff);
						int t5 = bin_num*indxY+indxX;
						//int t6 = bin_num*indxY+indxX;
						for( int indxOri = ori_bin_indx; indxOri < ori_bin_indx + 2; indxOri++){
							int indx = (indxOri >= ori_bin_num) ? t5 : bin_num*bin_num*indxOri + t5;

							feature_vec[indx] += val;

							val = valX * orioff;
						}
						valX = valY * xoff;   
					}                     
					valY = grad_val * yoff;
				}
			}
		}
	}
#ifdef __SSE2__
	_mm_free(buf);
#else
	free (buf);
#endif

	//normalize vector to unit length to reduce the effects of illumination change
	float sum = 0;
	int    N = bin_num*bin_num*ori_bin_num; //number of features
#ifndef __SSE2__
	for( int i = 0; i < N; i++ ) sum += (feature_vec[i]*feature_vec[i]);
#else
	__m128 v = _mm_setzero_ps();
	for(int i=0;i<N-3;i+=4)
	{
		__m128 f = _mm_loadu_ps(feature_vec+i);
		f = _mm_mul_ps(f, f);
		v = _mm_add_ps(v, f);
	}
	_MM_ALIGN16 float ttmp[4];
	_mm_store_ps(ttmp,v);
	sum = ttmp[0]+ttmp[1]+ttmp[2]+ttmp[3];

	for(int i=N-(N%4); i<N; i++)
		sum += (feature_vec[i]*feature_vec[i]);
#endif

	float  sum2 = 0;
	sum = 1/sqrt(sum+FLT_EPSILON);
#ifndef __SSE2__
	for( int i = 0; i < N; i++ ){
		feature_vec[i] *= sum;
		//threshold the value to reduce the influence of large gradient 
		if( feature_vec[i] > threshold ) feature_vec[i] = threshold;        
		sum2 += (feature_vec[i]*feature_vec[i]);
	}
#else 
	v = _mm_setzero_ps();
	__m128 s = _mm_set1_ps(sum);
	__m128 t = _mm_set1_ps(threshold);
	for(int i=0;i<N-3;i+=4)
	{
		__m128 f = _mm_loadu_ps(feature_vec+i);
		f        = _mm_mul_ps(f, s);
		__m128 m = _mm_cmpgt_ps(f,t);
		f        = _mm_or_ps(_mm_and_ps(m,t), _mm_andnot_ps(m,f));
		_mm_storeu_ps(feature_vec+i, f);
		f        = _mm_mul_ps(f,f);
		v        = _mm_add_ps(v, f);
	}
	_MM_ALIGN16 float tsum[4];
	_mm_store_ps(tsum,v);
	sum2 = tsum[0]+tsum[1]+tsum[2]+tsum[3];

	for(int i=N-(N%4); i<N; i++)
	{
		feature_vec[i] *= sum;
		//threshold the value to reduce the influence of large gradient 
		if( feature_vec[i] > threshold ) feature_vec[i] = threshold;        
		sum2 += (feature_vec[i]*feature_vec[i]);  
	}
#endif

	sum2 = 1/sqrt(sum2+FLT_EPSILON);
	if( sum!=sum2 ){
		//normalize again
		for( int i = 0; i < N; i++ ) feature_vec[i] *= sum2;   
	}
}

#else

// original implementation
void icvDenseSIFTFeaturesConst( IplImage*  src_img,
						  float* feature_vec, int bin_num, int ori_bin_num, float coeff_sigma, float threshold,
						  CvPoint pt, float sigma, float angle, int Rx, int Ry )
{
	CvRect rect = cvGetImageROI(src_img);
	int step = GET_IMG_STEP(src_img);
	float* data = (float*)src_img->imageData+rect.y*step+rect.x;
	memset(feature_vec, 0, bin_num*bin_num*ori_bin_num*sizeof(float));
	float bin_size = sigma*coeff_sigma; //size of feature bin (in pixels)
	//   int R = cvRound( bin_size*sqrt(2.0)*(bin_num+1)*0.5); //maximal radius of interest pixels (in pixels)

	//explore keypoint location and fill feature vector
	for( int yy = MAX(1, pt.y - Ry); yy <= MIN(rect.height-2, pt.y + Ry); yy++ )
	{
		for( int xx = MAX(1, pt.x - Rx); xx <= MIN(rect.width-2, pt.x + Rx); xx++ )
		{
			int y = yy - pt.y;
			int x = xx - pt.x;
			//rotate pixel fixed by wuyi
			float x_bin_offset = (cos(angle)*x-sin(angle)*y)/bin_size; //X offset from keypoint location (in bins)
			float y_bin_offset = (sin(angle)*x+cos(angle)*y)/bin_size; //Y offset (in bins)

			float x_bin = x_bin_offset + (bin_num - 1)*0.5f; // current bin position (in bins)
			float y_bin = y_bin_offset + (bin_num - 1)*0.5f; //
			int   x_bin_indx = (int)((x_bin >= 0) ? x_bin : x_bin - 1); //get first indexes of adjacent histogram bins
			int   y_bin_indx = (int)((y_bin >= 0) ? y_bin : y_bin - 1);

			if( x_bin > -1 && x_bin < bin_num && y_bin > -1 && y_bin < bin_num )
			{
				double gradX = data[yy*step+xx+1]-data[yy*step+xx-1];
				double gradY = data[(yy+1)*step+xx]-data[(yy-1)*step+xx];
				//calc weighted gradient
				double grad_val = sqrt(gradX*gradX+gradY*gradY)*
					exp(-(x_bin_offset*x_bin_offset+y_bin_offset*y_bin_offset)/(bin_num*bin_num*0.5));
				//extract orientation bin angle
				double new_angle = atan2(gradY, gradX) - angle; //can varied from -2Pi to 2Pi (need to be in [0, 2Pi])
				if( new_angle < 0 ) new_angle += 2*CV_PI;
				//get orientation bin index
				//float ori_bin = cvRound(new_angle*(ori_bin_num-1)/(2*CV_PI));
				double ori_bin = new_angle*ori_bin_num/(2*CV_PI);
				int   ori_bin_indx = (int)(ori_bin);
				//perform an interpolation to distribute the value of each gradient sample into 8 adjacent histogram bins
				double valY = grad_val*(1-(y_bin-y_bin_indx));
				for(int indxY = MAX(y_bin_indx, 0); indxY < MIN(y_bin_indx+2, bin_num); indxY++ )
				{
					double valX = valY*(1-(x_bin-x_bin_indx));
					for(int indxX = MAX(x_bin_indx, 0); indxX < MIN(x_bin_indx+2, bin_num); indxX++ )
					{
						double val = valX*(1-(ori_bin-ori_bin_indx));
						for( int indxOri = ori_bin_indx; indxOri < ori_bin_indx + 2; indxOri++)
						{
							int indx = (indxOri >= ori_bin_num) ?
								bin_num*indxY+indxX : bin_num*bin_num*indxOri+bin_num*indxY+indxX;
							feature_vec[indx] += val;
							val = valX*(ori_bin-ori_bin_indx);
						}
						valX = valY*(x_bin-x_bin_indx);
					}
					valY = grad_val*(y_bin-y_bin_indx);
				}
			}
		}//x
	}//y

	//normalize vector to unit length to reduce the effects of illumination change
	float sum = 0;
	int    N = bin_num*bin_num*ori_bin_num; //number of features
	for( int i = 0; i < N; i++ ) sum += (feature_vec[i]*feature_vec[i]);
	double  sum2 = 0;
	sum = 1/sqrt(sum+FLT_EPSILON);
	for( int i = 0; i < N; i++ ){
		feature_vec[i] *= sum;
		//threshold the value to reduce the influence of large gradient
		if( feature_vec[i] > threshold ) feature_vec[i] = threshold;
		sum2 += (feature_vec[i]*feature_vec[i]);
	}
	sum2 = 1/sqrt(sum2+FLT_EPSILON);
	if( sum!=sum2 ){
		//normalize again
		for( int i = 0; i < N; i++ ) feature_vec[i] *= sum2;
	}
	//for( int i = 0; i < N; i++ ) printf("%d ", (int)(255*feature_vec[i]));printf("\n");
}
#endif

/*
Interpolates an entry into the array of orientation histograms that form the feature descriptor
// both osu & FastSIFT implementation

@param hist 1D array of orientation histograms
@param xbin sub-bin column coordinate of entry
@param ybin sub-bin row coordinate of entry
@param obin sub-bin orientation coordinate of entry
@param mag size of entry
@param d width of 2D array of orientation histograms
@param n number of bins per orientation histogram
*/
void interp_hist_entry(float* hist, float xbin, float ybin, float obin, float mag, 
					   int nXBin /*= 4*/, int nYBin /*= 4*/, int nOBin /*= 8*/)
{
	float d_r, d_c, d_o, v_r, v_c, v_o;
	int r0, c0, o0, rb, cb, ob, r, c, o;

	c0 = cvFloor( xbin );
	r0 = cvFloor( ybin );
	o0 = cvFloor( obin );
	d_c = xbin - c0;
	d_r = ybin - r0; 
	d_o = obin - o0;

	for( r = 0; r <= 1; r++ )
	{
		rb = r0 + r;
		if( rb >= 0  &&  rb < nYBin ) // row
		{
			v_r = mag * ( ( r == 0 )? 1.0 - d_r : d_r );
			for( c = 0; c <= 1; c++ )
			{
				cb = c0 + c;
				if( cb >= 0  &&  cb < nXBin ) // col
				{
					v_c = v_r * ( ( c == 0 )? 1.0 - d_c : d_c );
					for( o = 0; o <= 1; o++ )
					{
						ob = ( o0 + o ) % nOBin;
						v_o = v_c * ( ( o == 0 )? 1.0 - d_o : d_o );

						int indx = nXBin*nYBin*ob + nXBin* rb + cb;
						hist[indx] += v_o;
					}
				}
			}
		}
	}
}

/*
@src_img  input images, should be CV32F single channel
@pFea   the sift descriptor: input point position, scale/orientation, output features
@rc    rect region for dense SIFT
keypoint at (rx.x + (rc.width-1)/2, rc.y + (rc.height-1)/2)
@bQuantize  quantization of descriptor
// for non-zoom images, the general sigma sequence is {0.5, 0.8, 1.2, 1.7, 2.3, 3.0, 3.8};
// diameter = cvRound( 6.0 * sigma*sqrtf(2.0f)*(bin_num+1)*0.5);
// the corresponding diameter sequence = {11, 18, 25, 36, ...}
*/

#if 0
int icvDenseSIFTFeatures(CvMat* src_img, float* pFea, CvRect rc, bool bQuantize /*=false*/,
						  int nx /*=4*/, int ny /*=4*/, int no /*=8*/)
{
	if( src_img == NULL )
		return -1;

	double PI2 = 2.0 * CV_PI;
	float coeff_sigma = 3.0f;  // 3-sigma
	float peak_threshold = 0.2f; // histogram peak threshold
	float sigma = 1.6;
	float angle = 0;

	int bin_num_x = max(nx, 1);  // nx*ny cell, default 4x4 cells
	int bin_num_y = max(ny, 1);
	int bin_num_o = max(no, 4);  // default 8 bins for each ceil

	int w = src_img->width;
	int h = src_img->height;
	int N = bin_num_x*bin_num_y*bin_num_o; //number of features
	memset(pFea, 0, N*sizeof(float));

	// float bin_size = coeff_sigma * sigma; //size of feature bin (in pixels)
	// // maximal radius of interest pixels (in pixels)
	// int R = cvRound( bin_size*sqrtf(2.0f)*(bin_num+1)*0.5);
	int Rx = (rc.width-1)/2;
	int Ry = (rc.height-1)/2;
	float bin_size_x = Rx/(0.707f*(bin_num_x+1));
	float bin_size_y = Ry/(0.707f*(bin_num_y+1));
	CvPoint2D32f pt;
	pt.x = (rc.x + Rx);
	pt.y = (rc.y + Ry);

	int xx, yy, i;
	float denom_x = 1.0/(SQR(bin_num_x)*0.5);
	float denom_y = 1.0/(SQR(bin_num_y)*0.5);
	float bins_per_rad = bin_num_o / PI2;
	float cos_t = cosf( angle );
	float sin_t = sinf( angle );

	//explore keypoint location and fill feature vector
	for(yy = max(1, pt.y - Ry); yy <= min(h-2, pt.y + Ry); yy++ )
	{
		for(xx = max(1, pt.x - Rx); xx <= min(w-2, pt.x + Rx); xx++ )
		{
			float x = xx - pt.x;
			float y = yy - pt.y;

			// rotate pixel
			// in OSU, Intel, FastSIFT definition
			float x_bin_offset = (cos_t*x - sin_t*y)/bin_size_x;
			float y_bin_offset = (sin_t*x + cos_t*y)/bin_size_y;

			float x_bin = x_bin_offset + (bin_num_x - 1)*0.5f; // current bin position (in bins)
			float y_bin = y_bin_offset + (bin_num_y - 1)*0.5f; //
			if( x_bin > -1 && x_bin < bin_num_x && y_bin > -1 && y_bin < bin_num_y )
			{
				// gradient
				float gradX = cvmGet(src_img, yy, xx+1) - cvmGet(src_img, yy, xx-1);
				float gradY = cvmGet(src_img, yy+1, xx) - cvmGet(src_img, yy-1, xx);

				//calc weighted gradient
				float mag = sqrtf(SQR(gradX) + SQR(gradY));
				float weight = expf(-denom_x*SQR(x_bin_offset) -denom_y*SQR(y_bin_offset) );
				float grad_val = mag * weight;

				//extract orientation bin angle
				//float new_angle = fast_atan2f_rad(gradY, gradX) - angle;
				float new_angle = atan2f(gradY, gradX) - angle;
				while( new_angle < 0.0 )
					new_angle += PI2;
				while( new_angle >= PI2 )
					new_angle -= PI2;

				//get orientation bin index
				float o_bin = new_angle * bins_per_rad;

				// Distribute the current sample into the 8 adjacent bins in an interpolation way
				// or place index in an interpolation way
				interp_hist_entry(pFea, x_bin, y_bin, o_bin, grad_val, 
					bin_num_x, bin_num_y, bin_num_o);
			}
		}//x
	}//y

	//normalize vector to unit length to reduce the effects of illumination change
	double sum = 0;
	for(i = 0; i < N; ++i) 
		sum += (pFea[i]*pFea[i]);
	double sum2 = 0;
	sum = 1/sqrt(sum + FLT_EPSILON);
	for(i = 0; i < N; i++ )
	{
		pFea[i] *= sum;
		//threshold the value to reduce the influence of large gradient
		if( pFea[i] > peak_threshold ) 
			pFea[i] = peak_threshold;
		sum2 += (pFea[i]*pFea[i]);
	}
	sum2 = 1/sqrt(sum2 + FLT_EPSILON);
	if( sum != sum2 )
	{
		//normalize again
		for(i = 0; i < N; i++ ) 
			pFea[i] *= sum2;
	}

	// convert floating-point descriptor to integer valued descriptor
	if( bQuantize )
	{  
		for(i = 0; i < N; i++)
		{
			int val = 512.0 * pFea[i];
			pFea[i] = min(255, val);
		}
	}

	return 1;
}

#else

int icvDenseSIFTFeatures(CvMat* src_img, float* pFea, CvRect rc, bool bQuantize /*=false*/,
						  int nx /*=4*/, int ny /*=4*/, int no /*=8*/)
{
	if( src_img == NULL )
		return -1;

	double PI2 = 2.0 * CV_PI;
	float coeff_sigma = 3.0f;  // 3-sigma
	float peak_threshold = 0.2f; // histogram peak threshold
	float sigma = 1.6;
	float angle = 0;

	int bin_num_x = max(nx, 1);  // nx*ny cell, default 4x4 cells
	int bin_num_y = max(ny, 1);
	int bin_num_o = max(no, 4);  // default 8 bins for each ceil

	int w = src_img->width;
	int h = src_img->height;
	int N = bin_num_x*bin_num_y*bin_num_o; //number of features
	memset(pFea, 0, N*sizeof(float));

	// float bin_size = coeff_sigma * sigma; //size of feature bin (in pixels)
	// // maximal radius of interest pixels (in pixels)
	// int R = cvRound( bin_size*sqrtf(2.0f)*(bin_num+1)*0.5);
	int Rx = (rc.width-1)/2;
	int Ry = (rc.height-1)/2;
	float bin_size_x = Rx/(0.707f*(bin_num_x+1));
	float bin_size_y = Ry/(0.707f*(bin_num_y+1));
	CvPoint2D32f pt;
	pt.x = (rc.x + Rx);
	pt.y = (rc.y + Ry);

	int xx, yy, i, j;
	float denom_x = 1.0/(SQR(bin_num_x)*0.5);
	float denom_y = 1.0/(SQR(bin_num_y)*0.5);
	float bins_per_rad = bin_num_o / PI2;
	float cos_t = cosf( angle );
	float sin_t = sinf( angle );
	
	int ymax = MIN(h-2, pt.y + Ry);
    int xmax = MIN(w-2, pt.x + Rx);
    int ymin = MAX(1, pt.y - Ry);
    int xmin = MAX(1, pt.x - Rx);

    float x_bin_num = (bin_num_x - 1)*0.5f;
	float y_bin_num = (bin_num_y - 1)*0.5f;
	float cos_bin_x = cos_t/bin_size_x;
	float sin_bin_x = sin_t/bin_size_x;
	float cos_bin_y = cos_t/bin_size_y;
	float sin_bin_y = sin_t/bin_size_y;
	
    int sz = ((ymax-ymin+1)*(xmax-xmin+1)/4+1)*4;
	
#ifdef __SSE2__
    float *pt_buf = (float *)_mm_malloc(sizeof(float)*(sz*6), 16);
#else
	float *pt_buf = (float *)malloc(sizeof(float)*(sz*6));
#endif
	
	float *x_bins = pt_buf;
	float *y_bins = x_bins+sz;
    float *o_bins = y_bins+sz;
	float *gradXs = o_bins+sz;
    float *gradYs = gradXs+sz;
    float *grad_vals = gradYs+sz;
    
	int step = src_img->step >> 2;
	float* data = (float*)src_img->data.fl;

	sz = 0;
	//explore keypoint location and fill feature vector
	for(yy = ymin; yy <= ymax; yy++ )
	{
		for(xx = xmin; xx <= xmax; xx++ )
		{
			float x = xx - pt.x;
			float y = yy - pt.y;

			// rotate pixel
			// in OSU, Intel, FastSIFT definition
			float x_bin_offset = x*cos_bin_x - y*sin_bin_x;
			float y_bin_offset = x*sin_bin_y + y*cos_bin_y;

			float x_bin = x_bin_offset + x_bin_num; // current bin position (in bins)
			float y_bin = y_bin_offset + y_bin_num; //

			if( x_bin > -1 && x_bin < bin_num_x && y_bin > -1 && y_bin < bin_num_y )
			{
				int t = sz++;
                y_bins[t] = y_bin;
                x_bins[t] = x_bin;

				gradXs[t] = *(data+yy*step+xx+1) - *(data+yy*step+xx-1);
                gradYs[t] = *(data+(yy+1)*step+xx)- *(data+(yy-1)*step+xx);
			}
		}
	}

#ifdef __SSE2__
	__m128 xmm_denom_x      = _mm_set1_ps(-denom_x);
	__m128 xmm_denom_y      = _mm_set1_ps(-denom_y);
	__m128 xmm_ybinnum      = _mm_set1_ps(y_bin_num);
	__m128 xmm_xbinnum      = _mm_set1_ps(x_bin_num);
	__m128 xmm_angle        = _mm_set1_ps(angle);
	__m128 xmm_PI2          = _mm_set1_ps(PI2);
	__m128 xmm_zero         = _mm_setzero_ps();
	__m128 xmm_binsperrad   = _mm_set1_ps(bins_per_rad);
#endif

	for(i=0; i<(sz/SIMD_WIDTH)*SIMD_WIDTH; i+=SIMD_WIDTH)
    {
#ifdef __SSE2__
		__m128 xmm_gradX    = _mm_load_ps(gradXs+i);
        __m128 xmm_gradY    = _mm_load_ps(gradYs+i);

		__m128 xmm_xbin     = _mm_load_ps(x_bins+i);
		__m128 xmm_ybin     = _mm_load_ps(y_bins+i);

		__m128 xmm_mag      = _mm_sqrt_ps(_mm_add_ps(_mm_sqr_ps(xmm_gradX), _mm_sqr_ps(xmm_gradY)));
		__m128 xmm_weight   = _mm_add_ps(_mm_mul_ps(xmm_denom_x, _mm_sqr_ps(_mm_sub_ps(xmm_xbin, xmm_xbinnum))), \
								_mm_mul_ps(xmm_denom_y, _mm_sqr_ps(_mm_sub_ps(xmm_ybin, xmm_ybinnum))));
		xmm_weight          = _mm_exp_ps(xmm_weight);
		_mm_store_ps (grad_vals+i, _mm_mul_ps(xmm_mag, xmm_weight));

		//extract orientation bin angle
		__m128 xmm_newangle = _mm_sub_ps(_mm_atan2_ps(xmm_gradY, xmm_gradX), xmm_angle);
		xmm_newangle        = _mm_add_ps(xmm_newangle, _mm_and_ps(_mm_cmplt_ps(xmm_newangle, xmm_zero), xmm_PI2));
		
		//get orientation bin index
		_mm_store_ps (o_bins+i, _mm_mul_ps(xmm_newangle, xmm_binsperrad));
#else
		for (j=0; j<SIMD_WIDTH; j++)
		{
			float gradX = gradXs[i+j];
			float gradY = gradYs[i+j];

			float x_bin = x_bins[i+j];
			float y_bin = y_bins[i+j];

			float x_bin_offset = x_bin - x_bin_num;
			float y_bin_offset = y_bin - y_bin_num;
	        
			//calc weighted gradient
			float mag = sqrtf(SQR(gradX) + SQR(gradY));
			float weight = expf(-denom_x*SQR(x_bin_offset) -denom_y*SQR(y_bin_offset) );
			grad_vals[i+j] = mag * weight;

			//extract orientation bin angle
			float new_angle = atan2f(gradY, gradX) - angle;
			
			if( new_angle < 0 ) new_angle += PI2;

			//get orientation bin index
			o_bins[i+j] = new_angle * bins_per_rad;
		}
#endif

		// Distribute the current sample into the 8 adjacent bins in an interpolation way
		// or place index in an interpolation way
		for (j=0; j<SIMD_WIDTH; j++)
		{
			interp_hist_entry(pFea, x_bins[i+j], y_bins[i+j], o_bins[i+j], grad_vals[i+j], 
				bin_num_x, bin_num_y, bin_num_o);
		}
	}

	for(; i<sz; i++)
    {
        float gradX = gradXs[i];
        float gradY = gradYs[i];

		float x_bin = x_bins[i];
		float y_bin = y_bins[i];

		float x_bin_offset = x_bin - x_bin_num;
        float y_bin_offset = y_bin - y_bin_num;
        
		//calc weighted gradient
		float mag = sqrtf(SQR(gradX) + SQR(gradY));
		float weight = expf(-denom_x*SQR(x_bin_offset) -denom_y*SQR(y_bin_offset) );
		float grad_val = mag * weight;

		//extract orientation bin angle
		float new_angle = atan2f(gradY, gradX) - angle;
		
		if( new_angle < 0 ) new_angle += PI2;

		//get orientation bin index
		float o_bin = new_angle * bins_per_rad;
	
		// Distribute the current sample into the 8 adjacent bins in an interpolation way
		// or place index in an interpolation way
		interp_hist_entry(pFea, x_bin, y_bin, o_bin, grad_val, 
			bin_num_x, bin_num_y, bin_num_o);
	}

#ifdef __SSE2__
    _mm_free(pt_buf);
#else
	free (pt_buf);
#endif

	//normalize vector to unit length to reduce the effects of illumination change
	double sum = 0;
	
#ifndef __SSE2__
	for(i = 0; i < N; ++i) 
		sum += (pFea[i]*pFea[i]);
#else
	F128 xmm_sum;
	xmm_sum.pack = _mm_setzero_ps();
	for(i = 0; i < N; i += SIMD_WIDTH)	//by default N is a multiple of 4/8
	{
		xmm_sum.pack = _mm_add_ps(xmm_sum.pack, _mm_sqr_ps (_mm_loadu_ps(pFea+i)));
	}
	sum = xmm_sum.f[0]+xmm_sum.f[1]+xmm_sum.f[2]+xmm_sum.f[3];
#endif

	double sum2 = 0;
	sum = 1/sqrt(sum + FLT_EPSILON);

#ifndef __SSE2__
	for(i = 0; i < N; i++ )
	{
		pFea[i] *= sum;
		//threshold the value to reduce the influence of large gradient
		if( pFea[i] > peak_threshold ) 
			pFea[i] = peak_threshold;
		sum2 += (pFea[i]*pFea[i]);
	}
#else
	F128 xmm_sum2;
	xmm_sum2.pack = _mm_setzero_ps();
	xmm_sum.pack  = _mm_set1_ps(sum);
    __m128 xmm_th = _mm_set1_ps(peak_threshold);

	for(i = 0; i < N; i += SIMD_WIDTH ) //by default N is a multiple of 4/8
	{
		__m128 xmm_fea = _mm_mul_ps(_mm_loadu_ps(pFea+i), xmm_sum.pack);
		//threshold the value to reduce the influence of large gradient
		__m128 xmm_com = _mm_cmpgt_ps(xmm_fea, xmm_th);

		xmm_fea        = _mm_or_ps(_mm_and_ps(xmm_com, xmm_th), _mm_andnot_ps(xmm_com, xmm_fea));
		_mm_storeu_ps(pFea+i, xmm_fea);
		xmm_sum2.pack  = _mm_add_ps(xmm_sum2.pack, _mm_sqr_ps(xmm_fea));
	}
	sum2 = xmm_sum2.f[0]+xmm_sum2.f[1]+xmm_sum2.f[2]+xmm_sum2.f[3];
#endif

	sum2 = 1/sqrt(sum2 + FLT_EPSILON);
	if( sum != sum2 )
	{
		//normalize again
		for(i = 0; i < N; i++ ) 
			pFea[i] *= sum2;
	}

	// convert floating-point descriptor to integer valued descriptor
	if( bQuantize )
	{  
		for(i = 0; i < N; i++)
		{
			int val = (int)(512.0 * pFea[i]);
			pFea[i] = min(255, val);
		}
	}

	return 1;
}

#endif
