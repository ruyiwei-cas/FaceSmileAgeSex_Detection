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

#include <stdio.h>
#include "opencv/highgui.h"

#include "cxfaceutil.hpp"

//#define __SSE2__

#ifndef max
#define max(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif

#if defined (__GNUC__) && !defined (_MM_ALIGN16)
  #define _MM_ALIGN16 __attribute__ ((aligned(16)))
#endif

#ifdef _OPENMP
#include <omp.h>
#endif // end _OPENMP

#ifdef __SSE2__
#include <xmmintrin.h>  // SSE
#include <emmintrin.h>  // SSE2

union F128 
{
	__m128 pack;
	float part[4];
};

union I128 
{
	__m128i pack;
	int part[4];
};
#endif // end __SSE2__

void cxDrawCrossPoint( IplImage *img, CvPoint pt, int thickness )
{
	cvLine( img, cvPoint(pt.x-2, pt.y ),
		cvPoint(pt.x+2, pt.y), CV_RGB(0,255,0), thickness );
	cvLine( img, cvPoint(pt.x, pt.y-2 ),
		cvPoint(pt.x, pt.y+2), CV_RGB(0,255,0), thickness );
}


void cxDrawCaption( IplImage *img, CvFont *pFont, char* sCaption )
{
	if( sCaption != NULL )
	{
		cvRectangle(img, cvPoint(0,(img->height-25)), cvPoint(img->width-1,img->height-1),
			CV_RGB(70,70, 255), CV_FILLED);
			//CV_RGB(100,100, 255), CV_FILLED);
		cvPutText( img, sCaption, cvPoint(5, (img->height-10)), pFont, CV_RGB(255,255,255) );
	}
}


void cxDrawFaceRect(IplImage *img, CvRect rect,  CvScalar cl_face /*CV_RGB(0,255,0)*/)
{
	int thickness;
	if(img->width < 500)  
		thickness =1;
	else
		thickness =2;

	int len = MAX(10, rect.width/3);
	cvLine( img, cvPoint(rect.x, rect.y),
		cvPoint(rect.x+ len, rect.y),
		cl_face, thickness);
	cvLine(img, cvPoint(rect.x, rect.y),
		cvPoint(rect.x, rect.y +len),
		cl_face, thickness);

	cvLine(img, cvPoint(rect.x+rect.width, rect.y),
		cvPoint(rect.x+rect.width-len, rect.y),
		cl_face, thickness);
	cvLine(img, cvPoint(rect.x+rect.width, rect.y),
		cvPoint(rect.x+rect.width, rect.y +len),
		cl_face, thickness);

	cvLine(img, cvPoint(rect.x, rect.y+rect.height),
		cvPoint(rect.x, rect.y+rect.height-len),
		cl_face, thickness);
	cvLine(img, cvPoint(rect.x, rect.y+rect.height),
		cvPoint(rect.x+len, rect.y+rect.height),
		cl_face, thickness);

	cvLine(img, cvPoint(rect.x+rect.width, rect.y+rect.height),
		cvPoint(rect.x+rect.width, rect.y+rect.height-len),
		cl_face, thickness);
	cvLine(img, cvPoint(rect.x+rect.width, rect.y+rect.height),
		cvPoint(rect.x+rect.width-len, rect.y+rect.height),
		cl_face, thickness);
}

/*
 *   1 x x 2      
 * x x  x   x x 
 * x x  x   x x
 *   3 x x 4
 * 
*/
void cxDrawFaceRectWithView(IplImage *img, CvRect rect,
	int vid /* = VIEW_ANGLE_FRONTAL */, CvScalar cl_face /* = CV_RGB */)
{
	int thickness;
	if(img->width < 500)  
		thickness =1;
	else
		thickness =2;

	int len = MAX(10, rect.width/3);
	int anglen = MAX(10, rect.width/4);

	switch (vid)
	{
	case 0:		// 90 degree
		cvLine( img, cvPoint(rect.x, rect.y),
			cvPoint(rect.x+ len, rect.y),
			cl_face, thickness);
		cvLine(img, cvPoint(rect.x, rect.y),
			cvPoint(rect.x, rect.y +len),
			cl_face, thickness);

		cvLine(img, cvPoint(rect.x+rect.width, rect.y),
			cvPoint(rect.x+rect.width-len, rect.y),
			cl_face, thickness);
		cvLine(img, cvPoint(rect.x+rect.width, rect.y),
			cvPoint(rect.x+rect.width, rect.y +len),
			cl_face, thickness);

		cvLine(img, cvPoint(rect.x, rect.y+rect.height),
			cvPoint(rect.x, rect.y+rect.height-len),
			cl_face, thickness);
		cvLine(img, cvPoint(rect.x, rect.y+rect.height),
			cvPoint(rect.x+len, rect.y+rect.height),
			cl_face, thickness);

		cvLine(img, cvPoint(rect.x+rect.width, rect.y+rect.height),
			cvPoint(rect.x+rect.width, rect.y+rect.height-len),
			cl_face, thickness);
		cvLine(img, cvPoint(rect.x+rect.width, rect.y+rect.height),
			cvPoint(rect.x+rect.width-len, rect.y+rect.height),
			cl_face, thickness);
		break;
	case 1:		// 45 degree
		cvLine( img, cvPoint(rect.x, rect.y),
			cvPoint(rect.x+len, rect.y),
			cl_face, thickness);
		cvLine(img, cvPoint(rect.x, rect.y),
			cvPoint(rect.x, rect.y +len),
			cl_face, thickness);

		cvLine(img, cvPoint(rect.x+rect.width, rect.y),
			cvPoint(rect.x+rect.width-len, rect.y),
			cl_face, thickness);
		cvLine(img, cvPoint(rect.x+rect.width, rect.y),
			cvPoint(rect.x+rect.width-anglen, rect.y +len),
			cl_face, thickness);

		cvLine(img, cvPoint(rect.x, rect.y+rect.height),
			cvPoint(rect.x, rect.y+rect.height-len),
			cl_face, thickness);
		cvLine(img, cvPoint(rect.x, rect.y+rect.height),
			cvPoint(rect.x+len, rect.y+rect.height),
			cl_face, thickness);

		cvLine(img, cvPoint(rect.x+rect.width, rect.y+rect.height),
			cvPoint(rect.x+rect.width-anglen, rect.y+rect.height-len),
			cl_face, thickness);
		cvLine(img, cvPoint(rect.x+rect.width, rect.y+rect.height),
			cvPoint(rect.x+rect.width-len, rect.y+rect.height),
			cl_face, thickness);
		break;
	case 2: 		// 135 degree
		cvLine( img, cvPoint(rect.x, rect.y),
			cvPoint(rect.x+len, rect.y),
			cl_face, thickness);
		cvLine(img, cvPoint(rect.x, rect.y),
			cvPoint(rect.x+anglen, rect.y +len),
			cl_face, thickness);

		cvLine(img, cvPoint(rect.x+rect.width, rect.y),
			cvPoint(rect.x+rect.width-len, rect.y),
			cl_face, thickness);
		cvLine(img, cvPoint(rect.x+rect.width, rect.y),
			cvPoint(rect.x+rect.width, rect.y +len),
			cl_face, thickness);

		cvLine(img, cvPoint(rect.x, rect.y+rect.height),
			cvPoint(rect.x+anglen, rect.y+rect.height-len),
			cl_face, thickness);
		cvLine(img, cvPoint(rect.x, rect.y+rect.height),
			cvPoint(rect.x+len, rect.y+rect.height),
			cl_face, thickness);

		cvLine(img, cvPoint(rect.x+rect.width, rect.y+rect.height),
			cvPoint(rect.x+rect.width, rect.y+rect.height-len),
			cl_face, thickness);
		cvLine(img, cvPoint(rect.x+rect.width, rect.y+rect.height),
			cvPoint(rect.x+rect.width-len, rect.y+rect.height),
			cl_face, thickness);
		break;
	case 3:		// 0 degree
		cvLine( img, cvPoint(rect.x, rect.y),
			cvPoint(rect.x+ len, rect.y),
			cl_face, thickness);
		cvLine(img, cvPoint(rect.x, rect.y),
			cvPoint(rect.x-anglen, rect.y +len),
			cl_face, thickness);

		cvLine(img, cvPoint(rect.x+rect.width, rect.y),
			cvPoint(rect.x+rect.width-len, rect.y),
			cl_face, thickness);
		cvLine(img, cvPoint(rect.x+rect.width, rect.y),
			cvPoint(rect.x+rect.width-anglen, rect.y +len),
			cl_face, thickness);

		cvLine(img, cvPoint(rect.x, rect.y+rect.height),
			cvPoint(rect.x-anglen, rect.y+rect.height-len),
			cl_face, thickness);
		cvLine(img, cvPoint(rect.x, rect.y+rect.height),
			cvPoint(rect.x+len, rect.y+rect.height),
			cl_face, thickness);

		cvLine(img, cvPoint(rect.x+rect.width, rect.y+rect.height),
			cvPoint(rect.x+rect.width-anglen, rect.y+rect.height-len),
			cl_face, thickness);
		cvLine(img, cvPoint(rect.x+rect.width, rect.y+rect.height),
			cvPoint(rect.x+rect.width-len, rect.y+rect.height),
			cl_face, thickness);
		break;
	case 4:		// 180 degree
		cvLine( img, cvPoint(rect.x, rect.y),
			cvPoint(rect.x+ len, rect.y),
			cl_face, thickness);
		cvLine(img, cvPoint(rect.x, rect.y),
			cvPoint(rect.x+anglen, rect.y +len),
			cl_face, thickness);

		cvLine(img, cvPoint(rect.x+rect.width, rect.y),
			cvPoint(rect.x+rect.width-len, rect.y),
			cl_face, thickness);
		cvLine(img, cvPoint(rect.x+rect.width, rect.y),
			cvPoint(rect.x+rect.width+anglen, rect.y +len),
			cl_face, thickness);

		cvLine(img, cvPoint(rect.x, rect.y+rect.height),
			cvPoint(rect.x+len, rect.y+rect.height-len),
			cl_face, thickness);
		cvLine(img, cvPoint(rect.x, rect.y+rect.height),
			cvPoint(rect.x+anglen, rect.y+rect.height),
			cl_face, thickness);

		cvLine(img, cvPoint(rect.x+rect.width, rect.y+rect.height),
			cvPoint(rect.x+rect.width+anglen, rect.y+rect.height-len),
			cl_face, thickness);
		cvLine(img, cvPoint(rect.x+rect.width, rect.y+rect.height),
			cvPoint(rect.x+rect.width-len, rect.y+rect.height),
			cl_face, thickness);
		break;
	default:// 90 degree
		cvLine( img, cvPoint(rect.x, rect.y),
			cvPoint(rect.x+ len, rect.y),
			cl_face, thickness);
		cvLine(img, cvPoint(rect.x, rect.y),
			cvPoint(rect.x, rect.y +len),
			cl_face, thickness);

		cvLine(img, cvPoint(rect.x+rect.width, rect.y),
			cvPoint(rect.x+rect.width-len, rect.y),
			cl_face, thickness);
		cvLine(img, cvPoint(rect.x+rect.width, rect.y),
			cvPoint(rect.x+rect.width, rect.y +len),
			cl_face, thickness);

		cvLine(img, cvPoint(rect.x, rect.y+rect.height),
			cvPoint(rect.x, rect.y+rect.height-len),
			cl_face, thickness);
		cvLine(img, cvPoint(rect.x, rect.y+rect.height),
			cvPoint(rect.x+len, rect.y+rect.height),
			cl_face, thickness);

		cvLine(img, cvPoint(rect.x+rect.width, rect.y+rect.height),
			cvPoint(rect.x+rect.width, rect.y+rect.height-len),
			cl_face, thickness);
		cvLine(img, cvPoint(rect.x+rect.width, rect.y+rect.height),
			cvPoint(rect.x+rect.width-len, rect.y+rect.height),
			cl_face, thickness);
		break;
	}
}

void cxDrawFaceBlob( IplImage *img, CvFont* pFont, int id, CvRect rect, CvPoint2D32f *landmark6, float probSmile, 
					int bBlink, int bSmile, int bGender, int nAgeID, char* sFaceName, char* sCaption,
					IplImage *pImgSmileBGR /*=NULL*/, IplImage *pImgSmileBGRA /*=NULL*/, IplImage * pImgSmileMask /*=NULL*/)
{
	int thickness = 1;

	if(img->width < 500) 
		thickness =1;
	else
		thickness = 2;

	// draw face rect
	std::string   sGender = "";
	CvScalar cl_face = CV_RGB(0,255,0);
	if(bGender !=0)
	{
		if(bGender >0)
		{
			sGender = "F";
			cl_face = CV_RGB(255,0,0); //female
		}
		else
		{
			sGender = "M";
			cl_face = CV_RGB(0,0,255);
		}
	}

	cxDrawFaceRect(img, rect, cl_face);

	//draw landmarks
	CvPoint leftEye_l, leftEye_r, rightEye_l, rightEye_r, mouth_l, mouth_r;

	if( landmark6 != NULL )
	{
		leftEye_l  = cvPointFrom32f(landmark6[0]);
		leftEye_r  = cvPointFrom32f(landmark6[1]);
		rightEye_l = cvPointFrom32f(landmark6[2]);
		rightEye_r = cvPointFrom32f(landmark6[3]);
		mouth_l    = cvPointFrom32f(landmark6[4]);
		mouth_r    = cvPointFrom32f(landmark6[5]);
		//CvPoint nose     = landmark6[ ];

		CvPoint leye       = cvPoint( (leftEye_l.x+leftEye_r.x)/2,   (leftEye_l.y+leftEye_r.y)/2 );
		CvPoint reye       = cvPoint( (rightEye_l.x+rightEye_r.x)/2, (rightEye_l.y+rightEye_r.y)/2 );

		cxDrawCrossPoint( img, leye, 1 );
		cxDrawCrossPoint( img, reye, 1 );
		
		//cxDrawCrossPoint( img, nose, 1 );
		cxDrawCrossPoint( img, mouth_l, 1 );
		cxDrawCrossPoint( img, mouth_r, 1 );

		//draw blink
		if(bBlink !=0)
		{
			CvSize   ax_eye = bBlink >0 ? cvSize(12,5) : cvSize(5,5);
			CvScalar cl_eye = bBlink>0 ? CV_RGB(255,0,0) : CV_RGB(0,0,255);
			CvPoint  pt_eye = cvPoint( 
				cvRound(0.5*(leftEye_l.x + leftEye_r.x)),
				cvRound(0.5*(leftEye_l.y + leftEye_r.y)) );
			cvEllipse( img, pt_eye, ax_eye, 0, 0, 360, cl_eye, 1, CV_AA );

			pt_eye = cvPoint( 
				cvRound(0.5*(rightEye_l.x + rightEye_r.x)),
				cvRound(0.5*(rightEye_l.y + rightEye_r.y)) );
			cvEllipse( img, pt_eye, ax_eye, 0, 0, 360, cl_eye, 1, CV_AA );
		}
	}
	
	//draw smile
	if( probSmile >= 0.1 )
	{
		// draw smile probability score
		cvLine( img, cvPoint(rect.x+rect.width+5, rect.y+rect.height), 
			cvPoint(rect.x+rect.width+5, int(rect.y+(1-probSmile)*rect.height)), CV_RGB(250,150, 0), 2 );
	}

	// draw age 
	if( nAgeID < 0 ) nAgeID = 4;
	char sAge[5][256] ={"Baby", "Kid", "Adult", "Senior", ""};

	// draw id (face name)
	static CvScalar colormap[] = {
		{{0, 0, 255}}, {{128, 128, 255}},
		{{255, 0, 0}}, {{255, 128, 128}},
		{{0, 255, 0}}, {{128, 255, 128}},
		{{0, 0, 192}}, {{96, 96, 192}},
		{{192, 0, 0}}, {{192, 96, 96}},
		{{0, 192, 0}}, {{96, 128, 96}},
		{{0, 0, 128}}, {{64, 64, 128}},
		{{128, 0, 0}}, {{128, 64, 64}},
		{{0, 128, 0}}, {{64, 128, 64}}};

		CvScalar colorLabel = colormap[id%18];
		colorLabel = cvScalar(100,88,255); //cvScalar(138,128,255);

		char text[256];
		if(id >=0 )
			sprintf( text, "%d %s %s", id, sGender.c_str(), sAge[nAgeID]);
		else
			sprintf( text, "%s %s", sGender.c_str(), sAge[nAgeID]);
		//sprintf( text, "%d %s", id, sGender.c_str());
		cvPutText( img, text, cvPoint(rect.x, (rect.y-3)), pFont,  colorLabel);

		if(sFaceName != NULL)
		{
			cvRectangle(img, cvPoint(rect.x, rect.y + rect.height + 2),
				cvPoint(rect.x + max(rect.width, 72), rect.y + rect.height + 19),
				colorLabel, CV_FILLED);
			cvPutText(img, sFaceName,
				cvPoint(rect.x + 3, rect.y + rect.height + 14),
				pFont, cvScalar(255, 255, 255));
		}

		//drawFaceIcon
		if( pImgSmileBGR != NULL )
		{
			if(bGender< 0)  bGender= 0;
			if(bSmile < 0 ) bSmile = 0;

			if(nAgeID < 0  || nAgeID >3 ) nAgeID = 2;
			if(nAgeID == 0 ) nAgeID = 1;
			nAgeID = nAgeID-1;

			int imgID = nAgeID*4+bGender*2+bSmile;
			int imgSize = pImgSmileBGR->height;
			CvRect rectROI = cvRect( rect.x+rect.width+6, rect.y+rect.height-imgSize+1, 
				                     imgSize, imgSize );
			CvRect imgROI = cvRect(imgSize*imgID,0, imgSize, imgSize);

			if(pImgSmileBGR->width < 40 ) //only smile mode without age
			{
				if(bSmile)
					imgROI = cvRect(0,0, imgSize, imgSize);	
				else //disable draw smile icon by setting invalid rectROI
					rectROI = cvRect(img->width, img->height, img->width, img->height);
			}

			if( (rectROI.x + rectROI.width) < img->width 
				&& (rectROI.y + rectROI.height) < img->height )
			{

				cvSetImageROI( img, rectROI );
				cvSetImageROI( pImgSmileMask, imgROI );
				if( img->nChannels == 4 )
				{
					cvSetImageROI( pImgSmileBGRA, imgROI );
					cvCopy( pImgSmileBGRA, img, pImgSmileMask );
					cvResetImageROI( pImgSmileBGRA);
				}
				else if( img->nChannels == 3 )
				{
					cvSetImageROI( pImgSmileBGR, imgROI );
					cvCopy( pImgSmileBGR, img, pImgSmileMask );
				}

				cvResetImageROI( img );
				cvResetImageROI( pImgSmileMask);
			}
		}

		// video processing speed info
		if( sCaption != NULL )
		{
			cxDrawCaption(img, pFont, sCaption);
			//cvLine( img, cvPoint(0,(img->height-8)), cvPoint(img->width, (img->height-8)), CV_RGB(100,100, 255), 32 );
			//cvPutText( img, sCaption, cvPoint(5, (img->height-10)), pFont, CV_RGB(255,255,255) );
		}
}

void Img2RGB(const IplImage *pImg, BYTE*& RGB, int& NX, int& NY, int& nChannel)
{
	nChannel = pImg->nChannels;
	int depth = pImg->depth;
	NX = pImg->width;
	NY = pImg->height;

	long imgsize = NY*NX*nChannel;
	RGB = (BYTE *) cvAlloc(imgsize*sizeof(BYTE));

	// to make sure image is shown in horizontal way
	if( nChannel == 3 )
	{
		for(int y=0; y<NY; ++y)
		{
			for(int x=0; x<NX; ++x)
			{
				// OpenCV IplImage in BGR order
				RGB[y*NX*3 + 3*x] = CV_IMAGE_ELEM(pImg, BYTE, y, x*3+2);
				RGB[y*NX*3 + 3*x+1] = CV_IMAGE_ELEM(pImg, BYTE, y, x*3+1);
				RGB[y*NX*3 + 3*x+2] = CV_IMAGE_ELEM(pImg, BYTE, y, x*3);
			}
		}
	}
	else
	{
		for(int y=0; y<NY; ++y)
		{
			for(int x=0; x<NX; ++x)
			{
				RGB[y*NX + x] = CV_IMAGE_ELEM(pImg, BYTE, y, x);
			}
		}
	}
}

void Img2RGB(const char* name, BYTE*& RGB, int& NX, int& NY, int& nChannel)
{
	IplImage* pImg = cvLoadImage(name, 0);
	if( pImg == NULL )
	{
		//printf("Load Error!\n");
		exit(-1);
	}
	Img2RGB(pImg, RGB, NX, NY, nChannel);
	cvReleaseImage(&pImg);
}

inline float bilinear_lerp(IplImage* arr, int w, float x, float y, int nch=1, int ith=0)
{
	int mnx = (int)x;
	int mny = (int)y;
	int mxx = mnx + 1;
	int mxy = mny + 1;

	double alfa = mxx - x;
	double beta = mxy - y;
	double TH1 = 1e-4;
	double TH2 = 1.0 - TH1;
/*
	int fxy = CV_IMAGE_ELEM(arr, uchar, mxy, nch*mxx + ith);
	int fx1y = CV_IMAGE_ELEM(arr, uchar, mxy, nch*mnx + ith);
	int fxy1 = CV_IMAGE_ELEM(arr, uchar, mny, nch*mxx + ith);
	int fx1y1 = CV_IMAGE_ELEM(arr, uchar, mny, nch*mnx + ith);
*/
	
	int fxy = 0;
	int fx1y = 0;
	int fxy1 = 0;
	int fx1y1 = 0;
	if((mxy<arr->height) && (mxx<arr->width))
	{
		fxy = CV_IMAGE_ELEM(arr, uchar, mxy, nch*mxx + ith);
		fx1y = CV_IMAGE_ELEM(arr, uchar, mxy, nch*mnx + ith);
		fxy1 = CV_IMAGE_ELEM(arr, uchar, mny, nch*mxx + ith);
		fx1y1 = CV_IMAGE_ELEM(arr, uchar, mny, nch*mnx + ith);
	}

	if( alfa < TH1 ) 
		return float(beta * fxy1 + (1-beta) * fxy);

	if( alfa > TH2 )
		return float(beta * fx1y1 + (1-beta) * fx1y);

	if( beta < TH1 ) 
		return float(alfa * fx1y + (1-alfa) * fxy);	

	if( beta > TH2 )
		return float(alfa * fx1y1 + (1-alfa) * fxy1);

	return float( beta*(alfa * fx1y1 + (1-alfa)*fxy1) + (1-beta)*(alfa*fx1y + (1-alfa)*fxy) );	
}

void icvWarpAffineROI(IplImage* pRawImg, IplImage* pDstImg, CvMat* pMapmat, CvRect srcRect, int border)
{
	if( pRawImg == NULL || pDstImg == NULL || pMapmat == NULL )
		return;

	int w, h, nChannel;
	w = pRawImg->width;
	h = pRawImg->height;
	nChannel = pRawImg->nChannels;
	border = max(0, border);	// extended border	

	// inverse map-matrix
	double src_matrix[6], dst_matrix[6];
	CvMat srcAb = cvMat( 2, 3, CV_64F, src_matrix ),
		dstAb = cvMat( 2, 3, CV_64F, dst_matrix ),
		A, b, invA, invAb;

	// [R|t] -> [R^-1 | -(R^-1)*t]
	cvConvertScale(pMapmat, &srcAb );
	cvGetCols( &srcAb, &A, 0, 2 );
	cvGetCol( &srcAb, &b, 2 );
	cvGetCols( &dstAb, &invA, 0, 2 );
	cvGetCol( &dstAb, &invAb, 2 );
	cvInvert( &A, &invA, CV_SVD );
	cvGEMM( &invA, &b, -1, 0, 0, &invAb );

	// get bounding-box in src-image: srcRect is the rect in the warpped image
	CvPoint2D32f bnd[4];
	CvPoint2D32f pt;
	float tleft = (float)cvmGet(&dstAb, 0, 0), tmid = (float)cvmGet(&dstAb, 0, 1), tright = (float)cvmGet(&dstAb, 0, 2),
		  bleft = (float)cvmGet(&dstAb, 1, 0), bmid = (float)cvmGet(&dstAb, 1, 1), bright = (float)cvmGet(&dstAb, 1, 2);

	//pt = cvPoint2D32f(srcRect.x, srcRect.y);
	//bnd[0].x = tleft * (pt.x) + tmid * (pt.y) + tright;
	//bnd[0].y = bleft * (pt.x) + bmid * (pt.y) + bright;

	//pt = cvPoint2D32f(srcRect.x+srcRect.width, srcRect.y);
	//bnd[1].x = tleft * (pt.x) + tmid * (pt.y) + tright;
	//bnd[1].y = bleft * (pt.x) + bmid * (pt.y) + bright;

	//pt = cvPoint2D32f(srcRect.x, srcRect.y+srcRect.height);
	//bnd[2].x = tleft * (pt.x) + tmid * (pt.y) + tright;
	//bnd[2].y = bleft * (pt.x) + bmid * (pt.y) + bright;

	//pt = cvPoint2D32f(srcRect.x+srcRect.width, srcRect.y+srcRect.height);
	//bnd[3].x = tleft * (pt.x) + tmid * (pt.y) + tright;
	//bnd[3].y = bleft * (pt.x) + bmid * (pt.y) + bright;

	//bnd is the Rect in the output dst image
	bnd[0] = cvPoint2D32f(srcRect.x, srcRect.y);
	bnd[1] = cvPoint2D32f(srcRect.x+srcRect.width, srcRect.y);
	bnd[2] = cvPoint2D32f(srcRect.x, srcRect.y+srcRect.height);
	bnd[3] = cvPoint2D32f(srcRect.x+srcRect.width, srcRect.y+srcRect.height);
	
	// rect in the before warpped space/src-image
	float minx1, maxx1, miny1, maxy1;
	minx1 = min(min(bnd[0].x, bnd[2].x), min(bnd[1].x, bnd[3].x));
	maxx1 = max(max(bnd[0].x, bnd[2].x), max(bnd[1].x, bnd[3].x));
	miny1 = min(min(bnd[0].y, bnd[2].y), min(bnd[1].y, bnd[3].y));
	maxy1 = max(max(bnd[0].y, bnd[2].y), max(bnd[1].y, bnd[3].y));

	// decide the ROI warp region
	CvRect rc1;
	rc1.x = (int)max(minx1-border, 0);
	rc1.y = (int)max(miny1-border, 0);
	rc1.width  = (int)min(maxx1+border-rc1.x, w-rc1.x);
	rc1.height = (int)min(maxy1+border-rc1.y, h-rc1.y);

	cvSetZero(pDstImg);

#ifdef __SSE2__
	CvPoint2D32f *ptList = (CvPoint2D32f *)_mm_malloc( sizeof(CvPoint2D32f)*rc1.width*rc1.height, 16 );
#else
	CvPoint2D32f *ptList = (CvPoint2D32f *)malloc( sizeof(CvPoint2D32f)*rc1.width*rc1.height);
#endif

	int k=0;
	// only do the WarpAffine in the ROI region
	for(int j=0; j<rc1.height; ++j)
	{
		for(int i=0; i<rc1.width; ++i)
		{
			ptList[k].y = float(rc1.y + j);
			ptList[k].x = float(rc1.x + i);
			++k;	
		}
	}
	int sz = k;	

	// only do the WarpAffine in the ROI region
	if( nChannel == 1 )
	{
		float  eps = (float)1e-4, uneps = 1.0f-eps, wc = w-1.0f-eps, hc = h-1.0f-eps;
#ifdef __SSE2__
		__m128 tleftP		 = _mm_set1_ps( tleft );
		__m128 tmidP		 = _mm_set1_ps( tmid );		
		__m128 bleftP		 = _mm_set1_ps( bleft );
		__m128 bmidP		 = _mm_set1_ps( bmid );
		const __m128 trightP = _mm_set1_ps( tright);
		const __m128 brightP = _mm_set1_ps( bright);

		const __m128 wcP	 = _mm_set1_ps( wc );
		const __m128 hcP	 = _mm_set1_ps( hc );
		const __m128 zeroP   = _mm_set1_ps( 0 );
		const __m128 P1      = _mm_set1_ps(1);
		const __m128 P255    = _mm_set1_ps(255);
		const __m128 epsP	 = _mm_set1_ps( eps );
		const __m128i I1     = _mm_set1_epi32(1);
#endif // end __SSE2__

#ifdef _OPENMP 
	#pragma omp parallel for schedule( guided )
#endif // end _OPENMP

#ifndef __SSE2__  
		for( int k=0; k<sz-3; k+=4 )
		{	
			CvPoint2D32f pts[4];
			float x[4], y[4];
			int val; 

			for(int i=0; i<4; i++)
			{
				pts[i]	= ptList[k+i];

				x[i]	= tleft * pts[i].x + tmid * pts[i].y + tright;
				y[i]	= bleft * pts[i].x + bmid * pts[i].y + bright;

				if(x[i] > wc) x[i] = wc;
				if(x[i] < 0)  x[i] = eps;
				if(y[i] > hc) y[i] = hc;
				if(y[i] < 0)  y[i] = eps;

				val = (int)bilinear_lerp ((IplImage*)pRawImg, w, x[i], y[i]);
				if (val < 0)	   val  = 0;
				if (val > 255)     val  = 255;

				pts[i].x -= rc1.x; 	pts[i].y -= rc1.y; //move to pDstImg's ori point
				CV_IMAGE_ELEM( pDstImg, uchar, (int)(pts[i].y), (int)(pts[i].x) ) = (uchar)val;
			}
		}
#else  // use SSE2
		for( int k=0; k<sz-3; k+=4 )
		{	
			union F128 ptxP, ptyP, xP, yP, valP;

			__m128 lowP  = _mm_load_ps( &ptList[k].x);
			__m128 highP = _mm_load_ps( &ptList[k+2].x);
			ptxP.pack	 = _mm_shuffle_ps( lowP, highP, _MM_SHUFFLE(2,0,2,0) );
			ptyP.pack	 = _mm_shuffle_ps( lowP, highP, _MM_SHUFFLE(3,1,3,1) );			

			xP.pack      = _mm_add_ps( _mm_mul_ps(ptxP.pack, tleftP), _mm_mul_ps(ptyP.pack, tmidP) );
			xP.pack      = _mm_add_ps( xP.pack, trightP );
			yP.pack      = _mm_add_ps( _mm_mul_ps(ptxP.pack, bleftP), _mm_mul_ps(ptyP.pack, bmidP) );
			yP.pack      = _mm_add_ps( yP.pack, brightP );

			__m128 mask1 = _mm_cmpgt_ps( xP.pack, wcP );
			xP.pack		 = _mm_or_ps( _mm_and_ps(mask1, wcP), _mm_andnot_ps(mask1, xP.pack) );
			__m128 mask2 = _mm_cmplt_ps( xP.pack, zeroP );
			xP.pack		 = _mm_or_ps( _mm_and_ps(mask2, epsP), _mm_andnot_ps(mask2, xP.pack) );
			__m128 mask3 = _mm_cmpgt_ps( yP.pack, hcP );
			yP.pack		 = _mm_or_ps( _mm_and_ps(mask3, hcP), _mm_andnot_ps(mask3, yP.pack) );
			__m128 mask4 = _mm_cmplt_ps( yP.pack, zeroP );
			yP.pack		 = _mm_or_ps( _mm_and_ps(mask4, epsP), _mm_andnot_ps(mask4, yP.pack) );

			union I128 mnxP, mnyP, mxxP, mxyP;
			mnxP.pack = _mm_cvttps_epi32( xP.pack );
			mnyP.pack = _mm_cvttps_epi32( yP.pack );

			mxxP.pack = _mm_add_epi32(mnxP.pack, I1);
			mxyP.pack = _mm_add_epi32(mnyP.pack, I1);

			_MM_ALIGN16 int fxy[4], fx1y[4], fxy1[4], fx1y1[4];

			for(int i=0; i<4; i++)
			{
				fxy[i]   = CV_IMAGE_ELEM( pRawImg, uchar, mxyP.part[i], mxxP.part[i] );
				fx1y[i]  = CV_IMAGE_ELEM( pRawImg, uchar, mxyP.part[i], mnxP.part[i] );
				fxy1[i]  = CV_IMAGE_ELEM( pRawImg, uchar, mnyP.part[i], mxxP.part[i] );
				fx1y1[i] = CV_IMAGE_ELEM( pRawImg, uchar, mnyP.part[i], mnxP.part[i] );
			}

			__m128i fxyP   = _mm_load_si128 ((__m128i *)fxy);
			__m128i fx1yP  = _mm_load_si128 ((__m128i *)fx1y);
			__m128i fxy1P  = _mm_load_si128 ((__m128i *)fxy1);
			__m128i fx1y1P = _mm_load_si128 ((__m128i *)fx1y1);

			__m128 alfaP   = _mm_sub_ps( _mm_cvtepi32_ps(mxxP.pack), xP.pack );
			__m128 betaP   = _mm_sub_ps( _mm_cvtepi32_ps(mxyP.pack), yP.pack );
			__m128 unalfaP = _mm_sub_ps( P1, alfaP );
			__m128 unbetaP = _mm_sub_ps( P1, betaP );
			__m128 beta1P  = _mm_add_ps( _mm_mul_ps(alfaP, _mm_cvtepi32_ps(fx1y1P)),
				_mm_mul_ps(unalfaP, _mm_cvtepi32_ps(fxy1P)) );
			__m128 beta0P  = _mm_add_ps( _mm_mul_ps(alfaP, _mm_cvtepi32_ps(fx1yP)), 
				_mm_mul_ps(unalfaP, _mm_cvtepi32_ps(fxyP)) );

			valP.pack	   = _mm_add_ps( _mm_mul_ps(betaP, beta1P), _mm_mul_ps(unbetaP, beta0P) );
			__m128 masklt  = _mm_cmplt_ps( valP.pack, zeroP );
			valP.pack	   = _mm_or_ps( _mm_and_ps(masklt, zeroP), _mm_andnot_ps(masklt, valP.pack) );
			__m128 maskgt  = _mm_cmpgt_ps( valP.pack, P255 );
			valP.pack	   = _mm_or_ps( _mm_and_ps(maskgt, P255), _mm_andnot_ps(maskgt, valP.pack) );

			for(int i=0; i<4; i++)
			{
				int val = (int)valP.part[i];
				//CV_IMAGE_ELEM(pDstImg, uchar, (int)(ptyP.part[i]), (int)(ptxP.part[i])) = (uchar)val;
				CV_IMAGE_ELEM(pDstImg, uchar, (int)(ptyP.part[i])-rc1.y, (int)(ptxP.part[i])-rc1.x) = (uchar)val; //move to pDstImg's ori point
			}
		}
#endif // end __SSE2__

		for( int k=sz-(sz%4); k<sz; k++ )
		{
			CvPoint2D32f pt = ptList[k];

			// (x,y) in the src image
			float x	= tleft * pt.x + tmid * pt.y + tright;
			float y	= bleft * pt.x + bmid * pt.y + bright;

			if(x > wc)  x = wc;
			if(x < 0)	x = eps;
			if(y > hc)  y = hc;
			if(y < 0)   y = eps;

			int val = (int)bilinear_lerp ( (IplImage*)pRawImg, w, x, y );
			if (val < 0)   val = 0;
			if (val > 255) val = 255;

			pt.x -= rc1.x; 	pt.y -= rc1.y;//move to pDstImg's ori point
			CV_IMAGE_ELEM(pDstImg, uchar, (int)pt.y, (int)pt.x) = (uchar)val;	
		}
	}
	else
	{
		for(k=0; k<sz; ++k)
		{
			CvPoint pt = cvPoint( (int)ptList[k].x, (int)ptList[k].y);
			// (x,y) in the src image
			float x = float( cvmGet(&dstAb, 0, 0) * pt.x + cvmGet(&dstAb, 0, 1) * pt.y + cvmGet(&dstAb, 0, 2) );
			float y = float( cvmGet(&dstAb, 1, 0) * pt.x + cvmGet(&dstAb, 1, 1) * pt.y + cvmGet(&dstAb, 1, 2) );

			int val;
			float eps = (float)1e-4;
			if(x > w-1-eps) x = w-1-eps;
			if(x < 0) x = eps;
			if(y > h-1-eps) y = h-1-eps;
			if(y < 0) y = eps;

			pt.x -= rc1.x; 	pt.y -= rc1.y; //move to pDstImg's ori point

			val = (int)bilinear_lerp((IplImage*)pRawImg, w, x, y, 3, 0);
			CV_IMAGE_ELEM(pDstImg, uchar, pt.y, 3*pt.x+0) = (uchar)min(max(val, 0), 255);

			val = (int)bilinear_lerp((IplImage*)pRawImg, w, x, y, 3, 1);
			CV_IMAGE_ELEM(pDstImg, uchar, pt.y, 3*pt.x+1) = (uchar)min(max(val, 0), 255);

			val = (int)bilinear_lerp((IplImage*)pRawImg, w, x, y, 3, 2);
			CV_IMAGE_ELEM(pDstImg, uchar, pt.y, 3*pt.x+2) = (uchar)min(max(val, 0), 255);
		}
	}
#ifdef __SSE2__
		_mm_free( ptList );
#else
		free( ptList );
#endif
}

// specific for AAM/ASM based landmark detector
IplImage* alignFace1(const IplImage *pRawImg, CvPoint2D32f &ptLeftEye,
					CvPoint2D32f &ptRightEye, CvPoint2D32f &ptMouth, 
					int nDstImgW, int nDstImgH, bool bHistEq /*=true*/, float* sclxyud /*=NULL*/, IplImage *pRetImg /*=NULL*/)
{
	if( nDstImgH < 5 || nDstImgW < 5 )
		return NULL;

	float tilted = (float)(atan2(ptRightEye.y-ptLeftEye.y, ptRightEye.x-ptLeftEye.x)*180.0f/CV_PI); //degree
	int w = pRawImg->width;
	int h = pRawImg->height;
	int nChannel = pRawImg->nChannels;
	CvPoint2D32f center;
	center.x = (w-1.0f)/2.0f;
	center.y = (h-1.0f)/2.0f;

	float mapmat[6];
	CvMat pMapmat = cvMat(2, 3, CV_32F, mapmat);
	if (tilted != 0)
	{
		// get rotated face image
		cv2DRotationMatrix( center, tilted, 1.0, &pMapmat);

		CvPoint2D32f pt;
		pt = ptLeftEye;
		ptLeftEye.x  = float(cvmGet(&pMapmat, 0, 0) * (pt.x) + cvmGet(&pMapmat, 0, 1) * (pt.y) + cvmGet(&pMapmat, 0, 2));
		ptLeftEye.y  = float(cvmGet(&pMapmat, 1, 0) * (pt.x) + cvmGet(&pMapmat, 1, 1) * (pt.y) + cvmGet(&pMapmat, 1, 2));

		pt = ptRightEye;
		ptRightEye.x = float(cvmGet(&pMapmat, 0, 0) * (pt.x) + cvmGet(&pMapmat, 0, 1) * (pt.y) + cvmGet(&pMapmat, 0, 2));
		ptRightEye.y = float(cvmGet(&pMapmat, 1, 0) * (pt.x) + cvmGet(&pMapmat, 1, 1) * (pt.y) + cvmGet(&pMapmat, 1, 2));

		pt = ptMouth;
		ptMouth.x    = float(cvmGet(&pMapmat, 0, 0) * (pt.x) + cvmGet(&pMapmat, 0, 1) * (pt.y) + cvmGet(&pMapmat, 0, 2));
		ptMouth.y    = float(cvmGet(&pMapmat, 1, 0) * (pt.x) + cvmGet(&pMapmat, 1, 1) * (pt.y) + cvmGet(&pMapmat, 1, 2));
	}
	float dw = fabs(ptRightEye.x-ptLeftEye.x);
	float dh = fabs(ptLeftEye.y-ptMouth.y);
	float scl_left = 0.5f, scl_right = 0.5f;
	float scl_up = 0.5f, scl_down = 0.5f;
	if( sclxyud )
	{
		scl_left  = min(max(sclxyud[0], 0.1f), 1.0f); 
		scl_right = min(max(sclxyud[1], 0.1f), 1.0f); 
		scl_up    = min(max(sclxyud[2], 0.1f), 1.0f); 
		scl_down  = min(max(sclxyud[3], 0.1f), 1.0f); 
	}

	// according to statistics on 1k faces: dh = (0.77~1.4)dw
	int   xst   = (int)( ptLeftEye.x - dw *scl_left);
	int   xend  = (int)( ptRightEye.x + dw *scl_right );
	int   yst   = (int)(ptLeftEye.y - scl_up *dh);
	int   yend  = (int)(ptMouth.y + scl_down *dh);
	if( xst < 0 || yst < 0 || xend >= w || yend >= h || xend <= xst || yend <= yst )
	{
		//printf("image=(%d, %d), rect=(%d, %d, %d, %d)\n", w, h, xst, yst, xend-xst, yend-yst);
		return NULL;
	}

	CvRect srcRect;	// the rect in the warpped image
	srcRect.x = xst;
	srcRect.y = yst;
	srcRect.width = xend-xst+1;
	srcRect.height = yend-yst+1;
	if( srcRect.width * srcRect.height < 100 )
		return NULL;

	IplImage *pTmpImg = (IplImage *)pRawImg; 
	if( tilted != 0)
	{
		if( 0 ) 
		{
			pTmpImg = cvCloneImage(pRawImg);
			// one hotspot, just need computing a ROI rect
			// here we should refined to speed-up the computation
			cvWarpAffine( pRawImg, pTmpImg, &pMapmat);
		}
		else
		{
			//int borderw = (int) max(5, (srcRect.width* cosf(tilted) + srcRect.height* sinf(tilted)));
			//int borderh = (int) max(5, (srcRect.width* sinf(tilted) + srcRect.height* cosf(tilted)));
			//int border  = 0;

			//// warpping only the ROI region in src-image
			//pTmpImg = cvCloneImage(pRawImg);
			//icvWarpAffineROI((IplImage*)pRawImg, pTmpImg, &pMapmat, srcRect, border);

			int border  = 0;
			pTmpImg = cvCreateImage(cvSize(srcRect.width+border*2, srcRect.height+border*2), pRawImg->depth, pRawImg->nChannels); 
						
			icvWarpAffineROI((IplImage*)pRawImg, pTmpImg, &pMapmat, srcRect, border);
 			srcRect.x = srcRect.y =border;
		}
	}
	if(pRetImg == NULL)
		pRetImg = cvCreateImage(cvSize(nDstImgW, nDstImgH), IPL_DEPTH_8U, nChannel);

	cvSetImageROI(pTmpImg, srcRect);
	if( srcRect.width * srcRect.height > nDstImgW * nDstImgH )
		cvResize(pTmpImg, pRetImg, CV_INTER_AREA);
	else
		cvResize(pTmpImg, pRetImg);

	cvResetImageROI(pTmpImg);
	if( tilted != 0 ) cvReleaseImage(&pTmpImg);

	if( bHistEq && nChannel == 1 )
	{
		// hist-eq here
		IplImage* pImg = cvCloneImage(pRetImg);
		cvEqualizeHist(pImg, pRetImg);
		cvReleaseImage(&pImg);
	}
	return pRetImg;
}

#define SQR(x)  ((x)*(x))
#define CRD(x)  ((x)[0]*(x)[1])
inline float ncc(float v1[], float n1, float v2[], float n2)
{
	if( n1 > FLT_EPSILON && n2 > FLT_EPSILON )
	{
		return ((v1[0]*v2[0] + v1[1]*v2[1])/(n1*n2));
	}
	return -1;
}

// specially for Intel's 6/7-pt + detector, assume that 4 eye-points in a line
IplImage* alignFace2(const IplImage *pRawImg, const CvPoint2D32f pt6s[], CvRect* rc, 
					 int nDstImgW, int nDstImgH, bool bHistEq /*=false*/, float* sclxyud /*=NULL*/, IplImage *pRetImg /*=NULL*/)
{
	if( nDstImgH < 5 || nDstImgW < 5 )
		return NULL;

	IplImage *pFaceImg = (IplImage *)pRawImg;

	// copy 6pts
	CvPoint2D32f landMark6[6];
	for(int i =0; i< 6; i++)
	{
		landMark6[i].x = pt6s[i].x;
		landMark6[i].y = pt6s[i].y;
	}

	// S1: determine the rotation angle
	// [0,1] left-eye, [2,3] right-eye, [4,5] mouth
	const float cos18deg = 0.94f; // cos(19/180*CV_PI);
	const float rad2deg  = float(180.0f/CV_PI);
	float d10[2] = {landMark6[1].x-landMark6[0].x, landMark6[1].y-landMark6[0].y};
	float d32[2] = {landMark6[3].x-landMark6[2].x, landMark6[3].y-landMark6[2].y};
	float d54[2] = {landMark6[5].x-landMark6[4].x, landMark6[5].y-landMark6[4].y};
	float d21[2] = {landMark6[2].x-landMark6[1].x, landMark6[2].y-landMark6[1].y};
	float d31[2] = {landMark6[3].x-landMark6[1].x, landMark6[3].y-landMark6[1].y};
	float d20[2] = {landMark6[2].x-landMark6[0].x, landMark6[2].y-landMark6[0].y};
	float d30[2] = {landMark6[3].x-landMark6[0].x, landMark6[3].y-landMark6[0].y};

	float v01[2] = {d10[1], d10[0]};
	float v12[2] = {d21[1], d21[0]};
	float v23[2] = {d32[1], d32[0]};
	float v45[2] = {d54[1], d54[0]};
	float n01 = sqrtf(SQR(v01[0]) + SQR(v01[1]));
	float n12 = sqrtf(SQR(v12[0]) + SQR(v12[1]));
	float n23 = sqrtf(SQR(v23[0]) + SQR(v23[1]));
	float n45 = sqrtf(SQR(v45[0]) + SQR(v45[1]));
	float ncc0123 = ncc(v01, n01, v23, n23);
	float ncc0145 = ncc(v01, n01, v45, n45);
	float ncc2345 = ncc(v23, n23, v45, n45);
	float sumdxy, sumdx2, tilted = 0;

	// robust least square estimation
	if( ncc0123 >= cos18deg )
	{
		// assert( ncc0123 >= cos18deg );

		//two eyes the same slope direction
		float ncc0112 = ncc(v01, n01, v12, n12);
		float ncc1223 = ncc(v12, n12, v23, n23);
		if( min(ncc0112, ncc1223) >= cos18deg ) 
		{
			// when (01, 23) in the same row
			if( (ncc0145 >=cos18deg) && (ncc2345 >=cos18deg) )
			{
				// all points are inliers
				sumdxy = CRD(d10) + CRD(d32) + CRD(d20) + CRD(d31) + CRD(d30) + CRD(d54);
				sumdx2 = SQR(d10[0]) + SQR(d32[0]) + SQR(d20[0]) + SQR(d31[0]) + SQR(d30[0]) + SQR(d54[0]);
			}
			else
			{
				// (5,4) are outliers
				sumdxy = CRD(d10) + CRD(d32) + CRD(d20) + CRD(d31) + CRD(d30);
				sumdx2 = SQR(d10[0]) + SQR(d32[0]) + SQR(d20[0]) + SQR(d31[0]) + SQR(d30[0]);
			}
		}
		else
		{
			// when (01, 23) in different row
			if( ncc0145 >=cos18deg )
			{
				// pair (01, 23, 45) are inliers
				sumdxy = CRD(d10) + CRD(d32) + CRD(d54);
				sumdx2 = SQR(d10[0]) + SQR(d32[0]) + SQR(d54[0]);
			}
			else
			{
				// (5,4) are outliers
				sumdxy = CRD(d10) + CRD(d32);
				sumdx2 = SQR(d10[0]) + SQR(d32[0]);
			}
		}
		tilted = atan2(sumdxy, sumdx2)*rad2deg;
	}
	else
	{
		//two eyes with quite different slope direction
		// least-square fitting
		sumdxy = CRD(d10) + CRD(d32);
		sumdx2 = SQR(d10[0]) + SQR(d32[0]);

		tilted = atan2(sumdxy, sumdx2)*rad2deg;
	}

	int w = pFaceImg->width;
	int h = pFaceImg->height;
	int nChannel = pFaceImg->nChannels;
	
	CvPoint2D32f center;
	center.x = rc->x+rc->width/2;  //(w-1.0f)/2.0f; //move center to rect center
	center.y = rc->y+rc->height/2; //(h-1.0f)/2.0f;

	// S2: rotated the image if there is rotation angle
	float mapmat[6];
	CvMat pMapmat = cvMat(2, 3, CV_32F, mapmat);
	if(tilted != 0)
	{
		// get rotated face image
		cv2DRotationMatrix( center, tilted, 1.0, &pMapmat);

		CvPoint2D32f pt;
		for(int i=0; i<6; ++i)
		{
			pt = cvPoint2D32f(landMark6[i].x, landMark6[i].y);
			landMark6[i].x = float(cvmGet(&pMapmat, 0, 0) * (pt.x) + cvmGet(&pMapmat, 0, 1) * (pt.y) + cvmGet(&pMapmat, 0, 2));
			landMark6[i].y = float(cvmGet(&pMapmat, 1, 0) * (pt.x) + cvmGet(&pMapmat, 1, 1) * (pt.y) + cvmGet(&pMapmat, 1, 2));
		}
	}

	// S3: compute the eye-centers, mouth-center
	CvPoint2D32f ptLeftEye, ptRightEye, ptMouth;
	ptLeftEye.x  = (landMark6[0].x+landMark6[1].x)/2;
	ptLeftEye.y  = (landMark6[0].y+landMark6[1].y)/2;
	ptRightEye.x = (landMark6[3].x+landMark6[2].x)/2;
	ptRightEye.y = (landMark6[3].y+landMark6[2].y)/2;
	ptMouth.x = (landMark6[4].x+landMark6[5].x)/2;
	ptMouth.y = (landMark6[4].y+landMark6[5].y)/2;

	// S3A: adjust the eye-centers, mouth-center according to eye-distance
	// eye-distance
	float deye1 = fabsf(landMark6[1].x-landMark6[0].x);
	float deye2 = fabsf(landMark6[3].x-landMark6[2].x);

	float dw = fabs(ptRightEye.x-ptLeftEye.x);	// 2 eyes width
	float dh = fabs(ptLeftEye.y-ptMouth.y);
	// quite different eye-distance
	if( deye1 >= 2.0f*deye2 ) // wrong in left eyes
	{
		dw = 2 * deye2;
		ptLeftEye.x = ptRightEye.x - dw;
	}
	if( deye2 >= 2.0f*deye1 ) // wrong in right eyes
	{
		dw = 2 * deye1;
		ptRightEye.x = ptLeftEye.x + dw;
	}
	// mouth position error
	// according to statistics on 1k faces: dh = (0.77~1.4)dw
	if( dh > 1.4f * dw ) // quite good rule
	{
		dh = 1.4f*dw;
		ptMouth.y = ptRightEye.y + dh;
	}
	// too small face rectangle or mouth too beyond
	if( dh < 0.78f*dw )	// this value may impact final results, around 0.8
	{
		dh = 0.78f*dw;
		ptMouth.y = ptRightEye.y + dh;
	}

	// S4: using the standard-face portion to determine the normalized face rectangle
	// 三庭五眼标准
	// height: top => eye-brow, eye-brow => nose; nose => chin, each occupy 1/3.
	// width: five eyes width, we use 4.1 eyes width for smile detection
	// (~0.505, 0.52)
	float scl_left = 0.505f, scl_right = 0.505f;
	float scl_up = 0.52f, scl_down = 0.52f;
	// float scl_up = 0.75f, scl_down = 0.55f;	// for larger face-rect cases
	if( sclxyud )
	{
		scl_left  = min(max(sclxyud[0], 0.1f), 1.0f); 
		scl_right = min(max(sclxyud[1], 0.1f), 1.0f); 
		scl_up    = min(max(sclxyud[2], 0.1f), 1.0f); 
		scl_down  = min(max(sclxyud[3], 0.1f), 1.0f); 
	}

	// according to statistics on 1k faces: dh = (0.77~1.4)dw
	int   xst   = (int)( (ptLeftEye.x - dw *scl_left));
	int   xend  = (int)( (ptRightEye.x + dw *scl_right));
	int   yst   = (int)(ptLeftEye.y - scl_up* dh);
	int   yend  = (int)(ptMouth.y + scl_down* dh);
	if( xst < 0 || yst < 0 || xend >= w || yend >= h || xend <= xst || yend <= yst )
	{
		//printf("image=(%d, %d), rect=(%d, %d, %d, %d)\n", w, h, xst, yst, xend-xst, yend-yst);
		return NULL;
	}

	// obtain the final normalized face ROI
	CvRect srcRect;	// final rect in the warpped image
	srcRect.x = xst;
	srcRect.y = yst;
	srcRect.width = min(xend-xst+1, w);
	srcRect.height = min(yend-yst+1, h);
	/*
	if(srcRect.width>srcRect.height)
		srcRect.width=srcRect.height;
	else srcRect.height=srcRect.width;*/
	if( srcRect.width * srcRect.height < 100 )
		return NULL;

	//ressize big srcRect to dstRect size by pMapmat 
   	if( srcRect.width * srcRect.height > nDstImgW * nDstImgH )
	{
		double x0      = srcRect.x;
		double y0      = srcRect.y;
		double sx      = 1.0f*nDstImgW/srcRect.width;
		double sy      = 1.0f*nDstImgH/srcRect.height;

		srcRect.width  = nDstImgW;
		srcRect.height = nDstImgH;

		double tleft = cvmGet(&pMapmat, 0, 0), tmid = cvmGet(&pMapmat, 0, 1), tright = cvmGet(&pMapmat, 0, 2),
		bleft = cvmGet(&pMapmat, 1, 0), bmid = cvmGet(&pMapmat, 1, 1), bright = cvmGet(&pMapmat, 1, 2);

		tleft = tleft*sx; tmid =tmid*sx; tright = (tright-x0)*sx+x0;
		bleft = bleft*sy; bmid =bmid*sy; bright = (bright-y0)*sy+y0;

		cvmSet(&pMapmat, 0, 0, tleft); cvmSet(&pMapmat, 0, 1, tmid); cvmSet(&pMapmat, 0, 2, tright);
		cvmSet(&pMapmat, 1, 0, bleft); cvmSet(&pMapmat, 1, 1, bmid); cvmSet(&pMapmat, 1, 2, bright);
	}

	IplImage *pTmpImg = (IplImage *)pFaceImg;
	if( tilted != 0)
	{
		if( 0 ) 
		{
			pTmpImg = cvCloneImage(pFaceImg);
			// one hotspot, just need computing a ROI rect
			// here we should refined to speed-up the computation
			cvWarpAffine( pFaceImg, pTmpImg, &pMapmat); 
		}
		else
		{
			int border  = 0;
			pTmpImg = cvCreateImage(cvSize(srcRect.width+border*2, srcRect.height+border*2), pFaceImg->depth, pFaceImg->nChannels); 
						
			icvWarpAffineROI((IplImage*)pFaceImg, pTmpImg, &pMapmat, srcRect, border);
 			srcRect.x = srcRect.y =border;
		}
	}
	if(pRetImg == NULL)
		pRetImg = cvCreateImage(cvSize(nDstImgW, nDstImgH), IPL_DEPTH_8U, nChannel);

	cvSetImageROI(pTmpImg, srcRect);
	if( srcRect.width * srcRect.height > nDstImgW * nDstImgH )
		cvCopy(pTmpImg, pRetImg);
	else
		cvResize(pTmpImg, pRetImg);

	cvResetImageROI(pTmpImg);
	if( tilted != 0 ) cvReleaseImage(&pTmpImg);

	if( bHistEq && nChannel == 1 )
	{
		// hist-eq here
		IplImage* pImg = cvCloneImage(pRetImg);
		cvEqualizeHist(pImg, pRetImg);
		cvReleaseImage(&pImg);
	}
	return pRetImg;
}
// specially for Intel's 6/7-pt + detector, assume that 4 eye-points in a line
IplImage* alignFace3(const IplImage *pRawImg, const CvPoint2D32f pt6s[], CvRect* rc, 
					 int nDstImgW, int nDstImgH, bool bHistEq /*=false*/, float* sclxyud /*=NULL*/, IplImage *pRetImg /*=NULL*/)
{
	if( nDstImgH < 5 || nDstImgW < 5 )
		return NULL;

	IplImage *pFaceImg = (IplImage *)pRawImg;

	// copy 6pts
	CvPoint2D32f landMark6[6];
	for(int i =0; i< 6; i++)
	{
		landMark6[i].x = pt6s[i].x;
		landMark6[i].y = pt6s[i].y;
	}

	// S1: determine the rotation angle
	// [0,1] left-eye, [2,3] right-eye, [4,5] mouth
	const float cos18deg = 0.94f; // cos(19/180*CV_PI);
	const float rad2deg  = float(180.0f/CV_PI);
	float d10[2] = {landMark6[1].x-landMark6[0].x, landMark6[1].y-landMark6[0].y};
	float d32[2] = {landMark6[3].x-landMark6[2].x, landMark6[3].y-landMark6[2].y};
	float d54[2] = {landMark6[5].x-landMark6[4].x, landMark6[5].y-landMark6[4].y};
	float d21[2] = {landMark6[2].x-landMark6[1].x, landMark6[2].y-landMark6[1].y};
	float d31[2] = {landMark6[3].x-landMark6[1].x, landMark6[3].y-landMark6[1].y};
	float d20[2] = {landMark6[2].x-landMark6[0].x, landMark6[2].y-landMark6[0].y};
	float d30[2] = {landMark6[3].x-landMark6[0].x, landMark6[3].y-landMark6[0].y};

	float v01[2] = {d10[1], d10[0]};
	float v12[2] = {d21[1], d21[0]};
	float v23[2] = {d32[1], d32[0]};
	float v45[2] = {d54[1], d54[0]};
	float n01 = sqrtf(SQR(v01[0]) + SQR(v01[1]));
	float n12 = sqrtf(SQR(v12[0]) + SQR(v12[1]));
	float n23 = sqrtf(SQR(v23[0]) + SQR(v23[1]));
	float n45 = sqrtf(SQR(v45[0]) + SQR(v45[1]));
	float ncc0123 = ncc(v01, n01, v23, n23);
	float ncc0145 = ncc(v01, n01, v45, n45);
	float ncc2345 = ncc(v23, n23, v45, n45);
	float sumdxy, sumdx2, tilted = 0;

	// robust least square estimation
	if( ncc0123 >= cos18deg )
	{
		// assert( ncc0123 >= cos18deg );

		//two eyes the same slope direction
		float ncc0112 = ncc(v01, n01, v12, n12);
		float ncc1223 = ncc(v12, n12, v23, n23);
		if( min(ncc0112, ncc1223) >= cos18deg ) 
		{
			// when (01, 23) in the same row
			if( (ncc0145 >=cos18deg) && (ncc2345 >=cos18deg) )
			{
				// all points are inliers
				sumdxy = CRD(d10) + CRD(d32) + CRD(d20) + CRD(d31) + CRD(d30) + CRD(d54);
				sumdx2 = SQR(d10[0]) + SQR(d32[0]) + SQR(d20[0]) + SQR(d31[0]) + SQR(d30[0]) + SQR(d54[0]);
			}
			else
			{
				// (5,4) are outliers
				sumdxy = CRD(d10) + CRD(d32) + CRD(d20) + CRD(d31) + CRD(d30);
				sumdx2 = SQR(d10[0]) + SQR(d32[0]) + SQR(d20[0]) + SQR(d31[0]) + SQR(d30[0]);
			}
		}
		else
		{
			// when (01, 23) in different row
			if( ncc0145 >=cos18deg )
			{
				// pair (01, 23, 45) are inliers
				sumdxy = CRD(d10) + CRD(d32) + CRD(d54);
				sumdx2 = SQR(d10[0]) + SQR(d32[0]) + SQR(d54[0]);
			}
			else
			{
				// (5,4) are outliers
				sumdxy = CRD(d10) + CRD(d32);
				sumdx2 = SQR(d10[0]) + SQR(d32[0]);
			}
		}
		tilted = atan2(sumdxy, sumdx2)*rad2deg;
	}
	else
	{
		//two eyes with quite different slope direction
		// least-square fitting
		sumdxy = CRD(d10) + CRD(d32);
		sumdx2 = SQR(d10[0]) + SQR(d32[0]);

		tilted = atan2(sumdxy, sumdx2)*rad2deg;
	}

	int w = pFaceImg->width;
	int h = pFaceImg->height;
	int nChannel = pFaceImg->nChannels;
	
	CvPoint2D32f center;
	center.x = rc->x+rc->width/2;  //(w-1.0f)/2.0f; //move center to rect center
	center.y = rc->y+rc->height/2; //(h-1.0f)/2.0f;

	// S2: rotated the image if there is rotation angle
	float mapmat[6];
	CvMat pMapmat = cvMat(2, 3, CV_32F, mapmat);
	if(tilted != 0)
	{
		// get rotated face image
		cv2DRotationMatrix( center, tilted, 1.0, &pMapmat);

		CvPoint2D32f pt;
		for(int i=0; i<6; ++i)
		{
			pt = cvPoint2D32f(landMark6[i].x, landMark6[i].y);
			landMark6[i].x = float(cvmGet(&pMapmat, 0, 0) * (pt.x) + cvmGet(&pMapmat, 0, 1) * (pt.y) + cvmGet(&pMapmat, 0, 2));
			landMark6[i].y = float(cvmGet(&pMapmat, 1, 0) * (pt.x) + cvmGet(&pMapmat, 1, 1) * (pt.y) + cvmGet(&pMapmat, 1, 2));
		}
	}

	// S3: compute the eye-centers, mouth-center
	CvPoint2D32f ptLeftEye, ptRightEye, ptMouth;
	ptLeftEye.x  = (landMark6[0].x+landMark6[1].x)/2;
	ptLeftEye.y  = (landMark6[0].y+landMark6[1].y)/2;
	ptRightEye.x = (landMark6[3].x+landMark6[2].x)/2;
	ptRightEye.y = (landMark6[3].y+landMark6[2].y)/2;
	ptMouth.x = (landMark6[4].x+landMark6[5].x)/2;
	ptMouth.y = (landMark6[4].y+landMark6[5].y)/2;

	// S3A: adjust the eye-centers, mouth-center according to eye-distance
	// eye-distance
	float deye1 = fabsf(landMark6[1].x-landMark6[0].x);
	float deye2 = fabsf(landMark6[3].x-landMark6[2].x);

	float dw = fabs(ptRightEye.x-ptLeftEye.x);	// 2 eyes width
	float dh = fabs(ptLeftEye.y-ptMouth.y);
	// quite different eye-distance
	if( deye1 >= 2.0f*deye2 ) // wrong in left eyes
	{
		dw = 2 * deye2;
		ptLeftEye.x = ptRightEye.x - dw;
	}
	if( deye2 >= 2.0f*deye1 ) // wrong in right eyes
	{
		dw = 2 * deye1;
		ptRightEye.x = ptLeftEye.x + dw;
	}
	// mouth position error
	// according to statistics on 1k faces: dh = (0.77~1.4)dw
	if( dh > 1.4f * dw ) // quite good rule
	{
		dh = 1.4f*dw;
		ptMouth.y = ptRightEye.y + dh;
	}
	// too small face rectangle or mouth too beyond
	if( dh < 0.78f*dw )	// this value may impact final results, around 0.8
	{
		dh = 0.78f*dw;
		ptMouth.y = ptRightEye.y + dh;
	}

	// S4: using the standard-face portion to determine the normalized face rectangle
	// 三庭五眼标准
	// height: top => eye-brow, eye-brow => nose; nose => chin, each occupy 1/3.
	// width: five eyes width, we use 4.1 eyes width for smile detection
	// (~0.505, 0.52)
	float scl_left = 0.505f, scl_right = 0.505f;
	float scl_up = 0.52f, scl_down = 0.52f;
	// float scl_up = 0.75f, scl_down = 0.55f;	// for larger face-rect cases
	if( sclxyud )
	{
		scl_left  = min(max(sclxyud[0], 0.1f), 1.0f); 
		scl_right = min(max(sclxyud[1], 0.1f), 1.0f); 
		scl_up    = min(max(sclxyud[2], 0.1f), 1.0f); 
		scl_down  = min(max(sclxyud[3], 0.1f), 1.0f); 
	}

	// according to statistics on 1k faces: dh = (0.77~1.4)dw
	int   xst   = (int)( (ptLeftEye.x - dw *scl_left));
	int   xend  = (int)( (ptRightEye.x + dw *scl_right));
	int   yst   = (int)(ptLeftEye.y - scl_up* dh);
	int   yend  = (int)(ptMouth.y + scl_down* dh);
	if( xst < 0 || yst < 0 || xend >= w || yend >= h || xend <= xst || yend <= yst )
	{
		//printf("image=(%d, %d), rect=(%d, %d, %d, %d)\n", w, h, xst, yst, xend-xst, yend-yst);
		return NULL;
	}

	// obtain the final normalized face ROI
	CvRect srcRect;	// final rect in the warpped image
	int nFaceWidth = xend-xst+1;
	int nFaceHeight = yend-yst + 1;
	srcRect.x = max(int(xst - nFaceWidth*0.5), 0);
	srcRect.y = max(int(yst - nFaceHeight*0.5),0);
	srcRect.width = min((xend-xst+1)*2, w);
	srcRect.height = min((yend-yst+1)*2, h);
	if( srcRect.width * srcRect.height < 100 )
		return NULL;

	//ressize big srcRect to dstRect size by pMapmat 
   	if( srcRect.width * srcRect.height > nDstImgW * nDstImgH )
	{
		double x0      = srcRect.x;
		double y0      = srcRect.y;
		double sx      = 1.0f*nDstImgW/srcRect.width;
		double sy      = 1.0f*nDstImgH/srcRect.height;

		srcRect.width  = nDstImgW;
		srcRect.height = nDstImgH;

		double tleft = cvmGet(&pMapmat, 0, 0), tmid = cvmGet(&pMapmat, 0, 1), tright = cvmGet(&pMapmat, 0, 2),
		bleft = cvmGet(&pMapmat, 1, 0), bmid = cvmGet(&pMapmat, 1, 1), bright = cvmGet(&pMapmat, 1, 2);

		tleft = tleft*sx; tmid =tmid*sx; tright = (tright-x0)*sx+x0;
		bleft = bleft*sy; bmid =bmid*sy; bright = (bright-y0)*sy+y0;

		cvmSet(&pMapmat, 0, 0, tleft); cvmSet(&pMapmat, 0, 1, tmid); cvmSet(&pMapmat, 0, 2, tright);
		cvmSet(&pMapmat, 1, 0, bleft); cvmSet(&pMapmat, 1, 1, bmid); cvmSet(&pMapmat, 1, 2, bright);
	}

	IplImage *pTmpImg = (IplImage *)pFaceImg;
	if( tilted != 0)
	{
		if( 0 ) 
		{
			pTmpImg = cvCloneImage(pFaceImg);
			// one hotspot, just need computing a ROI rect
			// here we should refined to speed-up the computation
			cvWarpAffine( pFaceImg, pTmpImg, &pMapmat); 
		}
		else
		{
			int border  = 0;
			pTmpImg = cvCreateImage(cvSize(srcRect.width+border*2, srcRect.height+border*2), pFaceImg->depth, pFaceImg->nChannels); 
			
			icvWarpAffineROI((IplImage*)pFaceImg, pTmpImg, &pMapmat, srcRect, border);
 			srcRect.x = srcRect.y =border;

		}
	}	

	if(pRetImg == NULL)
		pRetImg = cvCreateImage(cvSize(nDstImgW, nDstImgH), IPL_DEPTH_8U, nChannel);

	cvSetImageROI(pTmpImg, srcRect);
	if( srcRect.width * srcRect.height >= nDstImgW * nDstImgH )
		cvCopy(pTmpImg, pRetImg);
	else
		cvResize(pTmpImg, pRetImg);

	cvResetImageROI(pTmpImg);
	if( tilted != 0 ) cvReleaseImage(&pTmpImg);

	if( bHistEq && nChannel == 1 )
	{
		// hist-eq here
		IplImage* pImg = cvCloneImage(pRetImg);
		cvEqualizeHist(pImg, pRetImg);
		cvReleaseImage(&pImg);
	}
	return pRetImg;
}

IplImage* AlignFace6PtAffine(const IplImage *pRawImg, CvPoint2D32f pt6s[], 
					   int nDstImgW, int nDstImgH, float* sclxyud /*=NULL*/, IplImage *pRetImg /*= NULL*/)
{
	if( nDstImgH < 10 || nDstImgW < 10 || pRawImg == NULL )
		return NULL;

	CvPoint2D32f ptLeftEye = cvPoint2D32f((pt6s[0].x+pt6s[1].x)/2, (pt6s[0].y+pt6s[1].y)/2);
	CvPoint2D32f ptRightEye = cvPoint2D32f((pt6s[2].x+pt6s[3].x)/2, (pt6s[2].y+pt6s[3].y)/2);
	CvPoint2D32f ptMouth = cvPoint2D32f((pt6s[4].x+pt6s[5].x)/2, (pt6s[4].y+pt6s[5].y)/2);
	CvPoint2D32f mideye = cvPoint2D32f( (ptLeftEye.x+ptRightEye.x)/2, (ptLeftEye.y+ptRightEye.y)/2 );

	float mapforward[6];
	CvMat pMapmat = cvMat(2, 3, CV_32F, mapforward);
	CvPoint2D32f srcPt[3];
	srcPt[0] = ptLeftEye;
	srcPt[1] = ptRightEye;
	srcPt[2] = ptMouth;

	//////////////////////////////////////////////////////////////////////////
	float scl_left = 0.5f, scl_right = 0.5f;
	float scl_up = 0.5f, scl_down = 0.5f;
	if( sclxyud )
	{
		scl_left = min(max(sclxyud[0], 0.1), 1.0); 
		scl_right = min(max(sclxyud[1], 0.1), 1.0); 
		scl_up = min(max(sclxyud[2], 0.1), 1.0); 
		scl_down = min(max(sclxyud[3], 0.1), 1.0); 
	}
	float dw, dh;
	dw = sqrtf( SQR(ptLeftEye.x-ptRightEye.x) + SQR(ptLeftEye.y-ptRightEye.y) );
	dh = sqrtf( SQR(ptMouth.x-mideye.x) + SQR(ptMouth.y-mideye.y) );
	float facewidth = dw*(1.0f +scl_left + scl_right);
	if( dw * dh < 100 )
		return NULL;

	CvPoint2D32f dstPt[3];
	dstPt[0] = cvPoint2D32f(dw*scl_left, dh*scl_up);
	dstPt[1] = cvPoint2D32f(dw*(1+scl_left), dh*scl_up);
	dstPt[2] = cvPoint2D32f(facewidth/2.0f, dh*(1+scl_up) );
	cvGetAffineTransform(srcPt, dstPt, &pMapmat);

	//////////////////////////////////////////////////////////////////////////
	int w = pRawImg->width;
	int h = pRawImg->height;
	int nChannel = pRawImg->nChannels;
	{
		CvPoint2D32f pt;
		pt = ptLeftEye;
		ptLeftEye.x = cvmGet(&pMapmat, 0, 0) * (pt.x) + cvmGet(&pMapmat, 0, 1) * (pt.y) + cvmGet(&pMapmat, 0, 2);
		ptLeftEye.y = cvmGet(&pMapmat, 1, 0) * (pt.x) + cvmGet(&pMapmat, 1, 1) * (pt.y) + cvmGet(&pMapmat, 1, 2);

		pt = ptRightEye;
		ptRightEye.x = cvmGet(&pMapmat, 0, 0) * (pt.x) + cvmGet(&pMapmat, 0, 1) * (pt.y) + cvmGet(&pMapmat, 0, 2);
		ptRightEye.y = cvmGet(&pMapmat, 1, 0) * (pt.x) + cvmGet(&pMapmat, 1, 1) * (pt.y) + cvmGet(&pMapmat, 1, 2);

		pt = ptMouth;
		ptMouth.x = cvmGet(&pMapmat, 0, 0) * (pt.x) + cvmGet(&pMapmat, 0, 1) * (pt.y) + cvmGet(&pMapmat, 0, 2);
		ptMouth.y = cvmGet(&pMapmat, 1, 0) * (pt.x) + cvmGet(&pMapmat, 1, 1) * (pt.y) + cvmGet(&pMapmat, 1, 2);
	}
	int   xst   = (int)( (ptLeftEye.x - dw *scl_left));
	int   xend  = (int)( (ptRightEye.x + dw *scl_right));
	int   yst   = (int)(ptLeftEye.y - scl_up *dh);
	int   yend  = (int)(ptMouth.y + scl_down *dh);
	int dx = min(max(cvRound((xend - xst)* 0.02), 1), 16);
	int dy = min(max(cvRound((yend - yst)* 0.02), 1), 16);
	if( xst < 0 && xst >= -dx ) xst = 0;
	if( yst < 0 && yst >= -dy ) yst = 0;
	if( xend >= w && xend <= w+dx ) xend = w-1;
	if( yend >= h && yend <= h+dy ) yend = h-1;
	if( xst < 0 || yst < 0 || xend >= w || yend >= h || xend <= xst || yend <= yst )
	{
		//printf("image=(%d, %d), rect=(%d, %d, %d, %d)\n", w, h, xst, yst, xend-xst, yend-yst);
		return NULL;
	}

	CvRect srcRect;	// the rect in the warpped image
	srcRect.x = xst;
	srcRect.y = yst;
	srcRect.width = xend-xst+1;
	srcRect.height = yend-yst+1;
	if( srcRect.width * srcRect.height < 100 )
		return NULL;

	//////////////////////////////////////////////////////////////////////////
	IplImage *pTmpImg = cvCloneImage(pRawImg);
	cvWarpAffine( pRawImg, pTmpImg, &pMapmat);

	if(pRetImg == NULL)
		pRetImg = cvCreateImage(cvSize(nDstImgW, nDstImgH), IPL_DEPTH_8U, nChannel);
	
	cvSetImageROI(pTmpImg, srcRect);
	if( srcRect.width * srcRect.height > nDstImgW * nDstImgH )
		cvResize(pTmpImg, pRetImg, CV_INTER_AREA);
	else
		cvResize(pTmpImg, pRetImg);

	cvResetImageROI(pTmpImg);
	cvReleaseImage(&pTmpImg);

	return pRetImg;
}

//////////////////////////////////////////////////////////////////////////
inline float icvAvg(CvMat* pSrc)
{
	int w, h, x, y;
	w = pSrc->width;
	h = pSrc->height;

	// adjust the dynamic range to the 8-bit interval
	float ave = 0;
	for(y=0; y<h; ++y)
	{
		for(x=0; x<w; ++x)
		{
			float v = (float)cvmGet(pSrc, y, x);
			ave += v;
		}
	}
	return (ave/(w*h));
}

//auto focus FaceImage into image center
void autoFocusFaceImage(IplImage* pSrc, IplImage* pDest, CvRectItem *vFaceRect, int faceNum, float alpha /*= 0.05*/)
{
	//Set face center as the rect with maximum face size
	int faceWidth = -1;
	CvPoint faceCenter = cvPoint(pSrc->width/2, pSrc->height/2);
	for( int i=0; i<faceNum; i++ )
	{
		CvRect rect = vFaceRect[i].rc;
		if(faceWidth < rect.width)
		{
			faceWidth = rect.width;
			faceCenter.x = rect.x+rect.width/2;
			faceCenter.y = rect.y+rect.height/2;
		}
	}

	if(faceCenter.x > pSrc->width*3/4)
		faceCenter.x = pSrc->width*3/4;
	else if(faceCenter.x < pSrc->width/4)
		faceCenter.x = pSrc->width/4;

	if(faceCenter.y > pSrc->height*3/4)
		faceCenter.y = pSrc->height*3/4;
	else if(faceCenter.y < pSrc->height/4)
		faceCenter.y = pSrc->height/4;

	static CvPoint curImgCenter = cvPoint(pSrc->width/2, pSrc->height/2);;
	
	CvPoint offsetPt = cvPoint(faceCenter.x- curImgCenter.x,  faceCenter.y- curImgCenter.y);
	curImgCenter    =  cvPoint( int(curImgCenter.x + offsetPt.x*alpha), int(curImgCenter.y + offsetPt.y*alpha) );

	int halfWidth  = min(curImgCenter.x, pSrc->width  - curImgCenter.x);
	int halfHeight = min(curImgCenter.y, pSrc->height - curImgCenter.y);
	float curScale = 1.0f*halfWidth/halfHeight;
	float oriScale = 1.0f*pSrc->width/pSrc->height;
	
	if(curScale > oriScale)
		halfWidth  = int(halfHeight*oriScale);
	else
		halfHeight = int(halfWidth/oriScale);
		
	CvRect roiRect = cvRect(curImgCenter.x-halfWidth, curImgCenter.y-halfHeight, halfWidth*2, halfHeight*2);

	cvSetImageROI(pSrc, roiRect);
	cvResize(pSrc, pDest);
	cvResetImageROI(pSrc);

}

void readFileList(const char* listName, std::vector<std::string> & fileList)
{
	char buf[2048];
	int dotpos, endpos;
	std::string extname, substr, fname;

	FILE* fp = fopen(listName, "r");
	if( fp == NULL )
	{
		//printf("File open error!\n");
		exit(-1);
	}

	fileList.clear();
	while( !feof(fp) )
	{
		if( fgets(buf, 2047, fp) == NULL )
			break;

		buf[ strlen(buf) - 1]= '\0';
		extname = buf;
		dotpos = (int)extname.find('.');
		if( dotpos == std::string::npos )
			continue;

		substr = extname.substr(dotpos, extname.length() - dotpos);
		endpos = (int)substr.find(' ');
		// without space
		if( endpos == std::string::npos )
		{
			fname = extname;
		}
		else
		{
			// with space
			fname = extname.substr(0, dotpos + endpos);
		}

		fileList.push_back( fname );
	}	
	fclose(fp);
}

IplImage* getThumbnail( IplImage* image, CvRect rect, IplImage* thumbnail /*=NULL*/)
{
	if(image == NULL )
		return false;

	//CvRect subOriRect = cvRect( rect.x- rect.width/4,  rect.y- rect.height*2/3, rect.width*3/2, rect.height*2);
	CvRect subOriRect = cvRect( rect.x- rect.width/2,  rect.y- rect.height*1.2, rect.width*2, rect.height*3);
	if(thumbnail == NULL)
		thumbnail = cvCreateImage(cvSize(subOriRect.width, subOriRect.height), IPL_DEPTH_8U, image->nChannels);

	CvRect subDstRect = cvRect(0,0, thumbnail->width, thumbnail->height);
	cvZero(thumbnail);

	float ratio = subDstRect.width*1.0f/subOriRect.width;
	if(subOriRect.x < 0)
	{
		int oriDx = -subOriRect.x;
		int dstDx = int(oriDx* ratio);
		subOriRect.x = subOriRect.x + oriDx;
		subOriRect.width = subOriRect.width - oriDx;

		subDstRect.x = subDstRect.x + dstDx;
		subDstRect.width = subDstRect.width - dstDx;
	}
	if(subOriRect.y < 0)
	{
		int oriDy = -subOriRect.y;
		int dstDy = int(oriDy* ratio);
		subOriRect.y = subOriRect.y + oriDy;
		subOriRect.height = subOriRect.height - oriDy;

		subDstRect.y = subDstRect.y + dstDy;
		subDstRect.height = subDstRect.height - dstDy;
	}

	if(subOriRect.x + subOriRect.width > image->width)
	{
		int oriDx = -(subOriRect.x + subOriRect.width - image->width);
		int dstDx = int(oriDx* ratio);
		subOriRect.width = subOriRect.width + oriDx;
		subDstRect.width = subDstRect.width + dstDx;
	}

	if(subOriRect.y + subOriRect.height > image->height)
	{
		int oriDy = -(subOriRect.y + subOriRect.height - image->height);
		int dstDy = int(oriDy* ratio);
		subOriRect.height = subOriRect.height + oriDy;
		subDstRect.height = subDstRect.height + dstDy;
	}

	cvSetImageROI(image, subOriRect);
	cvSetImageROI(thumbnail, subDstRect);
	cvResize(image, thumbnail);
	cvResetImageROI(image);
	cvResetImageROI(thumbnail);

	return thumbnail;
}

//////////////////////////////////////////////////////////////////
CxAlignFace::CxAlignFace(int sizeSmallface/*=64*/, int sizeBigface /*= 128*/)
{
	size_bigface   = sizeBigface;
	size_smallface = sizeSmallface;
	//age_sclxyud[4] = {0.505, 0.505, 0.75, 0.55}; 
	age_sclxyud[0] = 0.505f;
	age_sclxyud[1] = 0.505f;
	age_sclxyud[2] = 0.75f;
	age_sclxyud[3] = 0.55f; 

	m_pImgGrayRef     = NULL;
	m_cutface_big     = NULL;
	m_cutface_small   = NULL;

	if(m_cutface_big == NULL)     m_cutface_big    = cvCreateImage(cvSize(size_bigface, size_bigface), IPL_DEPTH_8U, 1);     //aligned face
	if(m_cutface_small == NULL)   m_cutface_small  = cvCreateImage(cvSize(size_smallface, size_smallface), IPL_DEPTH_8U, 1); //resized from cutface_big

	m_bExtCutFaceBig   = false;
	m_bExtCutFaceSmall = false;
}

CxAlignFace::CxAlignFace(IplImage* pGrayImg, CvRect rect, CvPoint2D32f landmark6[])
{
	size_bigface   = 128;
	size_smallface = 64;
	//age_sclxyud[4] = {0.505, 0.505, 0.75, 0.55}; 
	age_sclxyud[0] = 0.505f;
	age_sclxyud[1] = 0.505f;
	age_sclxyud[2] = 0.75f;
	age_sclxyud[3] = 0.55f; 

	if(m_cutface_big == NULL)     m_cutface_big    = cvCreateImage(cvSize(size_bigface, size_bigface), IPL_DEPTH_8U, 1);     //aligned face
	if(m_cutface_small == NULL)   m_cutface_small  = cvCreateImage(cvSize(size_smallface, size_smallface), IPL_DEPTH_8U, 1); //resized from cutface_big

	m_bExtCutFaceBig   = false;
	m_bExtCutFaceSmall = false;

	init(pGrayImg, rect, landmark6);
}

CxAlignFace::~CxAlignFace()
{
	clear();
}

void CxAlignFace::init(IplImage* pGrayImg, CvRect rect, CvPoint2D32f landmark6[])
{
	m_pImgGrayRef = pGrayImg;
	m_rect        = rect;

	for(int i =0; i < 6; i++)
	{
		m_landmark6[i] = landmark6[i];
	}

	alignFace2(m_pImgGrayRef, m_landmark6, &m_rect, size_bigface, size_bigface, false, age_sclxyud, m_cutface_big);
    cvSmooth(m_cutface_big, m_cutface_big);

	m_bExtCutFaceBig   = true;
	m_bExtCutFaceSmall = false;
}

void CxAlignFace::clear()
{
	if(m_cutface_big  )   cvReleaseImage(&m_cutface_big); 
	if(m_cutface_small)   cvReleaseImage(&m_cutface_small); 

	m_cutface_big     = NULL;
	m_cutface_small   = NULL;

	m_bExtCutFaceBig   = false;
	m_bExtCutFaceSmall = false;
}

IplImage* CxAlignFace::getBigCutFace()
{
	if(m_bExtCutFaceBig == false && m_pImgGrayRef)
	{
		alignFace2(m_pImgGrayRef, m_landmark6, &m_rect, size_bigface, size_bigface, false, age_sclxyud, m_cutface_big);
        cvSmooth(m_cutface_big, m_cutface_big);
		m_bExtCutFaceBig = true;
	}
	return m_cutface_big;
}

IplImage* CxAlignFace::getSmallCutFace()
{
	if(m_bExtCutFaceSmall == false && m_pImgGrayRef) 
	{
		if(m_bExtCutFaceBig == false)
		{
			getBigCutFace();
		}

		cvResize(m_cutface_big, m_cutface_small);
		m_bExtCutFaceSmall = true;
	}

	return m_cutface_small;
}