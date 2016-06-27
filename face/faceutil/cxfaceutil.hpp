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
*	@file		faceutil.hpp
*	@brief		utility function for face analysis
*	@author		Jianguo Li, Intel Labs China
*	copyright reserved 2010, please do not remove this head
*/

#ifndef _FACE_UTIL_HPP
#define _FACE_UTIL_HPP

#include <vector>
#include <string>
#include <algorithm>

#include "opencv/cxcore.h"
#include "opencv/cv.h"
#include "basetypes.hpp"

typedef unsigned char	BYTE;
typedef std::vector<std::string> strArray;

void cxDrawCrossPoint( IplImage *img, CvPoint pt, int thickness =1 );

void cxDrawCaption( IplImage *img, CvFont *pFont, char* sCaption);

void cxDrawFaceRect(IplImage *img, CvRect rect,  CvScalar colors = CV_RGB(0,255,0) );

void cxDrawFaceRectWithView(IplImage *img, CvRect rect,  int vid = VIEW_ANGLE_FRONTAL, CvScalar cl_face = CV_RGB(0,255,0));


void cxDrawFaceBlob(IplImage *img, CvFont* pFont, int id, CvRect rect, CvPoint2D32f *landmark6, float probSmile, 
					int bBlink, int bSmile, int bGender, int nAgeID, char* sFaceName, char* sCaption,
					IplImage *pImgSmileBGR =NULL, IplImage *pImgSmileBGRA =NULL, IplImage * pImgSmileMask =NULL);


inline BYTE RGB2Grey(BYTE r, BYTE g, BYTE b)
{
	return (BYTE)(0.299f* r + 0.587f* g + 0.114f* b);
}

inline BYTE RGB2Grey(BYTE* rgb)
{
	return (BYTE)(0.299f*rgb[0] + 0.587f*rgb[1] + 0.114f*rgb[2]);
}

void Img2RGB(const IplImage *pImg, BYTE*& RGB, int& NX, int& NY, int& nChannel);

void Img2RGB(const char* name, BYTE*& RGB, int& NX, int& NY, int& nChannel);

/* // in AlignFaceX, 'sclxyud' is 4-value vector for determining face-rectangle, default value   
	// sclxyud[0] => scl_left = 0.5, 
	// sclxyud[1] => scl_right = 0.5, 
	// sclxyud[2] => scl_up = 0.5, 
	// sclxyud[3] => scl_down = 0.5
	// the default vector is a tight face-rectangle
	// for larger face-rectangle, possible good choice is (0.51, 0.51, 0.75, 0.55)
*/
// input image should be IPL_DEPTH_8U, may be 1 or 3 channels
// this is specially for AAM/ASM based landmark detector
// only eye-center/mouth-center are available and trusted
IplImage* alignFace1(const IplImage *pRawImg, CvPoint2D32f ptLeftEye,
					 CvPoint2D32f ptRightEye, CvPoint2D32f ptMouth, 
					 int nDstImgW, int nDstImgH, bool bHistEq =false, float* sclxyud = NULL, IplImage *pRetImg = NULL);

// specially for Intel's 6/7-pt + detector, assume that 4 eye-points are in a line
IplImage* alignFace2(const IplImage *pRawImg, const CvPoint2D32f pt6s[], CvRect* rc, 
					 int nDstImgW, int nDstImgH, bool bHistEq =false, float* sclxyud = NULL, IplImage *pRetImg = NULL);
IplImage* alignFace3(const IplImage *pRawImg, const CvPoint2D32f pt6s[], CvRect* rc, 
					 int nDstImgW, int nDstImgH, bool bHistEq /*=false*/, float* sclxyud /*=NULL*/, IplImage *pRetImg /*=NULL*/);
/*
// when rc = NULL, assume there is no tight face rect, only the 6-pts are trusted
// when rc != NULL, assume that the rc is tight and useful, specially for Prof-Ai's 6-pt + detector
IplImage* alignFace3(const IplImage *pRawImg, const CvPoint2D32f pt6s[], CvRect* rc, 
					 int nDstImgW, int nDstImgH, bool bHistEq =false, float* sclxyud = NULL, IplImage *pRetImg = NULL);
	*/				 
IplImage* AlignFace6PtAffine(const IplImage *pRawImg, CvPoint2D32f pt6s[], 
					   int nDstImgW, int nDstImgH, float* sclxyud = NULL, IplImage *pRetImg = NULL);					 

void autoFocusFaceImage(IplImage* pSrc, IplImage* pDest, CvRectItem *vFaceRect, int faceNum, float alpha = 0.05);

void readFileList(const char* listName, std::vector<std::string> &fileList);

IplImage* getThumbnail( IplImage* image, CvRect rect, IplImage* thumbnail = NULL);

class CxAlignFace
{
public:
	CxAlignFace(int sizeSmallface=64, int sizeBigface = 128);
	CxAlignFace(IplImage* pGrayImg, CvRect rect, CvPoint2D32f landmark6[]);
	~CxAlignFace();

	void init(IplImage* pGrayImg, CvRect rect, CvPoint2D32f landmark6[]);
	void clear();

	IplImage *getBigCutFace();
	IplImage *getSmallCutFace();
	int  getBigCutFaceSize() { return size_bigface; }
	int  getSmallCutFaceSize() { return size_smallface; }

protected:	
	int     size_bigface;  
	int     size_smallface;
	float   age_sclxyud[4];

	IplImage     *m_pImgGrayRef;
	CvRect        m_rect;
	CvPoint2D32f  m_landmark6[6];

	//aligned face
	bool      m_bExtCutFaceBig;
	bool      m_bExtCutFaceSmall;

	IplImage *m_cutface_big;
	IplImage *m_cutface_small;
};
#endif
