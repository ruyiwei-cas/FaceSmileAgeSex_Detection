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

#ifndef _CX_RECOGNIZERBASE_HPP_
#define _CX_RECOGNIZERBASE_HPP_


#include <opencv/cv.h>
#include "basetypes.hpp"
#include "cxfaceutil.hpp"

// This class abstracts the virtual base class for object recognizer
// Every implementation of a specific object recognizer class should
// inherent from this base class and keep consistent with its interface.

class CxRecognizerBase
{
public: 
	// constructor
	CxRecognizerBase() {};
	// destructor
	virtual ~CxRecognizerBase() {};

	// for load recognizer model
	virtual int    load( const char* path, const char* filename, int cutface_size = 64) = 0;
	
	// get recognizer property
	virtual CvSize getDefCutFaceSize() = 0; // recognizer's default aligned face size
	virtual int    getFeatureDim()     = 0; // recognizer's total feature  dimensition
	virtual int    getFeatureType()    = 0; // recognizer's feature type
	virtual float  getDefThreshold()   = 0; // default recognizer threshold 
	virtual int    getDefRound()       = 0; // default recognizer weak classifier number 

	// recognize
	virtual int    predict(IplImage* pCutFace, float *prob = NULL)  = 0;
	virtual int    predict(float *pFea,  float* retprob = NULL) = 0;
	virtual int    predictDiff(float *pFea1, float *pFea2, float* retprob = NULL) = 0;

	// extracte feature
	virtual void   extFeature(IplImage* pCutFace, float* pFea) = 0;
};

#endif // _CX_RECOGNIZERBASE_HPP_
