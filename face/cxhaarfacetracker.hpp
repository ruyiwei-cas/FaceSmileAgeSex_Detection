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

#ifndef _CX_HAARTRACKER_HPP_
#define _CX_HAARTRACKER_HPP_

#include <opencv/cv.h>

//#include "cxfacetracker.hpp"
#include "integrafea.hpp"

typedef int sumtype;
typedef double sqsumtype;

typedef struct CvHidHaarFeature
{
	struct
	{
		sumtype *p0, *p1, *p2, *p3;
		float weight;
	}
	rect[CV_HAAR_FEATURE_MAX];
}
CvHidHaarFeature;

typedef struct CvHidHaarTreeNode
{
	CvHidHaarFeature feature;
	float threshold;
	int left;
	int right;
}
CvHidHaarTreeNode;

typedef struct CvHidHaarClassifier
{
	int count;
	//CvHaarFeature* orig_feature;
	CvHidHaarTreeNode* node;
	float* alpha;
}
CvHidHaarClassifier;

typedef struct CvHidHaarStageClassifier
{
	int  count;
	float threshold;
	CvHidHaarClassifier* classifier;
	int two_rects;

	struct CvHidHaarStageClassifier* next;
	struct CvHidHaarStageClassifier* child;
	struct CvHidHaarStageClassifier* parent;
}
CvHidHaarStageClassifier;

struct CvHidHaarClassifierCascade
{
	int  count;
	int  is_stump_based;
	int  has_tilted_features;
	int  is_tree;
	double inv_window_area;
	CvMat sum, sqsum, tilted;
	CvHidHaarStageClassifier* stage_classifier;
	sqsumtype *pq0, *pq1, *pq2, *pq3;
	sumtype *p0, *p1, *p2, *p3;

	void** ipp_stages;
};
/*
class CxHaarFaceTracker: public CxFaceTracker
{
public:
	CxHaarFaceTracker(const char* filename = NULL);
	~CxHaarFaceTracker();
	
	// init tracker viewAngle, feaType and model file name
	virtual int init(EnumViewAngle viewAngle = VIEW_ANGLE_FRONTAL, char* modePath = NULL);

	// config tracker parameter 
	virtual void config( tagDetectConfig configParam, int level = TR_NLEVEL_2);

	// predict one specific rectangle (currently Haar does not support this)
	virtual int initPredict(IplImage* pImg);
	virtual int predict(IplImage* image, CvRectItem rc, float* prob = NULL);

protected:
	// detect face in a region instead of whole image
	int detect_region( IplImage* image, CvRect rc_src, void* faces );

	int track_region( IplImage* image, CvRect rc_src, int neighbors, void* faces );

protected:
	//haar
	CvHaarClassifierCascade*	haar_cascade;
	CvMatrix mat_sum;                   // integral (sum) image
	CvMatrix mat_sqsum;                 // squared sum image
	CvMatrix mat_tilted;                // tilted sum image
};
*/
#endif
