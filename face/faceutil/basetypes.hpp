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

#pragma once

#include "opencv/cv.h"
#include <vector>
#include <string>


typedef std::vector<int>         iarray;
typedef std::vector<float>       farray;
typedef std::vector<std::string> sarray;
typedef std::vector<float*>      pfarray;
typedef std::vector<IplImage*>   pmarray;

#define FACESET_MAXSIZE         32   //max cluster number of face set clusters
#define FACESETITEM_MAXSIZE     16   //max faces in each face set cluster
#define FACESETCONFLICT_MAXSIZE 16   //max conflict faces in each face set cluster
#define FACESET_DYNAMIC_STORE   1    //dynamically store face set's features and images, otherwise, save when exit or call CFaceListXML save

class FaceSet
{
public:
	FaceSet()  { nFaceSetID = nCurFrameID = nCurTrackID = -1; nFaceSetWeight = 0; }
	
	~FaceSet() { 
		vConflictFaceSetIDs.clear();
		vKeyFaceIDs.clear();
		vKeyFaceNames.clear();
		avgFaceFea.clear();
	}

	void clearFeas() {
		for(int i =0; i < (int)vKeyFaceFeas.size(); i++)
		{
			delete []vKeyFaceFeas[i];
			vKeyFaceFeas[i] = NULL;
		}
		vKeyFaceFeas.clear();

		for(int i =0; i < (int)vKeyFaceImgs.size(); i++)
		{
			cvReleaseImage(&vKeyFaceImgs[i]);
			vKeyFaceImgs[i] = NULL;
		}
	}

	int		    nFaceSetID;
	int		    nFaceSetWeight;
	int         nCurFrameID;
	int         nCurTrackID;
	std::string szFaceSetName;

	iarray	vConflictFaceSetIDs;
	iarray	vKeyFaceIDs;
	sarray	vKeyFaceNames;
	pfarray vKeyFaceFeas;
	pmarray vKeyFaceImgs;
	farray  avgFaceFea;
};

typedef std::vector<FaceSet> vFaceSet;


enum EnumViewAngle{
	// multi-profile detection
	VIEW_ANGLE_0          = 0x00000001,
	VIEW_ANGLE_45         = 0x00000002,
	VIEW_ANGLE_90         = 0x00000004,
	VIEW_ANGLE_135        = 0x00000008,
	VIEW_ANGLE_180        = 0x00000010,

	// multi-roll detection
	VIEW_ROLL_30          = 0x00000020,
	VIEW_ROLL_30N         = 0x00000040,
	VIEW_ROLL_60          = 0x00000080,
	VIEW_ROLL_60N         = 0x00000100,

	VIEW_ANGLE_FRONTAL    = 0x00000004,
	VIEW_ANGLE_HALF_MULTI = 0x0000000E,
	VIEW_ANGLE_MULTI      = 0x0000001F,
	
	VIEW_ANGLE_FRONTALROLL= 0x000001E4,
	VIEW_ANGLE_HALF_MULTI_FRONTALROLL = VIEW_ANGLE_HALF_MULTI | VIEW_ANGLE_FRONTALROLL,
	VIEW_ANGLE_MULTI_FRONTALROLL      = VIEW_ANGLE_MULTI | VIEW_ANGLE_FRONTALROLL,

	VIEW_ANGLE_OMNI       = 0xFFFFFFFF,
};

enum EnumFeaType{
	FEA_HAAR              = 0x00000000,
	FEA_SURF              = 0x00000001,
};

enum EnumTrackerType{
	TRA_HAAR              = 0x00000000,
	TRA_SURF              = 0x00000001,
	TRA_PF	              = 0x00000002,
	TRA_PFMS              = 0x00000003,
};

enum EnumLandmarkerType{
	LDM_6PT               = 0x00000000,
	LDM_7PT               = 0x00000001,
};

enum ThumbnailSize
{
	THUMBNAIL_WIDTH       = 256,
	THUMBNAIL_HEIGHT      = 384,
};

enum EnumRecognizerType{
	RECOGNIZER_BOOST_GB240  = 0,
	RECOGNIZER_BOOST_LBP59  = 1,
	RECOGNIZER_CAS_GLOH     = 2,
};

class CvRectItem
{
public:
	CvRectItem() { prob = 0; rc = cvRect(0,0,0,0); vid = VIEW_ANGLE_90;  angle= 0; neighbors = 0; fid = -1; }

	CvRect             rc;		// region
	float              prob;	// probability
	int          	   vid;		// view-id
	int                fid;     // face-id
	int				   angle;	// roll-angle
	int                neighbors;
	int                reserved[8]; // reserved[0] which stage the rectange in
};

class tagDetectConfig
{
public:
	tagDetectConfig()
	{
		scale_image = 0;
		scanmode    = 2; //1;  //2:default using auto-tuned scan  1: fast mode with low detection rate. 0:standard mode with higher detection rate.
		postfilter  = 0;

		minszx = 40;  minszy = 40;	maxsz = 4096;	//minszx = 24;  minszy = 24; //minszx = 48;  minszy = 48;	24
		step  = 2;
		min_neighbors = 2;     //3 for scanmode = 0;
		overlapratio  = 0.64f; //0.7f;
		zscale = 1.20f;
		roc = 0;
		bsingle_obj   = false;
	}

	int scale_image;	// whether scale the image, current only support scale rectangle
	int scanmode;		// scan mode: 0=> standard, 1=> fast coarse-to-fine
	int postfilter;		// whether perform post-filter or not, default 0

	int minszx;			// min-win-size in x, default 32
	int minszy;			// min-win-size in y, default 32
	int maxsz;			// max-win-size, default 512
	int step;			// slide-window moving step, default 2
	int roc;			// whether output score for roc generating, default 0
	float zscale;		// scale-ratio for scaling rectangle, default 1.2

	//////////////////////////////////////////////////////////////////////////
	int   min_neighbors;// merge when with at least min_neighbors, default 3
	float overlapratio;	// merge with overlap ratio; default 0.7
	bool  bsingle_obj;  // true: detect single object to speed up; false: detect multi objects
};

enum EnumTrackLevel {TR_NLEVEL_1 = 1, TR_NLEVEL_2 = 2, TR_NLEVEL_3 = 3, TR_NLEVEL_4};

// mouse event processing
class CxMouseParam
{
public:
	CxMouseParam () { play =false; rects = NULL; cut_big_face = NULL; image = NULL; faceRecognizer = NULL; }

	bool      play;
	bool      updated;
	int	      face_num;
	CvRectItem *rects;
	IplImage *cut_big_face;
	IplImage *image;

	int       typeRecognizer; //0: cxLibFaceAnalyzer, 1: cxLibFaceRecognizer
	void     *faceRecognizer; //cxLibFaceAnalyzer or cxLibFaceRecognizer by typeRecognizer value

	int       ret_facetrack_id;
	int       ret_faceset_id;
	int       ret_online_collecting;
};



