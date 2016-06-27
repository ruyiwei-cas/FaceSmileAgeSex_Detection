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

#include <opencv/cv.h>
#include "cxface.hpp"

#include "pthread.h"

#define _MAX_FACES_ 16
#define _MAX_LDMKS_ 8
class CxFaceAnalyzer
{
public:
	CxFaceAnalyzer(EnumViewAngle viewAngle = VIEW_ANGLE_FRONTAL, EnumTrackerType traType = TRA_SURF, int nFaceDetectorNo = 0, 
		const char *str_facesetxml =NULL, int recognizerType = RECOGNIZER_CAS_GLOH,
		bool bEnableAutoCluster = false, bool bLandmarkRegressor = false,
		const char* xml_eye_leftcorner = NULL,
		const char* xml_mth_leftcorner = NULL,
		const char *sFaceRecognizer_modelpath = NULL);



	~CxFaceAnalyzer(void);

	
	//prop_estimate: #fk #fa #fs #mk #ma #ms #smile
	bool Face_Detection(IplImage *pImgSrc,int nMin_FaceSize, char *ThumbnailImgFilename);
	
	void loadFaceModelXML(const char *sPathXML);
	void saveFaceModelXML(const char *sPathXML, vFaceSet *pvecFaceSet =NULL);
	void getMergedFaceSet(vFaceSet& vvClusters, int minWeight = 0);

	char*  getFaceImgDBPath();
	vFaceSet* getFaceSets() { return m_pfaceRecognizer->getFaceSets(); }
	int   getFaceSetID(int nFaceSetIdx) { return m_pfaceRecognizer->getFaceSetID(nFaceSetIdx); }
	int   getFaceSetSize(int nFaceSetIdx = -1); 
	const char* getKeyFacePath(int nFaceSetIdx, int nFaceIdx = -1);

	// output results of faceAnalyze()
	static const int MAX_FACES      = _MAX_FACES_; 
	static const int MAX_LDMKS      = _MAX_LDMKS_; 
	static const int SIZE_BIGFACE   = 128; 
	static const int SIZE_SMALLFACE = 64; 

	CxBoostFaceRecog *getFaceRecognizer() { return m_pfaceRecognizer; }
	IplImage *getBigCutFace() { return m_cutface_big; }

	int         getFaceNum()              { return m_face_num; }
	CvRectItem  getFaceRect(int idx)      { return m_rects[idx];  }
	CvRectItem *getFaceRects()            { return m_rects;  }
    CvPoint2D32f* getFaceLdmks(int idx)    { return m_ldmks[idx]; }

	int       getFaceID(int idx)        { return m_faceID[idx]; } 
	float     getFaceProb(int idx)      { return m_faceProb[idx]; } 
	char     *getFaceName(int idx)      { return m_faceName[idx]; } 

private:
	//input param
	//pthread_t        m_thread_id;
	//threadArgs_face  m_threadArgs;

	IplImage        *m_pImgGray;

	float           m_age_sclxyud[4];
	IplImage		*m_cutface_small;  
	IplImage		*m_cutface_big;  
	IplImage		*m_cutface_big_array[MAX_FACES];  
	IplImage		*m_cutface_small_array[MAX_FACES];
	int				m_cutface_flag[MAX_FACES];

	//output param
	int				 m_face_num;
	CvRectItem  	 m_rects[MAX_FACES];
    CvPoint2D32f     m_ldmks[MAX_FACES][MAX_LDMKS];
	
	int				 m_blink[MAX_FACES];
	int				 m_smile[MAX_FACES];
	int				 m_gender[MAX_FACES];
	int				 m_age[MAX_FACES];
	float            m_smileProb[MAX_FACES];
	float            m_faceProb[MAX_FACES];
	char             m_faceName[MAX_FACES][16];
	int              m_faceID[MAX_FACES];

	int				 m_male_num, m_female_num;
	int              m_baby_num, m_kid_num, m_adult_num, m_senior_num;
	int				 m_male_smile_num, m_female_smile_num;

	bool m_bEnableAutoCluster;

	CxCompDetector   *m_plandmarkDetector;
	CxBoostFaceRecog *m_pfaceRecognizer;

	int m_nFaceDetectorNo;	
};