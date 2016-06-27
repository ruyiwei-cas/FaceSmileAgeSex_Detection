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

#include "cxfaceanalyzer.hpp"

//#pragma comment(lib, "pthreadVC2.lib")


#include "NewFaceDetector\Frontal_Realboosting_Dll.h"

CxFaceAnalyzer::CxFaceAnalyzer(EnumViewAngle viewAngle /*= VIEW_ANGLE_FRONTAL*/, EnumTrackerType traType /*= TRA_SURF*/,
	int nFaceDetectorNo, 
	const char *str_facesetxml /*=NULL*/,
	int  recognizerType /*= RECOGNIZER_CAS_GLOH*/,
	bool bEnableAutoCluster /*=false*/,
	bool bLandmarkRegressor /*= false*/,
	const char* xml_eye_leftcorner,
	const char* xml_mth_leftcorner,
	const char *sFaceRecognizer_modelpath
	)
{
	m_pImgGray = NULL;

	// init face objects
	//	m_pfaceTracker		= NULL;
	m_plandmarkDetector = NULL;
	m_pfaceRecognizer = NULL;

	// set cutface size and image
	m_age_sclxyud[0] = 0.505f;  // {0.505, 0.505, 0.75, 0.55};
	m_age_sclxyud[1] = 0.505f;
	m_age_sclxyud[2] = 0.75f;
	m_age_sclxyud[3] = 0.55f;
	int       size_bigface = SIZE_BIGFACE;
	int       size_smallface = SIZE_SMALLFACE;
	m_cutface_big = cvCreateImage(cvSize(size_bigface, size_bigface), IPL_DEPTH_8U, 1);     //aligned face
	m_cutface_small = cvCreateImage(cvSize(size_smallface, size_smallface), IPL_DEPTH_8U, 1); //resized from cutface_big

	for (int i = 0; i<MAX_FACES; i++)
	{
		m_cutface_big_array[i] = cvCreateImage(cvSize(size_bigface, size_bigface), IPL_DEPTH_8U, 1);     //aligned face
		m_cutface_small_array[i] = cvCreateImage(cvSize(size_smallface, size_smallface), IPL_DEPTH_8U, 1); //resized from cutface_big
		m_cutface_flag[i] = 0;
	}

	// 6pts landmark detector

	std::string xml_eye_lc = "./opencv_input/haarcascade_eye_leftcorner.xml";
	std::string xml_mth_lc = "./opencv_input/haarcascade_mth_leftcorner.xml";
	if (xml_eye_leftcorner != NULL)
		xml_eye_lc = xml_eye_leftcorner;
	if (xml_mth_leftcorner != NULL)
		xml_mth_lc = xml_mth_leftcorner;

	m_plandmarkDetector = new CxCompDetector(xml_eye_lc.c_str(), xml_mth_lc.c_str());

	m_pfaceRecognizer = new CxBoostFaceRecog();

	if (sFaceRecognizer_modelpath == NULL)
	{
		if (recognizerType == RECOGNIZER_BOOST_GB240)
			m_pfaceRecognizer->load(recognizerType, "opencv_input/facemod/Gb240", "face_ann_cascade.mod", SIZE_BIGFACE);
		else if (recognizerType == RECOGNIZER_BOOST_LBP59)
			m_pfaceRecognizer->load(recognizerType, "opencv_input/facemod/LBP59", "face_ann_cascade.mod", SIZE_BIGFACE);
		else if (recognizerType == RECOGNIZER_CAS_GLOH)
			m_pfaceRecognizer->load(recognizerType, "opencv_input/facemod/GLOH", "face_ann_cascade.mod", SIZE_BIGFACE);
	}
	else
	{
		if (recognizerType == RECOGNIZER_BOOST_GB240)
			m_pfaceRecognizer->load(recognizerType, sFaceRecognizer_modelpath, "face_ann_cascade.mod", SIZE_BIGFACE);
		else if (recognizerType == RECOGNIZER_BOOST_LBP59)
			m_pfaceRecognizer->load(recognizerType, sFaceRecognizer_modelpath, "face_ann_cascade.mod", SIZE_BIGFACE);
		else if (recognizerType == RECOGNIZER_CAS_GLOH)
			m_pfaceRecognizer->load(recognizerType, sFaceRecognizer_modelpath, "face_ann_cascade.mod", SIZE_BIGFACE);
	}

	if (str_facesetxml != NULL)
		m_pfaceRecognizer->loadFaceModelXML(str_facesetxml);
	else
		m_pfaceRecognizer->loadFaceModelXML("faceset_model.xml");

	m_bEnableAutoCluster = bEnableAutoCluster;


	for (int i = 0; i < MAX_FACES; i++)
	{
		m_faceName[i][0] = '\0';
		m_blink[i] = 0;
		m_smile[i] = 0;
		m_gender[i] = 0;
		m_age[i] = -1;
		m_smileProb[i] = 0;;

		//	m_threadArgs.gbuff_faceName[i][0] = '\0';
	}

	m_nFaceDetectorNo = nFaceDetectorNo;
}

CxFaceAnalyzer::~CxFaceAnalyzer(void)
{
	// notify the worker thread to quit
	//pthread_mutex_lock (&m_threadArgs.glock);
	//m_threadArgs.gbuff_enque_flag  = 1;
	//pthread_mutex_unlock (&m_threadArgs.glock);
	//pthread_cond_signal(&m_threadArgs.gcond);

	//m_threadArgs.gbuff_run_flag = 0;
/*
#ifdef WIN32
	Sleep(200);
#else
	sleep(200);
#endif
*/
	// pthread_join(m_thread_id, NULL);
/*
#ifdef WIN32
	Sleep(300);
#else
	sleep(300);
#endif
*/
	// release buff
	cvReleaseImage(&m_pImgGray);

	// release objects
	delete m_plandmarkDetector;	
	delete m_pfaceRecognizer;

	m_plandmarkDetector = NULL;		
	m_pfaceRecognizer= NULL;

	if(m_cutface_big)     cvReleaseImage(&m_cutface_big);   
	if(m_cutface_small)   cvReleaseImage(&m_cutface_small);   
	for(int i=0;i<MAX_FACES;i++)
	{
		if(m_cutface_big_array[i])     cvReleaseImage(&(m_cutface_big_array[i]));   
		if(m_cutface_small_array[i])   cvReleaseImage(&(m_cutface_small_array[i]));   
	}
}

bool CxFaceAnalyzer::Face_Detection(IplImage *pImgSrc,int nMin_FaceSize = 80, char *ThumbnailImgFilename=NULL)
{
	if(pImgSrc == NULL) return false;

    if( m_pImgGray != NULL )
        cvReleaseImage( &m_pImgGray );
	
	m_pImgGray = cvCreateImage(cvGetSize(pImgSrc), IPL_DEPTH_8U, 1);

	if( pImgSrc->nChannels == 4 )
		cvCvtColor( pImgSrc, m_pImgGray, CV_BGRA2GRAY );
	if( pImgSrc->nChannels == 3 )
		cvCvtColor( pImgSrc, m_pImgGray, CV_BGR2GRAY );
	if( pImgSrc->nChannels == 1 )
		cvCopy( pImgSrc, m_pImgGray );

	SetFaceSizeRange(m_nFaceDetectorNo, nMin_FaceSize, pImgSrc->width*0.5);
	// do face tracking
	//m_face_num = m_Facedetector->detect( m_pImgGray, m_rects, MAX_FACES);
	FdRect FaceArea[MAX_FACES];
	//m_face_num = FrontalView_FaceDetection(m_nFaceDetectorNo, m_pImgGray, FaceArea);
	m_face_num = FrontalView_ColorImage_FaceDetection(m_nFaceDetectorNo, m_pImgGray, FaceArea, 0);
	for(int i=0;i<m_face_num;i++)
	{
		m_rects[i].rc.x = FaceArea[i].x;
		m_rects[i].rc.y = FaceArea[i].y;
		m_rects[i].rc.width = FaceArea[i].width;
		m_rects[i].rc.height = FaceArea[i].height;
		m_rects[i].angle = FaceArea[i].view;
	}

	ClearFaceSizeRange(m_nFaceDetectorNo);
	//detect and recognize each face
	int nLargest_Face_Size = -1;
	int nLargest_ID = -1;
	bool    bLandmark;
	CvRect rect;
	CvPoint2D32f* landmark6 ;
	int    angle;
	
	for( int i=0; i < m_face_num; i++ )
	{
		m_cutface_flag[i] = 0;
		// init
		// get face rect and id from tracker
		rect = m_rects[i].rc;
		angle = m_rects[i].angle;

		if(rect.x+rect.width  > m_pImgGray->width  || rect.x < 0) continue;
		if(rect.y+rect.height > m_pImgGray->height || rect.y < 0) continue;
		if(rect.width<nMin_FaceSize) continue;
		// detect landmark 

        landmark6 = m_ldmks[i];
		bLandmark = false;
 
		bLandmark = m_plandmarkDetector->detect( m_pImgGray, &rect, landmark6, NULL, angle );
		if(bLandmark ==false) continue;

		if(rect.width> nLargest_Face_Size)
		{
			nLargest_Face_Size = rect.width;
			nLargest_ID = i;
		}
	}
	if(nLargest_ID>-1)
	{
		landmark6 = m_ldmks[nLargest_ID];
		rect = m_rects[nLargest_ID].rc;
		alignFace2(m_pImgGray, landmark6, &rect, m_cutface_big->width, m_cutface_big->height, false, m_age_sclxyud, m_cutface_big);

		cvResize(m_cutface_big, m_cutface_small);
		
		IplImage *lpTest = alignFace3(pImgSrc, landmark6, &rect, m_cutface_big->width * 2, m_cutface_big->height * 2, false, m_age_sclxyud, NULL);
		cvSaveImage(ThumbnailImgFilename,lpTest);
		cvReleaseImage(&lpTest);

	}
	
	cvReleaseImage(&m_pImgGray);	
	m_pImgGray = NULL;
	if(nLargest_ID>-1)
		return true;
	else return false;
}

void CxFaceAnalyzer::loadFaceModelXML(const char *sPathXML)
{
	m_pfaceRecognizer->loadFaceModelXML(sPathXML);
}

void CxFaceAnalyzer::saveFaceModelXML(const char *sPathXML, vFaceSet *pvecFaceSet /*=NULL*/)
{
	m_pfaceRecognizer->saveFaceModelXML(sPathXML, pvecFaceSet);
}

void CxFaceAnalyzer::getMergedFaceSet(vFaceSet& vvClusters, int minWeight /*=0*/)
{
	m_pfaceRecognizer->getMergedFaceSet(vvClusters, minWeight);
}

char* CxFaceAnalyzer::getFaceImgDBPath()
{
	return m_pfaceRecognizer->getFaceImgDBPath();
}

int   CxFaceAnalyzer::getFaceSetSize(int nFaceSetIdx /*= -1*/)
{
	return m_pfaceRecognizer->getFaceSetSize(nFaceSetIdx);
}

const char* CxFaceAnalyzer::getKeyFacePath(int nFaceSetIdx, int nFaceIdx /*=-1*/)
{
	return m_pfaceRecognizer->getKeyFacePath(nFaceSetIdx, nFaceIdx);
}
