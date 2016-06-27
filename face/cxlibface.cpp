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

#include <string>

#include "cxlibface.hpp"
#include "cxface.hpp"
#include "cxfaceanalyzer.hpp"

#include "NewFaceDetector/Frontal_Realboosting_Dll.h"

void cxlibDrawCrossPoint( IplImage *img, CvPoint pt, int thickness /*=1*/)
{
	cxDrawCrossPoint( img, pt, thickness);
}


void cxlibDrawCaption( IplImage *img, CvFont *pFont, char* sCaption )
{
	cxDrawCaption( img, pFont, sCaption );
}

void cxlibDrawFaceRect(IplImage *img, CvRect rect,  CvScalar cl_face /*=CV_RGB(0,255,0)*/)
{
	cxDrawFaceRect( img, rect,  cl_face );
}

void cxlibDrawFaceBlob( IplImage *img, CvFont* pFont, int id, CvRect rect, CvPoint2D32f *landmark6, float probSmile, 
					int bBlink, int bSmile, int bGender, int nAgeID, char* sFaceName, char* sCaption, 
					IplImage *pImgSmileBGR /*=NULL*/, IplImage *pImgSmileBGRA /*=NULL*/, IplImage * pImgSmileMask /*=NULL*/)
{
	cxDrawFaceBlob( img, pFont, id, rect, landmark6, probSmile, 
		bBlink, bSmile, bGender, nAgeID, sFaceName, sCaption, 
		pImgSmileBGR /*=NULL*/, pImgSmileBGRA /*=NULL*/, pImgSmileMask /*=NULL*/);
}


void cxlibAutoFocusFaceImage(IplImage* pSrc, IplImage* pDest, CvRectItem *vFaceRect, int faceNum, float alpha /*= 0.05*/)
{
	autoFocusFaceImage(pSrc, pDest, vFaceRect, faceNum, alpha);
}

/************************** CxlibAlignFace *****************************/
CxlibAlignFace::CxlibAlignFace(int sizeSmallface/*=64*/, int sizeBigface /*= 128*/)
{
	m_alignface = new CxAlignFace(sizeSmallface, sizeBigface);
}

CxlibAlignFace::CxlibAlignFace(IplImage* pGrayImg, CvRect rect, CvPoint2D32f landmark6[])
{
	m_alignface = new CxAlignFace;
	init(pGrayImg, rect, landmark6);
}

CxlibAlignFace::~CxlibAlignFace()
{
	delete m_alignface;
}

void CxlibAlignFace::init(IplImage* lpImage, CvRect rect, CvPoint2D32f landmark6[])
{
	if (lpImage->nChannels == 1)
	{
		m_alignface->init(lpImage, rect, landmark6);
		return;
	}
	IplImage *gray_image = cvCreateImage(cvGetSize(lpImage), IPL_DEPTH_8U, 1);
	IplImage *lpHSVImage = cvCreateImage(cvSize(lpImage->width, lpImage->height), lpImage->depth, 3);
	cvCvtColor(lpImage, lpHSVImage, CV_BGR2HSV);
	for (int i = 0; i < lpHSVImage->height; i++)
	for (int j = 0; j < lpHSVImage->width; j++)
	{
		gray_image->imageData[i*gray_image->widthStep + j] = lpHSVImage->imageData[i*lpHSVImage->widthStep + j * 3 + 2];
	}
	//	cvSaveImage("c:/temp/tee.jpg", grayImage);
	m_alignface->init(gray_image, rect, landmark6);

	cvReleaseImage(&lpHSVImage);
	cvReleaseImage(&gray_image);
}

void CxlibAlignFace::clear()
{
	m_alignface->clear();
}

IplImage* CxlibAlignFace::getBigCutFace()
{
	return m_alignface->getBigCutFace();
}

IplImage* CxlibAlignFace::getSmallCutFace()
{
	return m_alignface->getSmallCutFace();
}

int CxlibAlignFace::getBigCutFaceSize()
{
	return m_alignface->getBigCutFaceSize();
}

int CxlibAlignFace::getSmallCutFaceSize()
{
	return m_alignface->getSmallCutFaceSize();
}

/************************** FaceDetector *****************************/
CxlibFaceDetector::CxlibFaceDetector( )
{
	//m_detector = NULL;
}

CxlibFaceDetector::CxlibFaceDetector(EnumViewAngle viewAngle /*= VIEW_ANGLE_FRONTAL*/, EnumFeaType feaType /*= FEA_SURF*/, int nFaceDetectorNo, const char* Modelfile /*= NULL*/)
{
	//m_detector = NULL;
	init(viewAngle, feaType, nFaceDetectorNo, Modelfile);
}

#include "NewFaceDetector\Const.h"
void CxlibFaceDetector::init( EnumViewAngle viewAngle /*= VIEW_ANGLE_FRONTAL*/, EnumFeaType feaType /*= FEA_SURF*/, int nFaceDetector_No, const char* Modelfile /*= NULL*/)
{
	FaceDetection_Init(WIN_WIDTH, WIN_HEIGHT, (char*)Modelfile);
	m_nFaceDetectorNo = nFaceDetector_No;
}


// detect face only in the image centeral region 
void CxlibFaceDetector::ClearFaceDetectionROI()
{
	ClearFaceROI(m_nFaceDetectorNo);
}
void CxlibFaceDetector::ClearFaceDetectionRange()
{
	ClearFaceSizeRange(m_nFaceDetectorNo);
}
void CxlibFaceDetector::SetFaceDetectionROI(IplImage* lpimage, double dCenterRatio)
{
	ROI_Rect FaceROI;
//	if((lpimage->width<=1024) &&(lpimage->height<=1024))
//		dCenterRatio = 1.0;
	double dOffsetRatio = (1 - dCenterRatio) * 0.3;
	FaceROI.x = int(lpimage->width * dOffsetRatio);
	FaceROI.y = int(lpimage->height * dOffsetRatio);
	FaceROI.width = int(lpimage->width*dCenterRatio);
	FaceROI.height = int(lpimage->height*dCenterRatio);

	SetFaceROI(m_nFaceDetectorNo, FaceROI);
}
void CxlibFaceDetector::SetFaceDetectionSizeRange(IplImage* lpimage, int nMinFaceSize)
{
	int nMinSize = nMinFaceSize;

	if((lpimage->width<=1024) ||(lpimage->height<=1024))
		nMinSize = 50;

	if(nMinSize<lpimage->width/15)
		nMinSize = int(lpimage->width/15);
	SetFaceSizeRange(m_nFaceDetectorNo, nMinSize, int(lpimage->width / 2));

}
int CxlibFaceDetector::detect(IplImage* image, CvRectItem rects[], int nColorFlag)
{
	FdRect FaceArea[100];
	int face_num;
	//if (nColorFlag >0)
	//	face_num = FrontalView_ColorImage_FaceDetection(m_nFaceDetectorNo, image, FaceArea, true, 0);
	//else face_num = FrontalView_ColorImage_FaceDetection(m_nFaceDetectorNo, image, FaceArea, false, 0);

	face_num = FrontalView_ColorImage_FaceDetection(m_nFaceDetectorNo, image, FaceArea, false, 0);
	for(int i=0;i<face_num;i++)
	{
		rects[i].rc.x = FaceArea[i].x;
		rects[i].rc.y = FaceArea[i].y;
		rects[i].rc.width = FaceArea[i].width;
		rects[i].rc.height = FaceArea[i].height;
		rects[i].angle = 0;// FaceArea[i].view;
	}
	return face_num;
}

CxlibFaceDetector::~CxlibFaceDetector(void)
{
	FaceDetection_Release();
}


/************************** Face Landmark *****************************/
CxlibLandmarkDetector::CxlibLandmarkDetector()
{
	m_comp = NULL;
}

CxlibLandmarkDetector::CxlibLandmarkDetector(EnumLandmarkerType landmarkerType /*= LDM_6PT*/,
											 const char* xml_eye_leftcorner /*=NULL*/, 
											 const char* xml_mth_leftcorner /*=NULL*/,
											 const char* xml_nose /*=NULL*/ )
{
	m_comp = NULL;
	init(landmarkerType, xml_eye_leftcorner, xml_mth_leftcorner, xml_nose);
}

void CxlibLandmarkDetector::init( EnumLandmarkerType landmarkerType /*= LDM_6PT*/,
											  const char* xml_eye_leftcorner /*=NULL*/, 
											  const char* xml_mth_leftcorner /*=NULL*/,
											  const char* xml_nose /*=NULL*/)
{	
	if(m_comp) delete m_comp;

	const char* sxml_eye_leftcorner = xml_eye_leftcorner ? xml_eye_leftcorner : "opencv_input/haarcascade_eye_leftcorner.xml";
	const char* sxml_mth_leftcorner = xml_mth_leftcorner ? xml_mth_leftcorner : "opencv_input/haarcascade_mth_leftcorner.xml";
	const char* sxml_nose = xml_nose ? xml_nose : "opencv_input/haarcascade_nose_20x20.xml";

	if(landmarkerType == LDM_7PT)
		m_comp = new CxCompDetector7pt( sxml_eye_leftcorner, sxml_mth_leftcorner, sxml_nose );
	else 
		m_comp = new CxCompDetector( sxml_eye_leftcorner, sxml_mth_leftcorner );
}

CxlibLandmarkDetector::~CxlibLandmarkDetector(void)
{
	delete m_comp;
}

// detect 6 points within 'rc_face', output to 'pt_comp'
bool CxlibLandmarkDetector::detect( const IplImage* image, CvRect* rect, CvPoint2D32f points[], float params[], int angle )
{
	if (image->nChannels == 1)
	{
		return m_comp->detect(image, rect, points, params, angle);
	}
	
	IplImage *gray_image = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);
	IplImage *lpHSVImage = cvCreateImage(cvSize(image->width, image->height), image->depth, 3);
	cvCvtColor(image, lpHSVImage, CV_BGR2HSV);
	for (int i = 0; i < lpHSVImage->height; i++)
	for (int j = 0; j < lpHSVImage->width; j++)
	{
		gray_image->imageData[i*gray_image->widthStep + j] = lpHSVImage->imageData[i*lpHSVImage->widthStep + j * 3 + 2];
	}
	//	cvSaveImage("c:/temp/tee.jpg", grayImage);
	bool bResult = m_comp->detect(gray_image, rect, points, params, angle);
	cvReleaseImage(&lpHSVImage);
	cvReleaseImage(&gray_image);

	return bResult;
}

bool CxlibLandmarkDetector::track( const IplImage* image, CvRect* rect, CvPoint2D32f points[], float params[], int angle )
{
	return m_comp->track( image, rect, points, params, angle );
}

// retrieve each comp
CvPoint2D32f CxlibLandmarkDetector::getPoint( int comp ) const
{
	return m_comp->getPoint( comp );
}

// Align Face
IplImage* CxlibLandmarkDetector::alignface(const IplImage *pGrayImg, const CvPoint2D32f pt6s[], CvRect rc, 
						  int nDstImgW, int nDstImgH, bool bHistEq /*=false*/, float* sclxyud /*=NULL*/, IplImage *pCutFace /*=NULL*/)
{
	//return alignFace3( pGrayImg, pt6s, &rc, nDstImgW, nDstImgH, bHistEq );
	alignFace2( pGrayImg, pt6s, &rc, nDstImgW, nDstImgH, bHistEq, sclxyud, pCutFace );
	cvSmooth(pCutFace, pCutFace);
	return pCutFace;
}

/************************** Smile Detector *****************************/
CxlibSmileDetector::CxlibSmileDetector( )
{
	m_smile_detector = NULL;
}

CxlibSmileDetector::CxlibSmileDetector( int cutimg_size /*= 64*/, const char* modelpath /*= NULL*/ )
{
	m_smile_detector = NULL;
	init(cutimg_size, modelpath);
}

void CxlibSmileDetector::init( int cutimg_size /*= 64*/, const char* modelpath /*= NULL*/ )
{
	if(m_smile_detector) delete m_smile_detector;

	m_smile_detector = new CxBoostDetect;
	
	const char* smodel_path = modelpath ? modelpath : "opencv_input/smilemod";
	
	if (!m_smile_detector->load( smodel_path, "smile_ann_cascade.mod", cutimg_size ))
	{
		OPENCV_ERROR( CV_StsBadArg, 
			"CxlibSmileDetector::CxlibSmileDetector()", 
			"Cannot load smile model file." );
	}
}

CxlibSmileDetector::~CxlibSmileDetector(void)
{
	delete m_smile_detector;
}

int CxlibSmileDetector::predict( CxlibAlignFace* pCutFace, float *prob  )
{
	return m_smile_detector->predict( pCutFace->getSmallCutFace(), prob );
}

int CxlibSmileDetector::predict( IplImage* pCutFace, float *prob  )
{
	return m_smile_detector->predict( pCutFace, prob );
}

int   CxlibSmileDetector::voteLabel(int face_trackid, int label, int vote_threshold /*= 2*/, int smooth_len /*= 8*/)
{
	return m_smile_detector->voteLabel( face_trackid, label, vote_threshold, smooth_len );
}

float CxlibSmileDetector::getDefThreshold()
{
	return m_smile_detector->getDefThreshold();
}

int   CxlibSmileDetector::getDefRound()
{
	return m_smile_detector->getDefRound();
}

/************************** Blink Detector *****************************/
CxlibBlinkDetector::CxlibBlinkDetector( )
{
	m_blink_detector = NULL;
}

CxlibBlinkDetector::CxlibBlinkDetector( int cutimg_size /*= 64*/, const char* modelpath /*= NULL*/ )
{
	m_blink_detector = NULL;
	init(cutimg_size, modelpath);
}

void CxlibBlinkDetector::init( int cutimg_size /*= 64*/, const char* modelpath /*= NULL*/ )
{
	if(m_blink_detector) delete m_blink_detector;

	m_blink_detector = new CxBoostDetect;
	
	const char* smodel_path = modelpath ? modelpath : "opencv_input/blinkmod";
	
	if (!m_blink_detector->load( smodel_path, "blink_ann_cascade.mod", cutimg_size ))
	{
		OPENCV_ERROR( CV_StsBadArg, 
			"CxlibBlinkDetector::CxlibBlinkDetector()", 
			"Cannot load blink model file." );
	}
}

CxlibBlinkDetector::~CxlibBlinkDetector(void)
{
	delete m_blink_detector;
}

int CxlibBlinkDetector::predict( CxlibAlignFace* pCutFace, float *prob  )
{
	return predict( pCutFace->getSmallCutFace(), prob );
}

int CxlibBlinkDetector::predict( IplImage* pCutFace, float *prob  )
{
	return m_blink_detector->predict( pCutFace, prob );
}

int   CxlibBlinkDetector::voteLabel(int face_trackid, int label, int vote_threshold /*= 1*/, int smooth_len /*= 2*/)
{
	return m_blink_detector->voteLabel( face_trackid, label, vote_threshold, smooth_len );
}

float CxlibBlinkDetector::getDefThreshold()
{
	return m_blink_detector->getDefThreshold();
}

int   CxlibBlinkDetector::getDefRound()
{
	return m_blink_detector->getDefRound();
}

/************************** Gender Detector *****************************/
CxlibGenderDetector::CxlibGenderDetector( )
{
	m_gender_detector = NULL;
}

CxlibGenderDetector::CxlibGenderDetector( int cutimg_size /*= 64*/,  const char* modelpath /*= NULL*/ )
{
	m_gender_detector = NULL;
	init(cutimg_size, modelpath);
}

void CxlibGenderDetector::init( int cutimg_size /*= 64*/,  const char* modelpath /*= NULL*/ )
{
	if(m_gender_detector) delete m_gender_detector;

	m_gender_detector = new CxBoostDetect;
	
	int nResult;
	if (modelpath == NULL)
		nResult = m_gender_detector->load("opencv_input/gendermod", "gender_ann_cascade.mod", cutimg_size);
	else nResult = m_gender_detector->load(modelpath, "gender_ann_cascade.mod", cutimg_size);
}

CxlibGenderDetector::~CxlibGenderDetector(void)
{
	delete m_gender_detector;
}

int CxlibGenderDetector::predict( CxlibAlignFace* pCutFace, float *prob  )
{
	return m_gender_detector->predict( pCutFace->getSmallCutFace(), prob );
}

int CxlibGenderDetector::predict( IplImage* pCutFace, float *prob  )
{
	return m_gender_detector->predict( pCutFace, prob );
}

int   CxlibGenderDetector::voteLabel(int face_trackid, int label)
{
	return m_gender_detector->voteLabel( face_trackid, label );
}

float CxlibGenderDetector::getDefThreshold()
{
	return m_gender_detector->getDefThreshold();
}

int   CxlibGenderDetector::getDefRound()
{
	return m_gender_detector->getDefRound();
}

/************************** Age Detector *****************************/
CxlibAgeDetector::CxlibAgeDetector( )
{
	m_age_detector = NULL;
}
CxlibAgeDetector::CxlibAgeDetector( int cutimg_size /*= 128*/, const char* modelpath /*= NULL*/ )
{
	m_age_detector = NULL;
	init(cutimg_size, modelpath);
}

void CxlibAgeDetector::init( int cutimg_size /*= 128*/, const char* modelpath /*= NULL*/ )
{
	if(m_age_detector) delete m_age_detector;

	m_age_detector = new CxMCBoostDetect;
	
	const char* smodel_path = modelpath ? modelpath : "opencv_input/agemod";
	
	if (!m_age_detector->load( smodel_path, "age_cascade.mod", cutimg_size ))
	{
		OPENCV_ERROR( CV_StsBadArg, 
			"CxlibAgeDetector::CxlibAgeDetector()", 
			"Cannot load age model file." );
	}
}

CxlibAgeDetector::~CxlibAgeDetector(void)
{
	delete m_age_detector;
}

int CxlibAgeDetector::predict( CxlibAlignFace* pCutFace, float *prob  )
{
	return m_age_detector->predict( pCutFace->getBigCutFace(), prob );
}

int CxlibAgeDetector::predict( IplImage* pCutFace, float *prob  )
{
	return m_age_detector->predict( pCutFace, prob );
}

int CxlibAgeDetector::voteLabel(int face_trackid, int label)
{
	return m_age_detector->voteLabel( face_trackid, label );
}

int   CxlibAgeDetector::getDefRound()
{
	return m_age_detector->getDefRound();
}

/************************** Face Recognizer *****************************/
CxlibFaceRecognizer::CxlibFaceRecognizer( )
{
	m_face_recognizer = NULL;
}

CxlibFaceRecognizer::CxlibFaceRecognizer( int cutimg_size /*= 128*/, int recognierType /*= RECOGNIZER_CAS_GLOH*/, const char* modelpath /*= NULL*/ )
{
	m_face_recognizer = NULL;
	init(cutimg_size, recognierType, modelpath);
}

void CxlibFaceRecognizer::init( int cutimg_size /*= 128*/, int recognierType /*= RECOGNIZER_CAS_GLOH*/, const char* modelpath /*= NULL*/ )
{
	if(m_face_recognizer) delete m_face_recognizer;

	m_face_recognizer = new CxBoostFaceRecog;

	const char* smodel_path;
	if(recognierType == RECOGNIZER_BOOST_GB240)
		smodel_path = modelpath ? modelpath : "opencv_input/facemod/Gb240";
	else if(recognierType == RECOGNIZER_BOOST_LBP59)
		smodel_path = modelpath ? modelpath : "opencv_input/facemod/LBP59";
	else if(recognierType == RECOGNIZER_CAS_GLOH)
		smodel_path = modelpath ? modelpath : "opencv_input/facemod/GLOH";

	if (!m_face_recognizer->load( recognierType, smodel_path, "face_ann_cascade.mod", cutimg_size ))
	{
		OPENCV_ERROR( CV_StsBadArg, 
			"CxlibFaceRecognizer::CxlibFaceRecognizer()", 
			"Cannot load face recognizer model file." );
	}
}

CxlibFaceRecognizer::~CxlibFaceRecognizer(void)
{
	delete m_face_recognizer;
}

int CxlibFaceRecognizer::loadFaceModelXML( const char* xmlpath )
{
	return m_face_recognizer->loadFaceModelXML( xmlpath );
}

void CxlibFaceRecognizer::saveFaceModelXML(const char *sPathXML, vFaceSet *pvecFaceSet /*=NULL*/)
{
	m_face_recognizer->saveFaceModelXML(sPathXML, pvecFaceSet);
}

void CxlibFaceRecognizer::getMergedFaceSet(vFaceSet& vvClusters, int minWeight /*=0*/)
{
	m_face_recognizer->getMergedFaceSet(vvClusters, minWeight);
}

int CxlibFaceRecognizer::predict( CxlibAlignFace* pCutFace, float *prob, bool bAutoCluster  /*= false*/, int face_trackid /*= -1*/, int frameid /*= -1*/)
{
	return m_face_recognizer->predict( pCutFace->getBigCutFace(), prob, bAutoCluster, face_trackid, frameid);
}

int CxlibFaceRecognizer::predict( IplImage* pCutFace, float *prob, bool bAutoCluster /*= false*/, int face_trackid /*= -1*/, int frameid /*= -1*/)
{
	return m_face_recognizer->predict( pCutFace, prob, bAutoCluster, face_trackid, frameid);
}

int CxlibFaceRecognizer::voteLabel(int face_trackid, int label)
{
	return m_face_recognizer->voteLabel( face_trackid, label );
}

const char* CxlibFaceRecognizer::getFaceName( int nFaceSetID )
{
	return m_face_recognizer->getFaceName( nFaceSetID );
}

int CxlibFaceRecognizer::getFaceSetID(int nFaceSetIdx)
{
	return m_face_recognizer->getFaceSetID(nFaceSetIdx);
}

int CxlibFaceRecognizer::getFaceSetIdx(int nFaceSetID)
{
	return m_face_recognizer->getFaceSetIdx(nFaceSetID);
}


std::vector<std::string> * CxlibFaceRecognizer::getKeyFacePaths(int faceSetIdx)
{
	return m_face_recognizer->getKeyFacePaths(faceSetIdx); 
}

int   CxlibFaceRecognizer::getFaceSetSize(int nFaceSetIdx /*= -1*/)
{
	return m_face_recognizer->getFaceSetSize(nFaceSetIdx);
} 

const char* CxlibFaceRecognizer::getKeyFacePath(int nFaceSetIdx, int nFaceIdx /*=-1*/)
{
	return m_face_recognizer->getKeyFacePath(nFaceSetIdx, nFaceIdx);
}

float CxlibFaceRecognizer::getDefThreshold()
{
	return m_face_recognizer->getDefThreshold();
}

int   CxlibFaceRecognizer::getDefRound()
{
	return m_face_recognizer->getDefRound();
}

int CxlibFaceRecognizer::getFeatureDim()
{
	return m_face_recognizer->getFeatureDim();
}

int CxlibFaceRecognizer::getFeatureType()
{
	return m_face_recognizer->getFeatureType();
}

vFaceSet* CxlibFaceRecognizer::getFaceSets()
{
	return m_face_recognizer->getFaceSets();
}

int  CxlibFaceRecognizer::insertEmptyFaceSet(char *name, bool createFolder /*= true*/, int nFaceSetID /*=-1*/)
{
	return m_face_recognizer->insertEmptyFaceSet(name, createFolder, nFaceSetID);
}

bool CxlibFaceRecognizer::tryInsertFace(IplImage* pCutFace, int nFaceSetIdx, bool bForceInsert /*= false*/)
{
	return m_face_recognizer->tryInsertFace(pCutFace, nFaceSetIdx, bForceInsert);
}

int CxlibFaceRecognizer::removeFaceSet(int nFaceSetIdx)
{
	return m_face_recognizer->removeFaceSet(nFaceSetIdx);
}

int CxlibFaceRecognizer::removeFace(int nFaceSetIdx, int faceIdx)
{
	return m_face_recognizer->removeFace(nFaceSetIdx, faceIdx);
}

void CxlibFaceRecognizer::extFeature(IplImage* pCutFace, float* pFea)
{
	return m_face_recognizer->extFeature(pCutFace, pFea);
}

bool CxlibFaceRecognizer::isSimilarFaces(float* pFea1, float* pFea2, float *pProb /*=NULL*/)
{
	return m_face_recognizer->isSimilarFaces(pFea1, pFea2, pProb);
}
						  
int  CxlibFaceRecognizer::forwardCluster(float* pFea, int faceID, char *sCutFaceImg, vFaceSet &vvClusters, vFaceSet &vvRepClusters, float fThreshold /*=-1*/)
{
	return m_face_recognizer->forwardCluster(pFea, faceID, sCutFaceImg, vvClusters, vvRepClusters, fThreshold);
}

int  CxlibFaceRecognizer::clusterHAC(CvMat* pmSim, vFaceSet& vvFaceSet, float fThreshold /*= -1*/,
									 int nMinClusterNum /*=-1*/, std::vector<int> *pvExemplars /*=NULL*/)
{
	return m_face_recognizer->clusterHAC(pmSim, vvFaceSet, fThreshold, nMinClusterNum, pvExemplars);
}

int  CxlibFaceRecognizer::rankOrderCluster(CvMat* pmSim, vFaceSet& vvClusters, float rankDistThresh /*= 12*/, float normDistThresh /*= 1.02*/)
{
	return m_face_recognizer->rankOrderCluster(pmSim, vvClusters, rankDistThresh, normDistThresh);
}

void CxlibFaceRecognizer::mergeClusters(vFaceSet& vvClusters, int cA, int cB, vFaceSet* vvRepClusters /*= NULL*/)
{
	return m_face_recognizer->mergeClusters(vvClusters, cA, cB, vvRepClusters);
}

CvMat* CxlibFaceRecognizer::clacSimMat(std::vector <std::string> vFaceImgList, CvMat* &pmSim)
{
	return m_face_recognizer->clacSimMat(vFaceImgList, pmSim);
}

CvMat* CxlibFaceRecognizer::clacSimMat(std::vector <CvMat*> matFea, CvMat* &pmSim)
{
	return m_face_recognizer->clacSimMat(matFea, pmSim);
}

/************************** Face Analyzer *****************************/

CxlibFaceAnalyzer::CxlibFaceAnalyzer()
{
	m_pfaceAnalyzer = NULL;
}

CxlibFaceAnalyzer::CxlibFaceAnalyzer(EnumViewAngle viewAngle, EnumTrackerType traType, int nFaceDetector,
									 const char *str_facesetxml , int recognierType,
									 bool bEnableAutoCluster , bool bLandmarkRegressor,
									 const char* xml_eye_leftcorner,
									 const char* xml_mth_leftcorner,
									 const char *sFaceRecognizer_modelpath)
{
	m_pfaceAnalyzer = NULL;
	init(viewAngle, traType, nFaceDetector, str_facesetxml, recognierType, bEnableAutoCluster, bLandmarkRegressor,
		xml_eye_leftcorner, xml_mth_leftcorner, sFaceRecognizer_modelpath);
}

void CxlibFaceAnalyzer::init(EnumViewAngle viewAngle, EnumTrackerType traType, int nFaceDetector,
							 const char *str_facesetxml , int recognierType ,
									 bool bEnableAutoCluster , bool bLandmarkRegressor,
									 const char* xml_eye_leftcorner,
									 const char* xml_mth_leftcorner,
									 const char *sFaceRecognizer_modelpath)
{
	if(m_pfaceAnalyzer) delete m_pfaceAnalyzer;

	m_pfaceAnalyzer = new CxFaceAnalyzer(viewAngle, traType, nFaceDetector, str_facesetxml, recognierType, bEnableAutoCluster, bLandmarkRegressor,
		xml_eye_leftcorner, xml_mth_leftcorner, sFaceRecognizer_modelpath);
}



CxlibFaceAnalyzer::~CxlibFaceAnalyzer(void)
{
	delete m_pfaceAnalyzer;
}

vFaceSet* CxlibFaceAnalyzer::getFaceSets()
{
	return m_pfaceAnalyzer->getFaceSets();
}

char* CxlibFaceAnalyzer::getFaceImgDBPath()
{
	return m_pfaceAnalyzer->getFaceImgDBPath();
}

int CxlibFaceAnalyzer::getFaceSetID(int nFaceSetIdx)
{
	return m_pfaceAnalyzer->getFaceSetID(nFaceSetIdx);
}

int   CxlibFaceAnalyzer::getFaceSetSize(int nFaceSetIdx )
{
	return m_pfaceAnalyzer->getFaceSetSize(nFaceSetIdx);
} 

const char* CxlibFaceAnalyzer::getKeyFacePath(int nFaceSetIdx, int nFaceIdx )
{
	return m_pfaceAnalyzer->getKeyFacePath(nFaceSetIdx, nFaceIdx);
}

bool CxlibFaceAnalyzer::Face_Detection(IplImage *pColorImg, int nMin_FaceSize, char *ThumbnailImgFilename)
{
	return m_pfaceAnalyzer->Face_Detection(pColorImg, nMin_FaceSize, ThumbnailImgFilename);
}

int CxlibFaceAnalyzer::predictFaceSet(IplImage *pCutFace, float *prob, char *sFaceName)
{
	CxBoostFaceRecog *pFaceRecog = m_pfaceAnalyzer->getFaceRecognizer();
	int face_round = pFaceRecog->getDefRound();
	int nFaceSetID = pFaceRecog->predict(pCutFace, prob);

	const char* name = pFaceRecog->getFaceName(nFaceSetID);
	strcpy(sFaceName, name);

	return nFaceSetID;
}

int CxlibFaceAnalyzer::insertEmptyFaceSet(char *sFaceName, bool createFolder , int nFaceSetID)
{
	CxBoostFaceRecog *pFaceRecog = m_pfaceAnalyzer->getFaceRecognizer();
	int nFaceSetIdx = pFaceRecog->insertEmptyFaceSet(sFaceName, createFolder, nFaceSetID);
	
	return nFaceSetIdx;
}

bool CxlibFaceAnalyzer::tryInsertFace(IplImage* pCutFace, int nFaceSetIdx, bool bForceInsert)
{
	CxBoostFaceRecog *pFaceRecog = m_pfaceAnalyzer->getFaceRecognizer();
	bool bSim = pFaceRecog->tryInsertFace(pCutFace, nFaceSetIdx, bForceInsert);

	return bSim;
}

int CxlibFaceAnalyzer::removeFaceSet(int nFaceSetIdx)
{
	CxBoostFaceRecog *pFaceRecog = m_pfaceAnalyzer->getFaceRecognizer();
	return pFaceRecog->removeFaceSet(nFaceSetIdx);
}

int CxlibFaceAnalyzer::removeFace(int nFaceSetIdx, int faceIdx)
{
	CxBoostFaceRecog *pFaceRecog = m_pfaceAnalyzer->getFaceRecognizer();
	return pFaceRecog->removeFace(nFaceSetIdx, faceIdx);
}


void CxlibFaceAnalyzer::loadFaceModelXML(const char *sPathXML)
{
	m_pfaceAnalyzer->loadFaceModelXML(sPathXML);
}

void CxlibFaceAnalyzer::saveFaceModelXML(const char *sPathXML, vFaceSet *pvecFaceSet )
{
	m_pfaceAnalyzer->saveFaceModelXML(sPathXML, pvecFaceSet);
}

void CxlibFaceAnalyzer::getMergedFaceSet(vFaceSet& vvClusters, int minWeight )
{
	m_pfaceAnalyzer->getMergedFaceSet(vvClusters, minWeight);
}

IplImage* CxlibFaceAnalyzer::getBigCutFace() 
{ 
	return m_pfaceAnalyzer->getBigCutFace(); 
}

int CxlibFaceAnalyzer::getFaceNum()              
{ 
	return m_pfaceAnalyzer->getFaceNum(); 
}

CvRectItem CxlibFaceAnalyzer::getFaceRect(int idx)      
{ 
	return m_pfaceAnalyzer->getFaceRect(idx);  
}

CvRectItem* CxlibFaceAnalyzer::getFaceRects()            
{ 
	return m_pfaceAnalyzer->getFaceRects();  
}

CvPoint2D32f* CxlibFaceAnalyzer::getFaceLdmks(int idx)            
{ 
	return m_pfaceAnalyzer->getFaceLdmks( idx );  
}


int CxlibFaceAnalyzer::getFaceID(int idx)       
{ 
	return m_pfaceAnalyzer->getFaceID(idx); 
} 

float CxlibFaceAnalyzer::getFaceProb(int idx)       
{ 
	return m_pfaceAnalyzer->getFaceProb(idx); 
} 

char* CxlibFaceAnalyzer::getFaceName(int idx)
{
	return m_pfaceAnalyzer->getFaceName(idx); 
}

/************************** CxlibFace end *****************************/