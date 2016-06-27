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
#include "basetypes.hpp"

#ifndef DLLEXPORT
#ifdef __GNUC__
#define DLLEXPORT 
#else
#define DLLEXPORT __declspec(dllexport)
#endif
#endif


// declare virtual classes
class CxFaceDetector;
//class CxTrackerBase;
class CxCompDetBase;
class CxCompDetector;
class CxCompDetector7pt;
class CxBoostDetect;
class CxMCBoostDetect;
class CxBoostFaceRecog;
class CxFaceAnalyzer;
class CxAlignFace;

// drawing face API
DLLEXPORT void cxlibDrawCrossPoint( IplImage *img, CvPoint pt, int thickness =1 );

DLLEXPORT void cxlibDrawFaceRect( IplImage *img, CvRect rect,  CvScalar colors = CV_RGB(0,255,0) );

DLLEXPORT void cxlibDrawCaption(  IplImage *img, CvFont *pFont, char* sCaption);

DLLEXPORT void cxlibDrawFaceBlob( IplImage *img, CvFont *pFont, int id, CvRect rect, CvPoint2D32f *landmark6, float probSmile = 0, 
					 int bBlink = 0, int bSmile = 0, int bGender = 0, int nAgeID = 0, char* sFaceName = NULL, char* sCaption = NULL,
					 IplImage *pImgSmileBGR = NULL, IplImage *pImgSmileBGRA = NULL, IplImage * pImgSmileMask = NULL );

DLLEXPORT void cxlibAutoFocusFaceImage(IplImage* pSrc, IplImage* pDest, CvRectItem *vFaceRect, int faceNum, float alpha = 0.05);

class CxlibAlignFace
{
public:
	DLLEXPORT CxlibAlignFace(int sizeSmallface=64, int sizeBigface = 128);
	DLLEXPORT CxlibAlignFace(IplImage* pGrayImg, CvRect rect, CvPoint2D32f landmark6[]);
	DLLEXPORT ~CxlibAlignFace();

	DLLEXPORT void init(IplImage* pImg, CvRect rect, CvPoint2D32f landmark6[]);
	DLLEXPORT void clear();

	DLLEXPORT IplImage *getBigCutFace();
	DLLEXPORT IplImage *getSmallCutFace();


	DLLEXPORT int getBigCutFaceSize();
	DLLEXPORT int getSmallCutFaceSize();

	DLLEXPORT IplImage *getAlignedFaceThumbnail(IplImage* pColorImg, CvRect rect, CvPoint2D32f landmark6[]);
private:
	CxAlignFace* m_alignface;
};

/************************** FaceDetector *****************************/
class CxlibFaceDetector
{
public:
	DLLEXPORT CxlibFaceDetector( ); 
	DLLEXPORT CxlibFaceDetector(EnumViewAngle viewAngle /*= VIEW_ANGLE_FRONTAL*/, EnumFeaType feaType = FEA_SURF, int nFaceDetectorNo = 0, const char* sModelFile = NULL); 
	DLLEXPORT ~CxlibFaceDetector(void);
	
	// init
	DLLEXPORT void init(EnumViewAngle viewAngle = VIEW_ANGLE_FRONTAL, EnumFeaType feaType = FEA_SURF, int nFaceDetectorNo = 0, const char* sModelFile = NULL);

	// configure parameters
//	DLLEXPORT void config( tagDetectConfig configParam = tagDetectConfig());

	// detect face, return the number and rect of faces
	DLLEXPORT  int detect( IplImage* image, CvRectItem rects[], int nColorFlag);
	DLLEXPORT  void SetFaceDetectionSizeRange(IplImage* lpimage, int nMinFaceSize=60);
	DLLEXPORT  void SetFaceDetectionROI(IplImage* lpimage, double dCenterRatio);
	DLLEXPORT  void ClearFaceDetectionRange();
	DLLEXPORT  void ClearFaceDetectionROI();
	// get face thumbnail from rect of image
	//DLLEXPORT  IplImage* getThumbnail( IplImage* image, CvRect rect, IplImage* thumbnail = NULL);

private:
	int m_nFaceDetectorNo;
};


/************************** FaceTracker *****************************/
class CxlibFaceTracker
{
public:
	DLLEXPORT CxlibFaceTracker();
	DLLEXPORT CxlibFaceTracker( EnumViewAngle viewAngle /*= VIEW_ANGLE_FRONTAL*/, EnumTrackerType traType = TRA_SURF, const char* xmlfile = NULL);
	DLLEXPORT ~CxlibFaceTracker(void);

	// init
	DLLEXPORT void init( EnumViewAngle viewAngle = VIEW_ANGLE_FRONTAL, EnumTrackerType traType = TRA_SURF, const char* xmlfile = NULL);

	// configure parameters
	DLLEXPORT void config( tagDetectConfig configParam = tagDetectConfig(), int level = TR_NLEVEL_3 );

	// detect faces in the image
	DLLEXPORT int detect( IplImage* image, CvRectItem rects[], int count);

	// tracking faces in a video, if pBGRImage != NULL, color face tracker will be enabled
	DLLEXPORT int track( IplImage* pGreyImage, CvRectItem rects[], int count, IplImage* pBGRImage = NULL);

	// get face thumbnail from rect of image
	DLLEXPORT  IplImage* getThumbnail( IplImage* image, CvRect rect, IplImage* thumbnail = NULL);

private:
//	CxTrackerBase* m_tracker;
};

/************************** Face Landmark *****************************/
class CxlibLandmarkDetector
{
public:
	DLLEXPORT  CxlibLandmarkDetector();
	DLLEXPORT  CxlibLandmarkDetector(EnumLandmarkerType landmarkerType /*= LDM_6PT*/, const char* xml_eye_leftcorner = NULL, 
		const char* xml_mth_leftcorner = NULL, const char* xml_nose = NULL );	
	DLLEXPORT  ~CxlibLandmarkDetector(void);

	// init
	DLLEXPORT  void init(EnumLandmarkerType landmarkerType = LDM_7PT, const char* xml_eye_leftcorner = NULL, 
		const char* xml_mth_leftcorner = NULL, const char* xml_nose = NULL );

	// detect 6 points within 'rc_face', output to 'pt_comp'
	DLLEXPORT  bool detect( const IplImage* image, CvRect* rect, CvPoint2D32f points[], float parameters[]=NULL, int angle = 0 );
	DLLEXPORT  bool track( const IplImage* image, CvRect* rect, CvPoint2D32f points[], float parameters[]=NULL, int angle = 0 );

	// retrieve each comp
	DLLEXPORT  CvPoint2D32f getPoint( int comp ) const;

	//align face: pGrayImg must be gray image
	DLLEXPORT IplImage* alignface(const IplImage *pGrayImg, const CvPoint2D32f pt6s[], CvRect rc, 
		int nDstImgW, int nDstImgH, bool bHistEq =true, float* sclxyud =NULL, IplImage *pCutFace =NULL);

private:
	CxCompDetBase* m_comp;	
};

/************************** Smile Detector *****************************/

class CxlibSmileDetector
{
public:
	DLLEXPORT CxlibSmileDetector();
	DLLEXPORT CxlibSmileDetector(int cutimg_size /*= 64*/, const char* modelpath = NULL);
	DLLEXPORT ~CxlibSmileDetector( void );

	// init
	DLLEXPORT void init(int cutimg_size = 64, const char* modelpath = NULL);

	//threshold of predict, no smile: [0,0.48], smile: (0.48, 1]
	DLLEXPORT int predict( CxlibAlignFace* pCutFace, float *prob = NULL );
	DLLEXPORT int predict( IplImage* pCutFace, float *prob = NULL );
	DLLEXPORT int   voteLabel(int face_trackid, int label, int vote_threshold = 2, int smooth_len = 8); 
	
	DLLEXPORT float getDefThreshold();
	DLLEXPORT int   getDefRound();
private:
	CxBoostDetect* m_smile_detector;
};

/************************** Blink Detector *****************************/
class CxlibBlinkDetector
{
public:
	DLLEXPORT CxlibBlinkDetector( );
	DLLEXPORT CxlibBlinkDetector(int cutimg_size /*= 64*/, const char* modelpath = NULL);
	DLLEXPORT ~CxlibBlinkDetector( void );

	// init
	DLLEXPORT void init(int cutimg_size = 64, const char* modelpath = NULL );

	//threshold of predict, open: [0,0.5], close: (0.5, 1]
	DLLEXPORT int predict( CxlibAlignFace* pCutFace, float *prob = NULL );
	DLLEXPORT int predict( IplImage* pCutFace, float *prob = NULL );
	DLLEXPORT int   voteLabel(int face_trackid, int label, int vote_threshold = 1, int smooth_len = 2); 

	DLLEXPORT float getDefThreshold();
	DLLEXPORT int   getDefRound();
private:
	CxBoostDetect* m_blink_detector;
};

/************************** Gender Detector *****************************/
class CxlibGenderDetector
{
public:
	DLLEXPORT CxlibGenderDetector( );
	DLLEXPORT CxlibGenderDetector( int cutimg_size /*= 64*/, const char* modelpath = NULL );
	DLLEXPORT ~CxlibGenderDetector( void );

	// init
	DLLEXPORT void init(int cutimg_size = 64, const char* modelpath = NULL );

	//threshold of predict, male: [0,0.42], female: (0.42, 1]
	DLLEXPORT int predict( CxlibAlignFace* pCutFace, float *prob = NULL ); 
	DLLEXPORT int predict( IplImage* pCutFace, float *prob = NULL ); 
	DLLEXPORT int   voteLabel(int face_trackid, int label); 

	DLLEXPORT float getDefThreshold();
	DLLEXPORT int   getDefRound();
private:
	CxBoostDetect* m_gender_detector;
};

/************************** Age Detector *****************************/
class CxlibAgeDetector
{
public:
	DLLEXPORT CxlibAgeDetector( );
	DLLEXPORT CxlibAgeDetector(int cutimg_size /*= 128*/, const char* modelpath = NULL);
	DLLEXPORT ~CxlibAgeDetector( void );

	// init
	DLLEXPORT void init(int cutimg_size = 128, const char* modelpath = NULL );

	// output class-id: 0=>baby, 1=>child, 2=>adult, 3=>senior
	DLLEXPORT int predict( CxlibAlignFace* pCutFace, float *prob = NULL ); 
	DLLEXPORT int predict( IplImage* pCutFace, float *prob = NULL ); 
	DLLEXPORT int voteLabel(int face_trackid, int label); 

	DLLEXPORT int   getDefRound();

private:
	CxMCBoostDetect* m_age_detector;
};

/************************** Face Recognizer *****************************/
class CxlibFaceRecognizer
{
public:
	DLLEXPORT CxlibFaceRecognizer( );
	DLLEXPORT CxlibFaceRecognizer(int cutimg_size /*= 128*/, int recognierType = RECOGNIZER_CAS_GLOH, const char* modelpath = NULL);
	DLLEXPORT ~CxlibFaceRecognizer( void );

	// init
	DLLEXPORT void init(int cutimg_size = 128, int recognierType = RECOGNIZER_CAS_GLOH, const char* modelpath = NULL );

	// load exemplar face set xml file
	DLLEXPORT int  loadFaceModelXML(const char *xmlpath);
	DLLEXPORT void saveFaceModelXML(const char *sPathXML, vFaceSet *pvecFaceSet =NULL);
	DLLEXPORT void getMergedFaceSet(vFaceSet& vvClusters, int minWeight = 0);

	DLLEXPORT int  insertEmptyFaceSet(char *name, bool createFolder = true, int nFaceSetID = -1);
	DLLEXPORT bool tryInsertFace(IplImage* pCutFace, int nFaceSetIdx, bool bForceInsert = false);
	DLLEXPORT int  removeFaceSet(int nFaceSetIdx);
	DLLEXPORT int  removeFace(int nFaceSetIdx, int faceIdx);

	DLLEXPORT vFaceSet   *getFaceSets(); 
	DLLEXPORT const char *getFaceName(int nFaceSetID); 
	DLLEXPORT int         getFaceSetID(int nFaceSetIdx); 
	DLLEXPORT int         getFaceSetIdx(int nFaceSetID);
	DLLEXPORT std::vector<std::string> *getKeyFacePaths(int faceSetIdx);
	DLLEXPORT int   getFaceSetSize(int nFaceSetIdx = -1); 
	DLLEXPORT const char* getKeyFacePath(int nFaceSetIdx, int nFaceIdx = -1);

	// output face-id, prob > 0.52 is the face-id person, otherwise reject the person.
	DLLEXPORT int predict( CxlibAlignFace* pCutFace, float *prob = NULL, bool bAutoCluster = false, int face_trackid = -1, int frameid = -1);
	DLLEXPORT int predict( IplImage* pCutFace, float *prob = NULL, bool bAutoCluster = false, int face_trackid = -1, int frameid = -1);

	DLLEXPORT int voteLabel(int face_trackid, int label); 

	DLLEXPORT float getDefThreshold();
	DLLEXPORT int   getDefRound();

	DLLEXPORT int   getFeatureDim();
	DLLEXPORT int   getFeatureType(); 
	
	DLLEXPORT void  extFeature(IplImage* pCutFaceImg, float* pFea);
	DLLEXPORT bool  isSimilarFaces(float* pFea1, float* pFea2, float *pProb =NULL);

	////////////////////////////////////////////////////////////////////
	//cluster faces
	DLLEXPORT int forwardCluster(float* pFea, int faceID, char *sCutFaceImg, vFaceSet &vvClusters, vFaceSet &vvRepClusters, float fThreshold =-1);
	
	DLLEXPORT int clusterHAC(CvMat* pmSim, vFaceSet& vvFaceSet, float fThreshold = -1,
		int nMinClusterNum =-1, std::vector<int> *pvExemplars =NULL);
	
	DLLEXPORT int rankOrderCluster(CvMat* pmSim, vFaceSet& vvClusters, float rankDistThresh = 12, float normDistThresh = 1.02);

	DLLEXPORT void  mergeClusters(vFaceSet& vvClusters, int cA, int cB, vFaceSet* vvRepClusters = NULL);

	DLLEXPORT CvMat* clacSimMat(std::vector <std::string> vFaceImgList, CvMat* &pmSim);
	DLLEXPORT CvMat* clacSimMat(std::vector <CvMat*> matFea, CvMat* &pmSim);

private:
	CxBoostFaceRecog* m_face_recognizer;
};

/************************** Face Analyzer *****************************/
class CxlibFaceAnalyzer
{
public:
	DLLEXPORT CxlibFaceAnalyzer();
	DLLEXPORT CxlibFaceAnalyzer(EnumViewAngle viewAngle, EnumTrackerType traType = TRA_SURF, int nFaceDetector =0,
		const char *str_facesetxml =NULL, int recognierType = RECOGNIZER_CAS_GLOH, bool bEnableAutoCluster = false,
		bool bLandmarkRegressor = false,
		const char* xml_eye_leftcorner = NULL,
		const char* xml_mth_leftcorner = NULL,
		const char *sFaceRecognizer_modelpath = NULL);
	DLLEXPORT ~CxlibFaceAnalyzer(void);
	
	// init
	DLLEXPORT void init(EnumViewAngle viewAngle = VIEW_ANGLE_FRONTAL, EnumTrackerType traType = TRA_SURF, int nFaceDetector =0,
		const char *str_facesetxml =NULL,int recognierType = RECOGNIZER_CAS_GLOH, bool bEnableAutoCluster = false, bool bLandmarkRegressor = false,
		const char* xml_eye_leftcorner = NULL,
		const char* xml_mth_leftcorner = NULL,
		const char *sFaceRecognizer_modelpath = NULL);


	DLLEXPORT void loadFaceModelXML(const char *sPathXML);
	DLLEXPORT void saveFaceModelXML(const char *sPathXML, vFaceSet *pvecFaceSet =NULL);
	DLLEXPORT void getMergedFaceSet(vFaceSet& vvClusters, int minWeight = 0);

	DLLEXPORT int  insertEmptyFaceSet(char *name, bool createFolder = true, int nFaceSetID = -1);
	DLLEXPORT bool tryInsertFace(IplImage* pCutFace, int nFaceSetIdx, bool bForceInsert = false);
	DLLEXPORT int  removeFaceSet(int nFaceSetIdx);
	DLLEXPORT int  removeFace(int nFaceSetIdx, int faceIdx);

	DLLEXPORT vFaceSet *getFaceSets(); 
	DLLEXPORT char     *getFaceImgDBPath();
	DLLEXPORT IplImage *getBigCutFace();
	DLLEXPORT int       getFaceSetID(int nFaceSetIdx); 
	DLLEXPORT int       getFaceSetSize(int nFaceSetIdx = -1); 
	DLLEXPORT const char* getKeyFacePath(int nFaceSetIdx, int nFaceIdx = -1);

	//prop_estimate: #fk #fa #fs #mk #ma #ms #smile
	//pStat: %02dperson (likely [%1dMale, %1dFemale]; [%1dkid, %1dadult, %1dsenior]), Prefer
	DLLEXPORT void detect(IplImage *pImg, int *prop_estimate, char *pStat = NULL);
	DLLEXPORT bool Face_Detection(IplImage *pColorImg, int nMin_FaceSize, char *ThumbnailImgFilanem);
	//DLLEXPORT void track(IplImage *pGreyImg, int *prop_estimate, char *pStat = NULL, IplImage *pBGRImg = NULL);
	
	//predict pCutFace's facesetID and sFaceName
	DLLEXPORT int  predictFaceSet(IplImage *pCutFace, float *prob, char *sFaceName = NULL);

	DLLEXPORT int       getMaxFaceNum() { return 16;}
	DLLEXPORT int       getFaceNum();              
	DLLEXPORT int       getFaceTrackID(int idx);   
	DLLEXPORT int      *getFaceTrackIDs();         
	DLLEXPORT CvRectItem    getFaceRect(int idx);      
	DLLEXPORT CvRectItem   *getFaceRects();            
	DLLEXPORT CvPoint2D32f *getFaceLdmks(int idx);            


	DLLEXPORT int       getFaceID(int idx);      
	DLLEXPORT float     getFaceProb(int idx);      
	DLLEXPORT char     *getFaceName(int idx);
    

private:
	CxFaceAnalyzer   *m_pfaceAnalyzer;
};

/************************** CxlibFace end *****************************/

