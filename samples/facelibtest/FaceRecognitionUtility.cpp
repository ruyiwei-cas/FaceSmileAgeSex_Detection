
#include "opencv/cv.h"
#include "opencv/highgui.h"	
#include "cxlibface.hpp"

#include "../Platform.h"

#ifdef __win__
#include <afx.h>
#endif
#include "MyType.h"
#include "FaceDetectionUtility.h"
#include "FaceRecognitionUtility.h"
#include "HumanClothesHistUtility.h"
#include "PhotoIndexingUtility.h"

#include "ContextBasedFaceRecognitionDLL.h"

CxlibFaceRecognizer faceRecognizer( size_bigface);

CxlibBlinkDetector  blinkDetector(size_smallface);
CxlibSmileDetector  smileDetector(size_smallface);
CxlibGenderDetector genderDetector(size_smallface);
CxlibAgeDetector    ageDetector(size_bigface);
CxlibLandmarkDetector landmarkDetector(LDM_6PT);
CxlibAlignFace cutFace(size_smallface, size_bigface);


int Face_Valid_Flag[MAX_FACE_NUMBER];	
int nFaceSetSize;
Face_Attribute FaceRecognitionResult[MAX_FACE_NUMBER];
Face_Attribute ProfileFaceRecognitionResult[MAX_FACE_NUMBER];
////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the Face Recognition Utility
///  written by Ren Haibing 
///   the face model is in the default directory
/// Acknowledge		: modified from Yimin's program
///
/// Function name	: InitFaceRecognition
/// Description	    : init face recognizer 
///
/// Argument		:	
///
/// Return type		: 
///
/// Create Time		: 2014-11-2  14:20
///
///
/// Side Effect		: 
///
///////////////////////////////////////////////////////////////////////////////////////////////
void InitFaceRecognition()
{
	char *str_facesetxml = "faceset_model.xml";
	faceRecognizer.loadFaceModelXML("faceset_model.xml");
	nFaceSetSize= faceRecognizer.getFaceSetSize();
}

////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the Face Recognition Utility
///  written by Ren Haibing 
/// Acknowledge		: modified from Yimin's program
///
/// Function name	: GetRecognizedFaceName
/// Description	    : Get the face name with the recognized face ID 
///
/// Argument		:	nFaceSetID -- face ID in the face model DB
///
/// Return type		: 
///
/// Create Time		: 2014-11-5  10:50
///
///
/// Side Effect		: 
///
///////////////////////////////////////////////////////////////////////////////////////////////
const char *GetRecognizedFaceName(int nFaceSetID)
{
	return faceRecognizer.getFaceName(nFaceSetID);
}

////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the Face Recognition Utility
///  written by Ren Haibing 
/// Acknowledge		: modified from Yimin's program
///
/// Function name	: FaceRecognitionApplication
/// Description	    : face identification and face attribute recognition 
///
/// Argument		:	color_image -- source image
/// Argument		:	nFace_Num -- detected face number
/// Argument		:	rects -- detected face region
///
/// Return type		: 
///
/// Create Time		: 2014-11-5  10:56
///
///
/// Side Effect		: int Face_Valid_Flag[MAX_face_numBER] -- face valid flag array
///                   Face_Attribute FaceRecognitionResult[MAX_face_numBER] -- final recogntion result 
///////////////////////////////////////////////////////////////////////////////////////////////
void FaceRecognitionApplication(IplImage* color_image, int nFace_Num, CvRectItem* rects)
{
	bool   DoBlink = true, DoSmile = true, DoGender = true, DoAge = true;
	float  smile_threshold, blink_threshold, gender_threshold; 
	int    bBlink = 0, bSmile = 0, bGender = 0;  //+1, -1, otherwise 0: no process 
	int    nAgeID = 0;
	float  probBlink = 0, probSmile = 0, probGender = 0, probAge[4];

	// config landmark detector ------------------------------------


	bool  bLandmark = false;
	CvPoint2D32f   landmark6[6+1]; // consider both 6-pt and 7-pt

	float probFaceID;
	int nFaceSetID;

	// blink/smile/gender/age/face recognize section
	for( int i=0; i< nFace_Num; i++ )
	{
		Face_Valid_Flag[i] = 0;
		bSmile = bBlink = bGender = -1;
		probSmile = 0, probBlink = 0, probGender = 0;		

		// get face rect and id from face tracker
		CvRect rect = rects[i].rc;
		int    face_trackid = rects[i].fid;
		float  like = rects[i].prob;
		int    angle= rects[i].angle;
				
		FaceRecognitionResult[i].FaceRegion = rect;
		FaceRecognitionResult[i].FaceView = 0;//frontal view

		// filter out outer faces
		if (rect.x + rect.width  > color_image->width || rect.x < 0) continue;
		if (rect.y + rect.height > color_image->height || rect.y < 0) continue;
		if (rect.width<color_image->width * 0.03) continue;
				
		// Landmark detection -----------------------------------------------------
		bLandmark = landmarkDetector.detect(color_image, &rect, landmark6, NULL, angle); //for imagelist input
		if(bLandmark == false) continue;
		cutFace.init(color_image, rect, landmark6);

		Face_Valid_Flag[i] = 1;   

		// detect blink----------------------------------------------
		bBlink = 0;	
		probBlink = 0;
		if (DoBlink)
		{
			blink_threshold = blinkDetector.getDefThreshold();//0.5;
			int ret = blinkDetector.predict( &cutFace, &probBlink);
					
			if(probBlink > blink_threshold )
				bBlink = 0;//1; //eye close
			else 
				bBlink = 1;//0; //eye open
			FaceRecognitionResult[i].Blink = bBlink;
			FaceRecognitionResult[i].Prob_Blink = probBlink;
		}

		// detect smile -----------------------------------------------------------
		bSmile    = 0;	
		probSmile = 0;
		if (DoSmile)
		{	
			smile_threshold = smileDetector.getDefThreshold(); //0.42;  
			int ret = smileDetector.predict(&cutFace, &probSmile);

			if(probSmile > smile_threshold)
				bSmile = 1;  //smile
			else 
				bSmile = 0; //not smile
			FaceRecognitionResult[i].Smile = bSmile;
			FaceRecognitionResult[i].Prob_Smile = probSmile;
		}
			
		//detect gender --------------------------------------------------------
		bGender    = 0;	
		probGender = 0;
		if(DoGender)
		{
			gender_threshold = genderDetector.getDefThreshold(); // 0.42; 

			cvSaveImage("c:/temp/gender.jpg", cutFace.getBigCutFace());
			int ret = genderDetector.predict(&cutFace, &probGender);


			if(probGender > gender_threshold)
				bGender =  1; //female
			else
				bGender =  0; //male
			FaceRecognitionResult[i].Gender = bGender;
			FaceRecognitionResult[i].Prob_Gender = probGender;
		}

		// estmage age -------------------------------------------------------------
		if(DoAge)
		{
			//nAgeID = 0:"Baby", 1:"Kid", 2:"Adult", 3:"Senior"
			nAgeID = ageDetector.predict(&cutFace, probAge);
			FaceRecognitionResult[i].Age = nAgeID;
			FaceRecognitionResult[i].Prob_Age = probAge[nAgeID];
		}

		//Face Recognition ---------------------------------------------------------
		if(bLandmark) // aligned face is needed
		{
			nFaceSetID = faceRecognizer.predict(&cutFace, &probFaceID);
			FaceRecognitionResult[i].FaceID = nFaceSetID;
			FaceRecognitionResult[i].Prob_FaceID = probFaceID;
		}

	}//for( int i=0; i< nFace_Num; i++ )

	// post-processing for the FaceID
	FaceID_PostProcessing2(FaceRecognitionResult, Face_Valid_Flag, nFace_Num);
}

////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the Face Recognition Utility
///  written by Ren Haibing 
/// Acknowledge		: None
///
/// Function name	: GetFrontalFace
/// Description	    : get face region from  face attribute recognition resutl 
///
/// Argument		:	FrontalFaceRegoin -- face region
/// Argument		:	nFace_Num -- face number
///
/// Return type		:  
///
/// Create Time		: 2014-12-29  10:31
///
///
/// Side Effect		: 
///                   
///////////////////////////////////////////////////////////////////////////////////////////////
void GetFrontalFace(Rect *FrontalFaceRegoin, int nFace_Num)
{
	for(int i=0;i<nFace_Num;i++)
	{
		FrontalFaceRegoin[i].x = FaceRecognitionResult[i].FaceRegion.x;
		FrontalFaceRegoin[i].y = FaceRecognitionResult[i].FaceRegion.y;
		FrontalFaceRegoin[i].width = FaceRecognitionResult[i].FaceRegion.width;
		FrontalFaceRegoin[i].height = FaceRecognitionResult[i].FaceRegion.height;
	}
}
////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the Face Recognition Utility
///  written by Ren Haibing 
/// Acknowledge		: None
///
/// Function name	: ProfileFaceRecognition
/// Description	    : recognize profile faces with clothes histogram
///
/// Argument		:	lpFilename -- file name
///
/// Return type		:   int -- face number
///
/// Create Time		: 2014-11-19  13:05
///
///
/// Side Effect		: 
///                   
///////////////////////////////////////////////////////////////////////////////////////////////
#define PROFILE_PROBILITY_THRESHOLD 0.2
int ProfileFaceRecognition(char *sFilename, int *FrontalFaceNum, Rect*  FrontalFaceRegoin)
{
	IplImage *color_image = ReadImage_Withexif(sFilename);
	if (color_image == NULL) return -2; // file not exist

	//cout << sFilename << endl;
	char sDate[12];
	int nValidDate = GetFileDate(sFilename, sDate);
	if(nValidDate != 1)
	{
		cvReleaseImage(&color_image);
		return -1; // no date information
	}

		// face detection
	IplImage *gray_image = cvCreateImage(cvGetSize(color_image),IPL_DEPTH_8U,1);
	cvCvtColor(color_image, gray_image, CV_RGB2GRAY);	

	Rect FaceRegion[MAX_FACE_NUMBER];

	//3.4 detect profile face region		

	double probFaceID;

	char sRecognitionResultFilename[1024];
	char sImgPath1[1024]; 
	char tmpStr1[1024]; 
	strcpy(tmpStr1, sFilename); 
	char* firstdot1 = strrchr(tmpStr1,'.');
	*firstdot1 = NULL;
	strcpy(sImgPath1, tmpStr1);   
	
	//output face detection and recongition image
	sprintf(sRecognitionResultFilename, "%sFDFR14.dat", tmpStr1);  

	int nFrontalFace_Num = LoadProfilingFile(sRecognitionResultFilename, FaceRecognitionResult, Face_Valid_Flag);

	GetFrontalFace(FrontalFaceRegoin, nFrontalFace_Num);
	//cvSaveImage("c:\\Temp\\1.jpg", gray_image);
	int face_num = DetectProfileFace(gray_image, FaceRegion,FrontalFaceRegoin, nFrontalFace_Num);	
	
	for(int i=0;i<face_num;i++)
	{
		ProfileFaceRecognitionResult[i].FaceRegion = FaceRegion[i];
		GetClothesFeature(color_image, &(ProfileFaceRecognitionResult[i]));

		ProfileFaceRecognitionResult[i].FaceID = PredictID_With_Clothes_matching(ProfileFaceRecognitionResult[i].Attribute_Feature, sDate, &probFaceID, true);
		if (ProfileFaceRecognitionResult[i].FaceID >= 0) // valid face ID
		{
			for (int j = 0; j < nFrontalFace_Num; j++)
			if (ProfileFaceRecognitionResult[i].FaceID == FaceRecognitionResult[j].FaceID)
			{
				ProfileFaceRecognitionResult[i].FaceID = -1;
				break;
			}

			ProfileFaceRecognitionResult[i].Prob_FaceID = probFaceID;
			if (probFaceID < PROFILE_PROBILITY_THRESHOLD)
				ProfileFaceRecognitionResult[i].FaceID = -1;
		}
	}
		
	// remove the two same IDs in the same image
	for (int i = 0; i<face_num; i++)
		if (ProfileFaceRecognitionResult[i].FaceID >= 0)
		{
			for (int j = i + 1; j<face_num; j++)
			{
				if (ProfileFaceRecognitionResult[i].FaceID == ProfileFaceRecognitionResult[j].FaceID)
				{
					if (ProfileFaceRecognitionResult[i].Prob_FaceID < ProfileFaceRecognitionResult[j].Prob_FaceID)
						ProfileFaceRecognitionResult[i].FaceID = -1;
					else ProfileFaceRecognitionResult[j].FaceID = -1;
				}
			}
		}

	//output face detection and recongition image
	sprintf(sRecognitionResultFilename, "%sFDFR15.dat", tmpStr1);  
	WriteProfilingFile(sRecognitionResultFilename, face_num, ProfileFaceRecognitionResult, nullptr);

	cvReleaseImage(&gray_image);
	cvReleaseImage(&color_image);
	return face_num;
}