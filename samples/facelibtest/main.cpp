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
#include <stdlib.h>

#include "cxoptions.hpp"

#include "opencv/cv.h"
#include "opencv/highgui.h"
#include "basetypes.hpp"
#include "cxlibface.hpp"

#include "libexif/exif-data.h"  //for libexif library

// for Linux
#include "../Platform.h"

#define _MY_DEBUG

//for scene classification
#include <opencv2/opencv.hpp>
#include <time.h>
#include "SceneClassificationDLL.h"

using namespace cv;
using namespace std;

//#include <afx.h>
#include "MyType_Main.h"
#include "FaceDetectionUtility.h"
#include "FaceRecognitionUtility.h"
#include "HumanClothesHistUtility.h"
#include "FaceRegistrationUtility.h"
#include "PhotoIndexingUtility.h"

extern Face_Attribute FaceRecognitionResult[MAX_FACE_NUMBER];
extern Face_Attribute ProfileFaceRecognitionResult[MAX_FACE_NUMBER];
extern int Face_Valid_Flag[MAX_FACE_NUMBER];
extern int nFaceSetSize;

//#pragma comment ( lib, "vl.lib")   //for vlfeature library 

#pragma warning ( disable: 4996 )
#pragma warning ( disable: 4305 )
#pragma warning ( disable: 4244 )

static const char* helpmsg =
	"\n"
    "Usage:  testfacelib [options] <video|camera>\n"
    "\n"
    "Video:\n"
    "  Input can be video file, camera seq or image list.\n"
    "\n"
    "  Options: \n"
    "  -m mode      choose 'bsgar' mode for blink/smile/gender/age/face recognition mode, \n"
	"  -t threads   number of threads, default is 1 to recognize each frame. if threads > 1, it will use another pthread to recognize the faces. \n"
    "  -x facexml   face set model xml file for face recognition, default is bin/faceset_model.xml \n"
    "  -a           use Haar face tracker, otherwise use surf tracker, default is surf face tracker if no haar\n"
	"  -p           use Particle filter face tracker, otherwise use surf tracker, default is surf face tracker if no particle filter\n"
	"  -P           use Particle filter meanshift face tracker, otherwise use surf tracker, default is surf face tracker if no particle filter\n"
	"  -v           use multiview face tracker, otherwise frontal view face tracker\n"
	"  -V           use multi-roll face tracker, otherwise frontal view face tracker"
	"  -g           use gabor face recognizer, otherwise cascade gloh face recognizer, default is cascade gloh face recognizer \n"
    "  -q           quiet, no window GUI and no instant information. \n"
    "  -s           save unrecognized face icons. \n"
    "  -h           help message. \n"

    "Examples: \n"
    "#  for video file testing mode  \n"
    "     testfacelib.exe -m sgar  ./video_data/group2.mpg \n"
    "\n"
    "#for online camera testing mode   \n"
    "     testfacelib.exe -m bsgar  0     \n"                                    
	"\n"
	"#for very fast online camera testing mode   \n"
	"     testfacelib.exe -t 2 -m bsgar  0     \n"                                    
	"\n"
	"#for image list testing mode   \n"
	"     testfacelib.exe -m bsgar  imageList.txt \n"                                    
	"\n"
    ;

static void help( const char* errmsg )
{
    if( errmsg != NULL )
    {
        fprintf( stderr, "testfacelib: %s\n", errmsg );
        fprintf( stderr, "Usage: testfacelib [options] <video>\n" );
        fprintf( stderr, "\nTry 'testfacelib -h' for more information.\n" );
    }
    else
        fprintf( stderr, "%s", helpmsg );
    exit( -1 );
}

IplImage* imgRotation90n(IplImage* srcImage, int angle);

int evaluateImgList(int argc, char *argv[])
{
	//init param
	if( argc < 3 )
	{
		//printf("testfacelib.exe -i imgList.txt <startImgID> <endImgID>\n");
		return 0;
	}

	char *sImgList         = argv[2];

	char *sDesFolderName =NULL;
	if( argc == 4 )
	{
		sDesFolderName = argv[3];
	}
	char sResultFile[1024]; 
	sprintf(sResultFile, "%s_result.log", sImgList);

	FILE *fpImgList    = fopen(sImgList, "rt");
	if (fpImgList == nullptr)
	{
		cout << sImgList << "  doesn't exist" << endl;
		return -1;
	}

	//read image list
	char sPath[1024];
	int imgNo=0;


#ifdef _MY_DEBUG
	float probFaceID;
	int nFaceSetID;
	IplImage *color_image_show = NULL;
	int color_image_show_width;
	double dcolor_image_show_scale;
	int color_image_show_height;
#endif


	while(fgets(sPath, 1024, fpImgList))
	{
		cout<<"---------------------------------------------------------"<<endl;
		cout<<sPath<<endl;
		if (sPath[strlen(sPath)-1]=='\n') //remove the end \n in file name, or it will cause error in reading file (e.g. libexif).
			sPath[strlen(sPath)-1]='\0';

		IplImage *color_image= ReadImage_Withexif(sPath);
		if(color_image == NULL) continue;
#ifdef _MY_DEBUG 
			color_image_show_width = 1024;
			dcolor_image_show_scale =  color_image_show_width * 1.0/color_image->width;
			color_image_show_height = int(color_image->height * dcolor_image_show_scale);
			color_image_show = cvCreateImage( cvSize(color_image_show_width, color_image_show_height), IPL_DEPTH_8U, color_image->nChannels );
			cvResize(color_image, color_image_show);
#endif	

		int scene_label=0;

		int face_num = PhotoIndexing_ImageEvaluation(sPath, &scene_label);

#ifdef _MY_DEBUG	// save the result into the images
		for(int i=0;i<face_num;i++)
		{
			CvRect rect = FaceRecognitionResult[i].FaceRegion;
			if(Face_Valid_Flag[i] >0)
			{
				probFaceID = FaceRecognitionResult[i].Prob_FaceID;
				nFaceSetID = FaceRecognitionResult[i].FaceID;
				if(probFaceID<0.10)  nFaceSetID = -1;	
				const char *name = GetRecognizedFaceName(nFaceSetID);
				char sFaceName[256];
				strcpy(sFaceName, name);

				CvRect Rect_show;
				Rect_show.x = rect.x * dcolor_image_show_scale;
				Rect_show.y = rect.y * dcolor_image_show_scale;
				Rect_show.width = rect.width * dcolor_image_show_scale;
				Rect_show.height = rect.height * dcolor_image_show_scale;
				//draw face detection rectangle
				cxlibDrawFaceRect(color_image_show, Rect_show), CV_RGB(0, 255, 0);

				CvFont *pFont = new CvFont;
				cvInitFont(pFont, CV_FONT_HERSHEY_PLAIN, 2, 2, 2, 4);
				char text[256];
				memset(text, 0, 256);
				sprintf(text, "%s(%f)", sFaceName, probFaceID);  
				cvPutText(color_image_show, text, cvPoint(Rect_show.x, (Rect_show.y - 10)), pFont, CV_RGB(0, 255, 0));

				memset(text, 0, 256);
				switch (FaceRecognitionResult[i].Age)
				{
				case 0:	sprintf(text, "Baby");
					break;
				case 1:	sprintf(text, "Kid");
					break;
				default:
					break;
				}
				
				//sprintf(text, "%s", sFaceName); 
			//	cvPutText(color_image_show, text, cvPoint(Rect_show.x, (Rect_show.y + Rect_show.height + 15)), pFont, CV_RGB(0, 255, 0));
				delete pFont;
				
			}
			else
			{
				CvRect Rect_show;
				Rect_show.x = rect.x * dcolor_image_show_scale;
				Rect_show.y = rect.y * dcolor_image_show_scale;
				Rect_show.width = rect.width * dcolor_image_show_scale;
				Rect_show.height = rect.height * dcolor_image_show_scale;
				//draw face detection rectangle
			//	cxlibDrawFaceRect(color_image_show, Rect_show, CV_RGB(0, 0, 255));
			}
		}//for(i=0;i<face_num;i++)
		//--------------------------------------------------------------------------
		
		CvFont *pFont = new CvFont;
		cvInitFont(pFont, CV_FONT_HERSHEY_PLAIN, 6, 6, 2, 2);

		char sSceneName[20];
		memset(sSceneName, 0, 20);
		GetSceneName(scene_label, sSceneName);
		cvPutText(color_image_show, sSceneName, cvPoint(10, 150), pFont, CV_RGB(0, 255, 255));

		delete pFont;
		
		//output face detection and recongition image
		char sImgPath1[1024]; 
		char tmpStr1[1024]; 
		strcpy(tmpStr1, sPath); 

		if(sDesFolderName == NULL)
		{
			char* firstdot1 = strrchr(tmpStr1,'.');
			*firstdot1 = NULL;
			strcpy(sImgPath1, tmpStr1);     
			sprintf(sImgPath1, "%sFDFR11.jpg", tmpStr1);  
		}
		else
		{
			char* lpPureName = strrchr(tmpStr1,'/');
			if(lpPureName == NULL)
				lpPureName = strrchr(tmpStr1,'\\');
			if(lpPureName == NULL)
				lpPureName = tmpStr1;
			else lpPureName++;
			char* firstdot1 = strrchr(lpPureName,'.');
			*firstdot1 = NULL;
			sprintf(sImgPath1, "%s/%sFDFR21.jpg",sDesFolderName, lpPureName);  
		}
		
		cvSaveImage(sImgPath1, color_image_show);
//		cvSaveImage("c:/temp/1.jpg", color_image_show);
//		cvShowImage("image", color_image_show);
//		cvWaitKey(10);
		cvReleaseImage(&color_image_show);
#endif 	

		cvReleaseImage(&color_image);

		imgNo++;
		//printf(" %d\r", imgNo);		
	}//while(!feof(fpImgList))

	
	fclose(fpImgList);
	return 0;
}


//evaluate face attribute detection
int ImgListProfileFaceRecognition(int argc, char *argv[])
{
	//init param
	char *sImgList         = argv[2];
	char *sDesFolderName =NULL;
	if( argc == 4 )
	{
		sDesFolderName = argv[3];
	}

	FILE *fpImgList         = fopen(sImgList, "rt");

	//read image list
	char sPath[1024];	
	int imgNo=0;
	
#ifdef _MY_DEBUG
	IplImage* color_image;
	IplImage *color_image_show = NULL;
	double dcolor_image_show_scale;
#endif

	double probFaceID;
	int nFaceSetID;
	while(fgets(sPath, 1024, fpImgList))
	{
		cout<<"---------------------------------------------------------"<<endl;
	
		if (sPath[strlen(sPath)-1]=='\n') //remove the end \n in file name, or it will cause error in reading file (e.g. libexif).
			sPath[strlen(sPath)-1]='\0';
		cout << sPath << endl;
		int nFrontalFaceNum;
		Rect FrontalFaceRegoin[16];
		int face_num = ProfileFaceRecognition(sPath, &nFrontalFaceNum, FrontalFaceRegoin);

		cout << "Profile Face Num: " << face_num <<endl;
		if (face_num ==-2)
			continue; // image doesn't exist
	
		char sProfileImgPath[1024]; 
		char sFrontalImgPath[1024];
		char tmpStr1[1024]; 
		strcpy(tmpStr1, sPath); 
		//output face detection and recongition image
		if (sDesFolderName == NULL)
		{
			char* firstdot1 = strrchr(tmpStr1, '.');
			*firstdot1 = NULL;
			strcpy(sProfileImgPath, tmpStr1);
			sprintf(sProfileImgPath, "%sFDFR15.jpg", tmpStr1);
			sprintf(sFrontalImgPath, "%sFDFR14.jpg", tmpStr1);			
		}
		else
		{
			char* lpPureName = strrchr(tmpStr1, '/');
			if (lpPureName == NULL)
				lpPureName = strrchr(tmpStr1, '\\');
			if (lpPureName == NULL)
				lpPureName = tmpStr1;
			else lpPureName++;
			char* firstdot1 = strrchr(lpPureName, '.');
			*firstdot1 = NULL;
			sprintf(sProfileImgPath, "%s/%sFDFR15.jpg", sDesFolderName, lpPureName);
			sprintf(sFrontalImgPath, "%s/%sFDFR14.jpg", sDesFolderName, lpPureName);
		}
		color_image = ReadImage_Withexif(sPath);
		color_image_show = cvLoadImage(sFrontalImgPath);
		dcolor_image_show_scale = color_image_show->width * 1.0 / color_image->width;

		for (int i = 0; i<face_num; i++)
		{
			CvRect rect = ProfileFaceRecognitionResult[i].FaceRegion;
				
			probFaceID = ProfileFaceRecognitionResult[i].Prob_FaceID;
			if (probFaceID < 0.2)
				continue;
			nFaceSetID = ProfileFaceRecognitionResult[i].FaceID;
			const char *name = GetRecognizedFaceName(nFaceSetID);
			char sFaceName[256];
			strcpy(sFaceName, name);

			CvRect Rect_show;
			Rect_show.x = rect.x * dcolor_image_show_scale;
			Rect_show.y = rect.y * dcolor_image_show_scale;
			Rect_show.width = rect.width * dcolor_image_show_scale;
			Rect_show.height = rect.height * dcolor_image_show_scale;
			//draw face detection rectangle
			cxlibDrawFaceRect(color_image_show, Rect_show, CV_RGB(255, 0, 0));
			
			char text[256];
			CvFont *pFont = new CvFont;
			cvInitFont(pFont, CV_FONT_HERSHEY_PLAIN, 2, 2, 2, 4);
				sprintf(text, "%s(%f)", sFaceName, ProfileFaceRecognitionResult[i].Prob_FaceID); 
 
			//sprintf(text, "%s", sFaceName); 
			cvPutText( color_image_show, text, cvPoint(Rect_show.x, (Rect_show.y-10)), pFont,  CV_RGB(255,0,0));
			delete pFont;

		}//for(i=0;i<face_num;i++)
		//--------------------------------------------------------------------------
		cvSaveImage(sProfileImgPath, color_image_show);
		cvReleaseImage(&color_image_show);
		cvReleaseImage(&color_image);

		imgNo++;
		//printf(" %d\r", imgNo);
	}//while(!feof(fpImgList))

	fclose(fpImgList);

	return 0;
}

int FaceRegistration(int argc, char *argv[])
{
	//init param
	if( argc < 3 )
	{
		//printf("testfacelib.exe -r imgList.txt sUserName\n");
		return 0;
	}
	
	char *sImgList         = argv[2];
	char *sUserName		   = argv[3];
	// init faceli
	//cout << "Begin to register: " << sUserName << endl;
	//cout << "List file name " << sImgList << endl;

	FILE *fpImgList         = fopen(sImgList, "rt");
	if (fpImgList == nullptr)
	{
		cout << " List file doesn't exist" << endl;
		return -1;
	}
	//read image list
	char sPath[1024];	
	int imgNo=0;

	IplImage *m_FaceTemplate[FACE_TEMPLATE_MAX_NUM];
	for(int i=0;i<FACE_TEMPLATE_MAX_NUM;i++)
		m_FaceTemplate[i] = NULL;

	int nTotalFaceNum = 0;
	while(fgets(sPath, 1024, fpImgList))
	{
		if (sPath[strlen(sPath)-1]=='\n') //remove the end \n in file name, or it will cause error in reading file (e.g. libexif).
			sPath[strlen(sPath)-1]='\0';		
		
		int nFaceNum = FaceRegistration_DetectFace(sPath, "c:/TEMP/Thumbnail",m_FaceTemplate,&nTotalFaceNum);

		imgNo++;
		//printf("Face Detection: %d\r", imgNo);

	}//while(!feof(fpImgList))

	fclose(fpImgList);
	FaceRegistration_AddUser(sUserName, m_FaceTemplate,nTotalFaceNum);

	for(int i=0;i<FACE_TEMPLATE_MAX_NUM;i++)
	{
		if(m_FaceTemplate[i] != NULL) cvReleaseImage(&(m_FaceTemplate[i]));
	}
	return 0;
}


int main_reg_face();
int main( int argc, char* argv[] )
{
//	main_reg_face();
//  return 1;
    // default command parameters
	// must be stumped based classifier, only this classifier can be SIMDized
	int         trackerType       = TRA_SURF;
	int         multiviewType     = VIEW_ANGLE_FRONTAL;
	int         recognizerType    = RECOGNIZER_CAS_GLOH;
	const char* str_facesetxml    = "faceset_model.xml";
	bool        bEnableAutoCluster= false;

	const char* sfolder = NULL;
	const char* video    = NULL;
	bool        blink    = false;
	bool        smile    = false;
	bool        gender   = false;
	bool        age		 = false;
	bool        recog    = false;
	bool        saveface = false;

	bool        quiet    = false;
    int         threads  = 1;  //recognize each frame if 1; otherwise recognize by pThread for faster speed
	
	bool        evalPairList = false;
	bool        evalImgList  = false;

    // option parsing
	const char* opts = "x:m:t:s:bsgar:apPvVgdoqshei";
	//const char* opts = "x:m:t:bsgar:apPvVgdoqshei";
    CxOptions opt( argc, argv, opts );

	PhotoIndexing_Init();

    char ch;
    while( (ch = opt.get()) != -1 )
    {
        const char* optarg = opt.arg();
        float value = (optarg == NULL) ? 0 : (float)atof(optarg); 
        int   optopt = opt.opt();
        switch( ch )
        {
		case 'p': //detect and recognize profile faces
			ImgListProfileFaceRecognition(argc, argv);
			break;
		case 'e': //evaluate match/nomatch pair list
			HumanClothesHist_Generation((char *)(argv[2]));
			break;
		case 'i': //evaluate img list with personID
			evaluateImgList(argc, argv); //evaluate imgList with personID
			HumanClothesHist_Generation((char *)(argv[2]));
			break;
		case 'r': // face tempalte registration
			FaceRegistration(argc, argv); 
			break;
        }
    }

	PhotoIndexing_Release();

    // non-option part
    if( opt.ind() == argc - 1 )
        video = argv[opt.ind()];
    else
        help( "missing input video." );

	if(sfolder)
		//printf("#save faces in %s\n", sfolder);
	return 1;
	// call the testing example
//	int status = consoleTestFaceLib( video, trackerType, multiviewType, recognizerType, str_facesetxml, threads, 
//		                             blink, smile, gender, age, recog, quiet, saveface, sfolder, bEnableAutoCluster );
	
    // return code
//    return status;
}




void init_face_reg(CxlibFaceAnalyzer *& ptrFaceAnalyzer)
{
	if (ptrFaceAnalyzer == NULL) {
		EnumTrackerType traType = TRA_HAAR; //TRA_PF;
		EnumViewAngle   viewAngle = VIEW_ANGLE_FRONTAL; //VIEW_ANGLE_HALF_MULTI; //

		bool bBlink = false;
		bool bSmile = false;
		bool bGender = false;
		bool bAge = false;
		bool bFaceRecog = true;

		int  sampleRate = 1;
		char str_facesetxml[] = "faceset_model.xml";
		int recognizerType = RECOGNIZER_CAS_GLOH;  //RECOGNIZER_BOOST_GB240
		bool bEnableAutoCluster = false;//false;//true;
		bool bEnableShapeRegressor = true;//false;//true;

		ptrFaceAnalyzer = new CxlibFaceAnalyzer(
			viewAngle, traType, 0,
			str_facesetxml, recognizerType, bEnableAutoCluster, bEnableShapeRegressor);
	}
}

int main_reg_face()
{
	InitFaceDetector();

	char * sUserName = "UserName";
	CxlibFaceAnalyzer * ptrFaceAnalyzer = nullptr;
	init_face_reg(ptrFaceAnalyzer);
	int nFaceSetIdx = ptrFaceAnalyzer->insertEmptyFaceSet(sUserName);
	bool bGetGoodFace;
	int Image_Num = 1;
	char * sFileName;

	IplImage * color_image;

	for (int i = 0; i < Image_Num; i++) {
		//load color image
		color_image = cvLoadImage("c:/temp/Picture6.jpg");
		bGetGoodFace = ptrFaceAnalyzer->Face_Detection(color_image, 80, "c:/temp/temp.jpg");
		if (bGetGoodFace) {
			IplImage *lpFace = ptrFaceAnalyzer->getBigCutFace();
			ptrFaceAnalyzer->tryInsertFace(lpFace, nFaceSetIdx, true);
		}
	}
	ptrFaceAnalyzer->saveFaceModelXML("faceset_model.xml");
	delete ptrFaceAnalyzer;
	ptrFaceAnalyzer = nullptr;

	return 1;
}