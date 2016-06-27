#pragma once
// The following ifdef block is the standard way of creating macros which make exporting 
// from a DLL simpler. All files within this DLL are compiled with the POSEESTIMATORDLL_EXPORTS
// symbol defined on the command line. this symbol should not be defined on any project
// that uses this DLL. This way any other project whose source files include this file see 
// POSEESTIMATORDLL_API functions as being imported from a DLL, wheras this DLL sees symbols
// defined with this macro as being exported.


#include <opencv\cv.h>
#include <opencv\cxcore.h>
#include <opencv\highgui.h>

#include "FD_Data.h"

#include "ColorGMM.h"
#include "FaceDetection.h"

#ifdef WIN32
	#define EXPORTIT	extern "C" __declspec(dllexport) 
#else
	#define EXPORTIT
	#define nullptr	NULL
#endif

EXPORTIT
void FaceDetection_Init(int nWinWidth, int nWinHeight, char *sColorModelFile);

EXPORTIT
void FaceDetection_Release();

EXPORTIT
void SetFaceROI(int nDetector_No, ROI_Rect FaceROI);

EXPORTIT
void SetFaceROI_Ratio(int nDetector_No, double dCenterRatio);
EXPORTIT
void ClearFaceROI(int nDetector_No);

EXPORTIT
void SetFaceSizeRange(int nDetector_No, int nMinSize, int nMaxSize);

EXPORTIT
void ClearFaceSizeRange(int nDetector_No);


// ------------ the following for frontal face detection

EXPORTIT
int FrontalView_FaceDetection(int nDetector_No, IplImage *FaceImage, FdRect *faces, int parallel_flag = 0);

EXPORTIT
int FrontalView_ColorImage_FaceDetection(int nDetector_No, IplImage *ColorImage, FdRect *faces, bool bSkinColor, int parallel_flag = 0);
