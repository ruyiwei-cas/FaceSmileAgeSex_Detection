// PoseEstimatorDLL.cpp : Defines the entry point for the DLL application.

#include "Frontal_Realboosting_Dll.h"
#include "Common_Func.h"

#include "FaceDetection.h"
//#include "FaceDepthImage.h"
#include "FrontalModel.h"
#include "ColorGMM.h"
using namespace cv;

bool FaceDetector_Init = false;
#define MAX_DETECTOR_NUM 8
FaceDetection *FaceDetector[MAX_DETECTOR_NUM];
CColorGMM *SkinColorModel[MAX_DETECTOR_NUM];


/// ------------ the following for  face detection

//extern "C" __declspec(dllexport) 
EXPORTIT
void FaceDetection_Init(int nWinWidth, int nWinHeight, char *sColorModelFile)
{
	if (FaceDetector_Init)
		return;
	for (int i = 0; i < MAX_DETECTOR_NUM; i++)
	{
		FaceDetector[i] = new FaceDetection(WIN_WIDTH, WIN_HEIGHT);
		FaceDetector[i]->LoadDefaultFrontalDetector(nFaceDetector_Int, FaceDetector_double);

		SkinColorModel[i] = new CColorGMM();
		bool bResult;
		if (sColorModelFile != NULL)
			bResult = SkinColorModel[i]->LoadLookupTable(sColorModelFile);// "opencv_input/New_Skin_LookupTable_int8_64.dat");
		else bResult = SkinColorModel[i]->LoadLookupTable("opencv_input/New_Skin_LookupTable_int8_64.dat");
		if (!bResult)
		{
			cout << "Could not load color model" << endl;
		}

	}
	FaceDetector_Init = true;
}

//extern "C" __declspec(dllexport) 
EXPORTIT
int FrontalView_FaceDetection(int nDetector_No, IplImage *FaceImage, FdRect *faces, int parallel_flag)
{
	if (nDetector_No >= MAX_DETECTOR_NUM) return -1;
	if (nDetector_No < 0) return -1;

	int nWidth = FaceImage->width;
	int nHeight = FaceImage->height;
    IplImage *grayImage;
    if (FaceImage->nChannels != 1)
    {
        grayImage = cvCreateImage(cvSize(FaceImage->width, FaceImage->height), FaceImage->depth, 1);
        cvCvtColor(FaceImage, grayImage, CV_RGB2GRAY);
    }
    else{
		grayImage = cvCloneImage(FaceImage);
    }
	//biGammaCorrection(grayImage);

	FaceDetector[nDetector_No]->m_dDetectionScale = 1.2;

	int nSrcWidth=nWidth;
	nSrcWidth+=(nSrcWidth%4==0 ? 0 : 4-nSrcWidth%4);
	FaceDetector[nDetector_No]->SetParameter(nSrcWidth, nHeight, 1.0);

	FdAvgComp small_faces[1000];
	
	int i, nFace;
	nFace = FaceDetector[nDetector_No]->DetectFace(grayImage, NULL, small_faces, parallel_flag);
	if (nFace >= 16)
		nFace = 16;
	for(i=0 ; i<nFace ; i++)
	{
		faces[i].x = small_faces[i].rect.x ;
		faces[i].y = small_faces[i].rect.y ;
		faces[i].width = small_faces[i].rect.width ;
		faces[i].height = small_faces[i].rect.height;
		faces[i].view = 3;//Frontal view //small_faces[i].view;
		faces[i].confidence = small_faces[i].confidence;
	}

	cvReleaseImage(&grayImage);
	return nFace;
}

//extern "C" __declspec(dllexport) 
EXPORTIT
int FrontalView_ColorImage_FaceDetection(int nDetector_No, IplImage *ColorImage, FdRect *faces, bool bSkinColor, int parallel_flag)
{
	if (nDetector_No >= MAX_DETECTOR_NUM) return -1;
	if (nDetector_No < 0) return -1;

	int nWidth = ColorImage->width;
	int nHeight = ColorImage->height;
	IplImage *grayImage;
	IplImage *lpSkinImage = nullptr;
	FaceDetector[nDetector_No]->m_dDetectionScale = 1.2;

	int nSrcWidth = nWidth;
	nSrcWidth += (nSrcWidth % 4 == 0 ? 0 : 4 - nSrcWidth % 4);
	FaceDetector[nDetector_No]->SetParameter(nSrcWidth, nHeight, 1.0);


#define MAX_FACE_NUM 100
	FdAvgComp small_faces[MAX_FACE_NUM];
	int i, nFace;

	if (ColorImage->nChannels != 1)
	{
		grayImage = cvCreateImage(cvSize(ColorImage->width, ColorImage->height), ColorImage->depth, 1);
		IplImage *lpHSVImage = cvCreateImage(cvSize(ColorImage->width, ColorImage->height), ColorImage->depth, 3);
		cvCvtColor(ColorImage, lpHSVImage, CV_BGR2HSV);
		for (int i = 0; i < lpHSVImage->height;i++)
		for (int j = 0; j < lpHSVImage->width; j++)
		{
			grayImage->imageData[i*grayImage->widthStep + j] = lpHSVImage->imageData[i*lpHSVImage->widthStep + j * 3 +2];
		}
		cvReleaseImage(&lpHSVImage);
		if (bSkinColor)
		{
			lpSkinImage = cvCreateImage(cvSize(grayImage->width, grayImage->height), IPL_DEPTH_8U, 1);
			SkinColorModel[nDetector_No]->RGB_to_ColorMap(ColorImage, lpSkinImage);
			nFace = FaceDetector[nDetector_No]->DetectFace(grayImage, (unsigned char *)(lpSkinImage->imageData), small_faces, 0);
		}
		else
			nFace = FaceDetector[nDetector_No]->DetectFace(grayImage, nullptr, small_faces, 0);
		
		cvReleaseImage(&lpSkinImage);
		cvReleaseImage(&grayImage);
	}
	else
	{
		nFace = FaceDetector[nDetector_No]->DetectFace(ColorImage, nullptr, small_faces, 0);
	}

	if (nFace >= MAX_FACE_NUM)
		nFace = MAX_FACE_NUM-1;
	for (i = 0; i<nFace; i++)
	{
		faces[i].x = small_faces[i].rect.x;
		faces[i].y = small_faces[i].rect.y;
		faces[i].width = small_faces[i].rect.width;
		faces[i].height = small_faces[i].rect.height;
		faces[i].view = 3;//Frontal view //small_faces[i].view;
		faces[i].confidence = small_faces[i].confidence;
	}

	return nFace;
}

//extern "C" __declspec(dllexport) 
EXPORTIT
void FaceDetection_Release()
{
	if (!FaceDetector_Init)
		return;
	for (int i = 0; i < MAX_DETECTOR_NUM; i++)
	{
		FaceDetector[i]->Release();
		delete FaceDetector[i];
		delete SkinColorModel[i];
	}
	FaceDetector_Init = false;
}


//extern "C" __declspec(dllexport) 
EXPORTIT
void SetFaceROI(int nDetector_No, ROI_Rect FaceROI)
{
	if (nDetector_No >= MAX_DETECTOR_NUM) return ;
	if (nDetector_No < 0) return ;

	CvRect FaceRegion;
	FaceRegion.x = FaceROI.x;
	FaceRegion.y = FaceROI.y;
	FaceRegion.width = FaceROI.width;
	FaceRegion.height = FaceROI.height;

	FaceDetector[nDetector_No]->SetImageROI(FaceRegion);
}

EXPORTIT
void SetFaceROI_Ratio(int nDetector_No, double dCenterRatio)
{
	if (nDetector_No >= MAX_DETECTOR_NUM) return;
	if (nDetector_No < 0) return;

	FaceDetector[nDetector_No]->SetImageROI(dCenterRatio);
}
//extern "C" __declspec(dllexport) 
EXPORTIT
void ClearFaceROI(int nDetector_No)
{
	if (nDetector_No >= MAX_DETECTOR_NUM) return;
	if (nDetector_No < 0) return;

	FaceDetector[nDetector_No]->ClearImageROI();
}
//extern "C" __declspec(dllexport) 
EXPORTIT
void SetFaceSizeRange(int nDetector_No, int nMinSize, int nMaxSize)
{
	if (nDetector_No >= MAX_DETECTOR_NUM) return;
	if (nDetector_No < 0) return;

	FaceDetector[nDetector_No]->SetFaceSizeRange(nMinSize, nMaxSize);
}
//extern "C" __declspec(dllexport) 
EXPORTIT
void ClearFaceSizeRange(int nDetector_No)
{
	if (nDetector_No >= MAX_DETECTOR_NUM) return;
	if (nDetector_No < 0) return;

	FaceDetector[nDetector_No]->ClearFaceSizeRange();
}



