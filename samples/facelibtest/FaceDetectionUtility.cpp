
#include "opencv/cv.h"
#include "opencv/highgui.h"	

#include "cxlibface.hpp"

#include "libexif/exif-data.h"  //for libexif library

#include "MyType.h"
#include "FaceRecognitionUtility.h"

IplImage* imgRotation90n(IplImage* srcImage, int angle);

CxlibFaceDetector detector;
////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the Face Detection Utility
///  written by Ren Haibing 
///   the face model is in the default directory
/// Acknowledge		: modified from Yimin's program
///
/// Function name	: InitFaceDetector
/// Description	    : init face detector 
///
/// Argument		:	
///
/// Return type		: 
///
/// Create Time		: 2014-10-28  14:20
///
///
/// Side Effect		: 
///
///////////////////////////////////////////////////////////////////////////////////////////////
void InitFaceDetector()
{
	tagDetectConfig configParam;
	EnumViewAngle  viewAngle = (EnumViewAngle)VIEW_ANGLE_FRONTAL;
	detector.init(viewAngle, FEA_HAAR, 2);//(EnumFeaType)trackerType);
//	detector.config( configParam );
}

////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the Face Detection Utility
///  written by Ren Haibing 
///   the face model is in the default directory
/// Acknowledge		: modified from Yimin's program
///
/// Function name	: FaceDetectionApplication
/// Description	    : face detection and rotate the image with face detection result(in the scenio of no exif file)
///
/// Argument		:	color_image -- source color image
/// Argument		:	gray_image -- source gray image
/// Argument		:	rects -- detected face region
/// Argument		:	MAX_face_numBER -- maximal face number
/// Argument		:	imgExif -- image exif information
///
/// Return type		:  int -- detected face number
///
/// Create Time		: 2014-10-28  16:20
///
///
/// Side Effect		: 
///
///////////////////////////////////////////////////////////////////////////////////////////////
#define LARGE_IMAGE_SIZE  20000
#define STANDARD_IMAGE_WIDTH_LONG  2000
#define STANDARD_IMAGE_WIDTH_SMALL  1200
int FaceDetectionApplication(IplImage* color_image, CvRectItem* rects, int MAX_face_numBER, bool bRotateTry)
{
	// 3.1 face detection
	int face_num;
	IplImage *Detect_Image;
	int nNewWidth, nNewHeight;
	double dScale;
	if ((color_image->width>LARGE_IMAGE_SIZE) || (color_image->height>LARGE_IMAGE_SIZE))
	{
		if (color_image->width>color_image->height)
			nNewWidth = STANDARD_IMAGE_WIDTH_LONG;
		else nNewWidth = STANDARD_IMAGE_WIDTH_SMALL;
		
		dScale = nNewWidth * 1.0 / color_image->width;
		
		nNewHeight = int(color_image->height * dScale);
		Detect_Image = cvCreateImage(cvSize(nNewWidth, nNewHeight), IPL_DEPTH_8U, color_image->nChannels);
		cvResize(color_image, Detect_Image);
//		detector.SetFaceDetectionSizeRange(Detect_Image);
//		detector.SetFaceDetectionROI(Detect_Image, 0.8);
		
		face_num = detector.detect( Detect_Image, rects, 0); 
//		detector.ClearFaceDetectionRange();
		//detector.ClearFaceDetectionROI();
		for (int i = 0; i<face_num; i++)
		{
			rects[i].rc.x = int(rects[i].rc.x  / dScale);
			rects[i].rc.y = int(rects[i].rc.y  / dScale);
			rects[i].rc.width = int(rects[i].rc.width  / dScale);
			rects[i].rc.height = int(rects[i].rc.height  / dScale);
		}
	}
	else
	{
		Detect_Image = cvCloneImage(color_image);
	//	detector.SetFaceDetectionSizeRange(Detect_Image);
	//	detector.SetFaceDetectionROI(Detect_Image, 0.8);
		for (int m = 0; m < 10;m++)
		face_num = detector.detect(Detect_Image, rects, 0);   //for imagelist input
	//	detector.ClearFaceDetectionRange();
	//	//detector.ClearFaceDetectionROI();
	}

	cvReleaseImage(&Detect_Image);

	return face_num;
}
