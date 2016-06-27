#include "opencv/cv.h"
#include "opencv/highgui.h"	
#include "cxlibface.hpp"
#include "libexif/exif-data.h"  //for libexif library
#include "MyType.h"

CxlibFaceAnalyzer *m_faceAnalyzer = NULL;
IplImage* imgRotation90n(IplImage* srcImage, int angle);
////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the Face Registration Utility
///  written by Ren Haibing 
///   the face model is in the default directory
/// Acknowledge		:
///
/// Function name	: FaceRegistration_Init
/// Description	    : init CxlibFaceAnalyzer for face registration  
///
/// Argument		:	
///
/// Return type		: 
///
/// Create Time		: 2014-11-18  10:20
///
///
/// Side Effect		: 
///
///////////////////////////////////////////////////////////////////////////////////////////////
void FaceRegistration_Init()
{
	if(m_faceAnalyzer == NULL)
	{
		EnumTrackerType traType   = TRA_HAAR; //TRA_PF;
		EnumViewAngle   viewAngle = VIEW_ANGLE_FRONTAL; //VIEW_ANGLE_HALF_MULTI; //

		int  sampleRate = 1;
		char str_facesetxml[] = "faceset_model.xml";
		int recognizerType = RECOGNIZER_CAS_GLOH;  //RECOGNIZER_BOOST_GB240
		bool bEnableAutoCluster =  false;//false;//true;
		bool bEnableShapeRegressor =  true;//false;//true;

		m_faceAnalyzer = new CxlibFaceAnalyzer(
            viewAngle, traType, 0,
            str_facesetxml, recognizerType, bEnableAutoCluster, bEnableShapeRegressor );
	}

}
////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the Face Registration Utility
///  written by Ren Haibing 
///  
/// Acknowledge		:
///
/// Function name	: FaceRegistration_DetectFace
/// Description	    : Given a image file name, detect the face region and insert it to template array  
///
/// Argument		:	sImageFilename -- input image file name
/// Argument		:	sDesFaceName -- save the detected face region
/// Argument		:	m_FaceTemplate -- face template array
/// Argument		:	nTotalFaceNum -- valid face template size
///
/// Return type		: 
///
/// Create Time		: 2014-11-18  10:41
///
///
/// Side Effect		: 
///
///////////////////////////////////////////////////////////////////////////////////////////////
#define LARGE_IMAGE_SIZE  2000
#define STANDARD_IMAGE_WIDTH_LONG  2000
#define STANDARD_IMAGE_WIDTH_SMALL  1200
int FaceRegistration_DetectFace(char *sImageFilename, char *sDesFaceName, IplImage *m_FaceTemplate[FACE_TEMPLATE_MAX_NUM], int *nTotalFaceNum)
{
	int imgOrientation,imgRotAngle;

	IplImage* color_image,*tmp_image;
	IplImage* gray_image = NULL;

	//reading image EXIF, orientation and GPS�� using LIBEXIF library
	imgOrientation = 1; //init to normal orientation
	ExifData *imgExif = exif_data_new_from_file(sImageFilename);
	if (imgExif) 
	{
        ExifByteOrder byteOrder = exif_data_get_byte_order(imgExif);
        ExifEntry *imgExifEntry = exif_data_get_entry(imgExif, EXIF_TAG_ORIENTATION);
        if (imgExifEntry)
            imgOrientation = exif_get_short(imgExifEntry->data, byteOrder);

        exif_data_free(imgExif);
    }
		
		
	if (imgOrientation == 1)
		color_image = cvLoadImage(sImageFilename);
	else
	{
		tmp_image = cvLoadImage(sImageFilename);
					
		if (imgOrientation == 8)  //image needs to rotate 90 degree counter clock wise
			imgRotAngle = 1;
		else if (imgOrientation == 3)  //image needs to rotate 180 degree counter clock wise
			imgRotAngle = 2;
		else                          //image needs to rotate 270 degree counter clock wise
			imgRotAngle = 3;

		if (tmp_image != NULL)
		{
			color_image = imgRotation90n(tmp_image, imgRotAngle);
			cvReleaseImage(&tmp_image);
		}
	}	
		
	if( color_image == NULL ) return 0; //// the image doesn't exist.

	int nNewWidth, nNewHeight;
	double dScale;
	IplImage *Detect_Image = nullptr;
	if ((color_image->width>LARGE_IMAGE_SIZE) || (color_image->height>LARGE_IMAGE_SIZE))
	{
		if (color_image->width>color_image->height)
			nNewWidth = STANDARD_IMAGE_WIDTH_LONG;
		else nNewWidth = STANDARD_IMAGE_WIDTH_SMALL;

		dScale = nNewWidth * 1.0 / color_image->width;

		nNewHeight = int(color_image->height * dScale);
		Detect_Image = cvCreateImage(cvSize(nNewWidth, nNewHeight), IPL_DEPTH_8U, color_image->nChannels);
		cvResize(color_image, Detect_Image);
		cvReleaseImage(&color_image);
		color_image = Detect_Image;
	}

	char sFilename[1024];
	sprintf(sFilename, "%s_%d.jpg",sDesFaceName, *nTotalFaceNum);
	bool bGetGoodFace = m_faceAnalyzer->Face_Detection(color_image, 80, sFilename);
	
	int nFaceNum = 0;
	if(bGetGoodFace)
	{
		nFaceNum = 1;
		IplImage *lpFace = m_faceAnalyzer->getBigCutFace();
					
		m_FaceTemplate[*nTotalFaceNum] = cvCreateImage(cvSize(lpFace->width, lpFace->height), IPL_DEPTH_8U,1);
		cvCopyImage(lpFace, m_FaceTemplate[*nTotalFaceNum]);	

		(*nTotalFaceNum)++;
		if(*nTotalFaceNum>FACE_TEMPLATE_MAX_NUM-1)
		{
			*nTotalFaceNum=FACE_TEMPLATE_MAX_NUM-1;	
			nFaceNum = 0;
		}

	}

	cvReleaseImage(&color_image);
	return nFaceNum;
}

////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the Face Registration Utility
///  written by Ren Haibing 
///  
/// Acknowledge		:
///
/// Function name	: FaceRegistration_AddUser
/// Description	    : register the user with the template array  
///
/// Argument		:	sUserName -- user name
/// Argument		:	m_FaceTemplate -- face template array
/// Argument		:	nTotalFaceNum -- valid face template size
///
/// Return type		: 
///
/// Create Time		: 2014-11-18  10:45
///
///
/// Side Effect		: 
///
///////////////////////////////////////////////////////////////////////////////////////////////
void FaceRegistration_AddUser(char *sUserName, IplImage *m_FaceTemplate[FACE_TEMPLATE_MAX_NUM], int nTotalFaceNum)
{
	// add the templates to the DB
	int nFaceSetIdx    = -1;
	int nFaceSetID     = -1;
	nFaceSetIdx = m_faceAnalyzer->insertEmptyFaceSet(sUserName);
				
	for(int i=0;i<nTotalFaceNum;i++)
	{
		m_faceAnalyzer->tryInsertFace(m_FaceTemplate[i], nFaceSetIdx,  true);
		nFaceSetID = m_faceAnalyzer->getFaceSetID(nFaceSetIdx);
	}
	m_faceAnalyzer->saveFaceModelXML("faceset_model.xml");

	for(int i=0;i<FACE_TEMPLATE_MAX_NUM;i++)
	{
		if(m_FaceTemplate[i] != NULL) cvReleaseImage(&(m_FaceTemplate[i]));
	}	
}
////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the Face Registration Utility
///  written by Ren Haibing 
///  
/// Acknowledge		:
///
/// Function name	: FaceRegistration_Release
/// Description	    : release the memory for face registration 
///
/// Argument		:
///
/// Return type		: 
///
/// Create Time		: 2014-11-18  10:50
///
///
/// Side Effect		: 
///
///////////////////////////////////////////////////////////////////////////////////////////////
void FaceRegistration_Release()
{
	if(m_faceAnalyzer)
	{
		delete m_faceAnalyzer;
		m_faceAnalyzer = NULL;
	}
}