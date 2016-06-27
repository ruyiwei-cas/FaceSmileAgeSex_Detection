#include "opencv/cv.h"
#include "opencv/highgui.h"	
#include "cxlibface.hpp"
#include "libexif/exif-data.h"  //for libexif library

#include "MyType_Main.h"
#include "FaceDetectionUtility.h"
#include "FaceRecognitionUtility.h"
#include "FaceRegistrationUtility.h"
#include "HumanClothesHistUtility.h"

#include <time.h>
#include <string>

#include "../Platform.h"

#include "SceneClassificationDLL.h"
#include "ContextBasedFaceRecognitionDLL.h"

#ifdef __win__
#include <opencv2\opencv.hpp>
#endif
#ifdef __linux__
#include <opencv2/opencv.hpp>
#endif

#include <time.h>



#define ITER_NUM 1


extern Face_Attribute FaceRecognitionResult[MAX_FACE_NUMBER];
extern int Face_Valid_Flag[MAX_FACE_NUMBER];
////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the Photo indexing Utility
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
void PhotoIndexing_Init()
{
	cout<<"Begin photo indexing initialization "<<endl;
	// Face Recognition -------------------------------------------------------
	InitFaceRecognition();

	// Face Detector ---------------------------------------------
	InitFaceDetector();
		
	//scene classification --------------------------------------------------------
	initialize_scene("./scene_model");

	//Registration
	FaceRegistration_Init();

	cout<<"Begin Profile face detection initialization  "<<endl;
//	InitProfileFaceDetector();	

	// human clothes identification
	cout<<"Begin clothes histogram initialization  "<<endl;
//	Init_ClothesFeature();
	cout<<"Finish photo indexing initialization  "<<endl;
}
////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the Photo indexing Utility
///  written by Ren Haibing 
///   
/// Acknowledge		:
///
/// Function name	: PhotoIndexing_Release
/// Description	    : release memory for photo indexing 
///
/// Argument		:	
///
/// Return type		: 
///
/// Create Time		: 2014-11-18  10:32
///
///
/// Side Effect		: 
///
///////////////////////////////////////////////////////////////////////////////////////////////
void PhotoIndexing_Release()
{
	destroy_scene();
	FaceRegistration_Release();
//	ReleaseProfileFaceDetector();
	cout<<"Release photo indexing memory"<<endl;
}
////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the Photo indexing Utility
///  written by Zhang, Yimin
///   
/// Acknowledge		:
///
/// Function name	: imgRotation90n
/// Description	    : rotate the image, given the rotation angle 
///
/// Argument		:	srcImage -- input image
/// Argument		:	angle -- rotation angle(1: 90,  2: 180 degree 3: 270 )
///
/// Return type		:  IplImage -- rotated image
///
/// Create Time		: 2014-11-18  10:32
///
///
/// Side Effect		: 
///
///////////////////////////////////////////////////////////////////////////////////////////////
//rotate (counter clockwise) an image by angle (1: 90,  2: 180 degree 3: 270 )
IplImage* imgRotation90n(IplImage* srcImage, int angle)
{
	assert(angle==1 || angle ==2 || angle ==3); 
	
	IplImage* dstImage = NULL;

	if (srcImage == NULL)
		return NULL;

	//set the center of rotation   
    CvPoint2D32f center;     
    center.x=float (srcImage->width/2);   
    center.y=float (srcImage->height/2);  

    //set the rotation matrix  
    float m[6];               
    CvMat M = cvMat( 2, 3, CV_32F, m );   
    cv2DRotationMatrix( center, angle*90,1, &M);   

	if (angle==2)
	{
		dstImage = cvCreateImage (cvSize(srcImage->width,srcImage->height), srcImage->depth,srcImage->nChannels);
	    //rotate the image   
        cvWarpAffine(srcImage,dstImage, &M,CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS,cvScalarAll(0) ); 
	}
	else
	{
		int maxHW = max(srcImage->width,srcImage->height); //the max of Height and Width

	    // Adjust rotation center to dst's center,
	    m[2] += (maxHW - srcImage->width) / 2;
	    m[5] += (maxHW - srcImage->height) / 2;
		dstImage = cvCreateImage (cvSize(srcImage->height,srcImage->width), srcImage->depth,srcImage->nChannels);
		IplImage* tmpImage = cvCreateImage (cvSize(maxHW, maxHW), srcImage->depth,srcImage->nChannels);
	    //rotate the image   
        cvWarpAffine(srcImage,tmpImage, &M,CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS,cvScalarAll(0) ); 

		if ( (srcImage->width) >= (srcImage->height))
		  cvSetImageROI(tmpImage,cvRect((maxHW-srcImage->height)/2,0, srcImage->height,srcImage->width));
		else
		  cvSetImageROI(tmpImage,cvRect(0,(maxHW-srcImage->width)/2,srcImage->height,srcImage->width));

		cvCopyImage(tmpImage, dstImage);  //just copy the ROI area
		cvReleaseImage(&tmpImage);
	}
	return dstImage;
}
////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the Photo indexing Utility
///  written by Zhang, Yimin; modified by Ren, Haibing
///   
/// Acknowledge		:
///
/// Function name	: ReadImage_Withexif
/// Description	    : read image and rotate according to exif infor 
///
/// Argument		:	sFilename -- input image jname
///
/// Return type		:  IplImage -- rotated image
///
/// Create Time		: 2014-11-18  10:32
///
///
/// Side Effect		: 
///
///////////////////////////////////////////////////////////////////////////////////////////////
IplImage *ReadImage_Withexif(char *sFilename)
{
	int imgNo=0;
	int imgOrientation,imgRotAngle;
	IplImage* color_image,*tmp_image;


	cout<<"Reading exif information "<<endl;
	//reading image EXIF, orientation and GPS�� using LIBEXIF library
	imgOrientation = 1; //init to normal orientation
	ExifData *imgExif = NULL;
	imgExif = exif_data_new_from_file(sFilename);
	if (imgExif) 
	{
        ExifByteOrder byteOrder = exif_data_get_byte_order(imgExif);
        ExifEntry *imgExifEntry = exif_data_get_entry(imgExif, EXIF_TAG_ORIENTATION);
        if (imgExifEntry)
            imgOrientation = exif_get_short(imgExifEntry->data, byteOrder);

        exif_data_free(imgExif);
    }

		
	cout<<"Loading image "<<endl;
	if (imgOrientation == 1)
		color_image = cvLoadImage(sFilename);
	else
	{
		tmp_image = cvLoadImage(sFilename);
					
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
		
	if( color_image == NULL ) return NULL; //// the image doesn't exist.
	return color_image;
}
////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the Photo indexing Utility
///  written by Ren, Haibing
///   
/// Acknowledge		:
///
/// Function name	: Scene_Classification
/// Description	    : given image name, get the scene category 
///
/// Argument		:	color_image -- input image file name
/// Argument		:	rects -- detected face region
/// Argument		:	Face_Valid_Flag -- valid face flag from face recognition
/// Argument		:	face_num -- detected face number
///
/// Return type		:  int -- scene category
///
/// Create Time		: 2014-12-1  16:22
///
///
/// Side Effect		: 
///
///////////////////////////////////////////////////////////////////////////////////////////////
#define SCENE_CONFIDENCE_THRESHOLD  -0.15
int Scene_Classification(IplImage* color_image,CvRectItem* rects, int *Face_Valid_Flag, int face_num)
{
	int nValid_Face_Num = 0;
	int nMax_Face_Size = -1;
	for(int i=0;i<face_num;i++)
	{
		if(Face_Valid_Flag[i] >0)
		{
			nValid_Face_Num++;
			if(rects[i].rc.width>nMax_Face_Size)
				nMax_Face_Size = rects[i].rc.width;
		}
	}
	////if(nValid_Face_Num>=1) return -2;// scene category: human
	////if(nMax_Face_Size>= color_image->width * 0.2) return -2;// scene category: human

	double dSceneResult =  test_single_img(color_image);
	int nSceneCategory = int(dSceneResult);
	double dConfidence = (dSceneResult- nSceneCategory) * 10;
	if(dConfidence<SCENE_CONFIDENCE_THRESHOLD)
		nSceneCategory = -1;
	return nSceneCategory;
}
////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the Photo indexing Utility
///  written by Zhang, Yimin; modified by Ren, Haibing
///   
/// Acknowledge		:
///
/// Function name	: PhotoIndexing_ImageEvaluation
/// Description	    : given image name, generate the label information for photo indexing 
///
/// Argument		:	sFilename -- input image file name
/// Argument		:	*scene_label -- scene classification result
///
/// Return type		:  int -- detected face number
///
/// Create Time		: 2014-11-18  13:10
///
///
/// Side Effect		: 
///
///////////////////////////////////////////////////////////////////////////////////////////////
int PhotoIndexing_ImageEvaluation(char* sFilename,int *scene_label)
{
	IplImage* color_image = ReadImage_Withexif(sFilename);
	if(color_image == NULL) return -1;

	CvRectItem rects[MAX_FACE_NUMBER];
	TickMeter	time_Cascades, photo_time_Cascades;
	
	// face detection
	time_Cascades.reset(); time_Cascades.start();
	int face_num=0;
	face_num = FaceDetectionApplication(color_image, rects, MAX_FACE_NUMBER, false);
	time_Cascades.stop();
	cout<<"Face number: "<<face_num<<endl;
	cout<<"Face detection time: "<<time_Cascades.getTimeSec()<<endl;



	//3.3 face attribute recognition and identification
	time_Cascades.reset(); time_Cascades.start();
	FaceRecognitionApplication(color_image, face_num, rects);
	time_Cascades.stop();
	cout<<"Face recognition time: "<<time_Cascades.getTimeSec()<<endl;


	//3.2 scene classification
	time_Cascades.reset(); time_Cascades.start();
	*scene_label = Scene_Classification(color_image, rects, Face_Valid_Flag, face_num);
	time_Cascades.stop();
	cout<<"Scene classification time: "<<time_Cascades.getTimeSec()<<endl;

	//3.4 Get Body part histogram
	char sRecognitionResultFilename[1024]; 
	char sImgPath1[1024]; 
	char tmpStr1[1024]; 
	strcpy(tmpStr1, sFilename); 
	char* firstdot1 = strchr(tmpStr1,'.');
	*firstdot1 = NULL;
	strcpy(sImgPath1, tmpStr1);       
	for(int i=0;i<face_num;i++)
	{
		if(Face_Valid_Flag[i] !=1) continue;
		GetClothesFeature(color_image, &(FaceRecognitionResult[i]));
	}

	time_Cascades.stop();
	cout<<"Face attribute recognition and identition time: "<<time_Cascades.getTimeSec()<<endl;

	//4. output face detection and recongition image
	sprintf(sRecognitionResultFilename, "%sFDFR14.dat", tmpStr1);  
	WriteProfilingFile(sRecognitionResultFilename, face_num, FaceRecognitionResult, Face_Valid_Flag);

	cvReleaseImage(&color_image);
	return face_num;
}
