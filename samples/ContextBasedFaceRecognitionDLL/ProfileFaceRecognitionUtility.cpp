
#include "opencv/cv.h"
#include "opencv/highgui.h"	

#include "../Platform.h"
#include "ProfileFaceDetectionUtility.h"
#include "HumanClothesHistUtility.h"

#ifdef __win__
#include <afx.h>
#endif
#include "MyType.h"

//int Face_Valid_Flag[MAX_FACE_NUMBER];	
//Face_Attribute FaceRecognitionResult[MAX_FACE_NUMBER];
Face_Attribute ProfileFaceRecognitionResult[MAX_FACE_NUMBER];
////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the Face Recognition Utility
///  written by Ren Haibing 
/// Acknowledge		: None
///
/// Function name	: WriteProfilingFile
/// Description	    : save face identification and face attribute recognition resutl to a file
///
/// Argument		:	lpFilename -- file name
/// Argument		:	nFace_Num -- detected face number
/// Argument		:	rects -- detected face region
///
/// Return type		: 
///
/// Create Time		: 2014-11-5  12:51
///
///
/// Side Effect		: 
///                   
///////////////////////////////////////////////////////////////////////////////////////////////
extern "C" __declspec(dllexport)
int WriteProfilingFile(char *lpFilename, int nFace_Num,Face_Attribute *FaceRecognitionResult, int *Face_Valid_Flag)
{
	FILE *fProfileFile = fopen(lpFilename, "wb+");
	if (fProfileFile == nullptr)
		return -1;
	fwrite(&nFace_Num, sizeof(int), 1, fProfileFile);
	//fprintf(fProfileFile, "Face Number:%d\n", nFace_Num);
	CvRect * rect;
	for (int i = 0; i<nFace_Num; i++)
	{
		if ((Face_Valid_Flag != nullptr) && (Face_Valid_Flag[i] <= 0))
			continue;

		//fprintf(fProfileFile, "Face No.:%d\n", i);
		fwrite(&i, sizeof(int), 1, fProfileFile);
		/*
		rect = &(FaceRecognitionResult[i].FaceRegion);
		fprintf(fProfileFile, "Face view: %d\n", FaceRecognitionResult[i].FaceView);
		fprintf(fProfileFile, "Face rect: %d, %d, %d, %d\n", rect->x,rect->y,rect->width, rect->height);
		fprintf(fProfileFile, "Face ID.:(%d, %f)\n", FaceRecognitionResult[i].FaceID, FaceRecognitionResult[i].Prob_FaceID);
		fprintf(fProfileFile, "Age :(%d, %f)\n", FaceRecognitionResult[i].Age, FaceRecognitionResult[i].Prob_Age);
		fprintf(fProfileFile, "Blink :(%d, %f)\n", FaceRecognitionResult[i].Blink, FaceRecognitionResult[i].Prob_Blink);
		fprintf(fProfileFile, "Gender:(%d, %f)\n", FaceRecognitionResult[i].Gender, FaceRecognitionResult[i].Prob_Gender);
		fprintf(fProfileFile, "Smile:(%d, %f)\n", FaceRecognitionResult[i].Smile, FaceRecognitionResult[i].Prob_Smile);
		*/
		fwrite(&(FaceRecognitionResult[i]), sizeof(Face_Attribute), 1, fProfileFile);

		/*
		//fprintf(fProfileFile,"Clothes Histogram: ");
		for (int j = 0; j<ATTRIBUTE_FEATURE_DIM; j++)
		fprintf(fProfileFile, "%f ", FaceRecognitionResult[i].Attribute_Feature[j]);
		fprintf(fProfileFile, "\n");
		*/
		fwrite(FaceRecognitionResult[i].Attribute_Feature, sizeof(double), ATTRIBUTE_FEATURE_DIM, fProfileFile);
	}
	fclose(fProfileFile);
}

////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the Face Recognition Utility
///  written by Ren Haibing 
/// Acknowledge		: None
///
/// Function name	: LoadProfilingFile
/// Description	    : load face identification and face attribute recognition resutl from a file
///
/// Argument		:	lpFilename -- file name
///
/// Return type		:   int -- face number
///
/// Create Time		: 2014-11-5  13:18
///
///
/// Side Effect		: 
///                   
///////////////////////////////////////////////////////////////////////////////////////////////
extern "C" __declspec(dllexport)
int LoadProfilingFile(char *lpFilename, Face_Attribute* FaceRecognitionResult, int *Face_Valid_Flag)
{
	int nFace_Num = 0;
	FILE *fProfileFile = fopen(lpFilename, "rb");
	if (fProfileFile == NULL) return -1;
	int nCount = fread(&nFace_Num, sizeof(int), 1, fProfileFile);
	if (nCount != 1)
		return -1;

	for (int i = 0; i<MAX_FACE_NUMBER; i++)
		Face_Valid_Flag[i] = 0;

	int nFaceNo = -1;
	for (int i = 0; i < nFace_Num; i++)
	{
		nCount = fread(&nFaceNo, sizeof(int), 1, fProfileFile);
		if (nCount != 1)
			break;
		if (nFaceNo < 0) break;
		Face_Valid_Flag[nFaceNo] = 1;

		nCount = fread(&(FaceRecognitionResult[nFaceNo]), sizeof(Face_Attribute), 1, fProfileFile);
		nCount = fread(FaceRecognitionResult[nFaceNo].Attribute_Feature, sizeof(double), ATTRIBUTE_FEATURE_DIM, fProfileFile);
	}
	return nFace_Num;
}
/*{
	int nFace_Num=0;
	FILE *fProfileFile = fopen(lpFilename, "r");
	if(fProfileFile == NULL) return -1;

	fscanf(fProfileFile, "Face Number:%d\n", & nFace_Num);
	if(nFace_Num==0)
		return 0;

	CvRect *rect;
	for(int i=0;i<MAX_FACE_NUMBER;i++)
		Face_Valid_Flag[i]=0;

	int nFaceNo = -1;
	int nTemp;
	int nReadNum;
	for(int i=0;i<nFace_Num;i++)
	{
		nReadNum = fscanf(fProfileFile, "Face No.:%d\n",&nTemp);
		if(nReadNum!=1) break;
		nFaceNo++;
		rect = &(FaceRecognitionResult[nFaceNo].FaceRegion);

		nReadNum = fscanf(fProfileFile, "Face view: %d\n", &(FaceRecognitionResult[i].FaceView));
		nReadNum = fscanf(fProfileFile, "Face rect: %d, %d, %d, %d\n", &(rect->x),&(rect->y),&(rect->width), &(rect->height));

		nReadNum = fscanf(fProfileFile, "Face ID.:(%d, %f)\n", &(FaceRecognitionResult[nFaceNo].FaceID), &(FaceRecognitionResult[nFaceNo].Prob_FaceID));
		nReadNum = fscanf(fProfileFile, "Age :(%d, %f)\n", &(FaceRecognitionResult[nFaceNo].Age), &(FaceRecognitionResult[nFaceNo].Prob_Age));
		nReadNum = fscanf(fProfileFile, "Blink :(%d, %f)\n", &(FaceRecognitionResult[nFaceNo].Blink), &(FaceRecognitionResult[nFaceNo].Prob_Blink));
		nReadNum = fscanf(fProfileFile, "Gender:(%d, %f)\n", &(FaceRecognitionResult[nFaceNo].Gender),&(FaceRecognitionResult[nFaceNo].Prob_Gender));
		nReadNum = fscanf(fProfileFile, "Smile:(%d, %f)\n", &(FaceRecognitionResult[nFaceNo].Smile), &(FaceRecognitionResult[nFaceNo].Prob_Smile));

		fscanf(fProfileFile,"Clothes Histogram: ");
		for (int j = 0; j<ATTRIBUTE_FEATURE_DIM; j++)
			fscanf(fProfileFile, "%lf ", &(FaceRecognitionResult[i].Attribute_Feature[j]));

		fscanf(fProfileFile, "\n");
	}
	fclose(fProfileFile);
	return nFaceNo+1;
}
*/
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
void GetFrontalFace(Rect *FrontalFaceRegoin, int nFace_Num, Face_Attribute* FaceRecognitionResult)
{
	for(int i=0;i<nFace_Num;i++)
	{
		FrontalFaceRegoin[i].x = FaceRecognitionResult[i].FaceRegion.x;
		FrontalFaceRegoin[i].y = FaceRecognitionResult[i].FaceRegion.y;
		FrontalFaceRegoin[i].width = FaceRecognitionResult[i].FaceRegion.width;
		FrontalFaceRegoin[i].height = FaceRecognitionResult[i].FaceRegion.height;
	}
}
/*
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
	IplImage *color_image = nullptr;// ReadImage_Withexif(sFilename);
	if (color_image == NULL) return -2; // file not exist

	cout << sFilename << endl;
	char sDate[12];
	int nValidDate = 0;// GetFileDate(sFilename, sDate);
	if(nValidDate != 1)
	{
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

	int Face_Valid_Flag[MAX_FACE_NUMBER];	
	Face_Attribute FaceRecognitionResult[MAX_FACE_NUMBER];
	int nFrontalFace_Num = LoadProfilingFile(sRecognitionResultFilename, FaceRecognitionResult, Face_Valid_Flag);

	GetFrontalFace(FrontalFaceRegoin, nFrontalFace_Num, FaceRecognitionResult);
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
	if(face_num>0)
	{
		sprintf(sRecognitionResultFilename, "%sFDFR15.txt", tmpStr1);  
		WriteProfilingFile(sRecognitionResultFilename, face_num, FaceRecognitionResult, Face_Valid_Flag);
	}

	cvReleaseImage(&gray_image);
	cvReleaseImage(&color_image);
	return face_num;
}
*/