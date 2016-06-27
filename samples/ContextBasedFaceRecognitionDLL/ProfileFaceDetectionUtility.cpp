
#include "opencv/cv.h"
#include "opencv/highgui.h"	
#include "ProfileFaceDetectionUtility.h"
CvHaarClassifierCascade* ProfileCascade1 = nullptr;
CvHaarClassifierCascade* ProfileCascade2 = nullptr;
//----------------------------------------------------------------------------------------------------------
//
//              The following is the profile face detection based on OpenCV
//
//
//
//----------------------------------------------------------------------------------------------------------
//#include "../humandet/Cascade_Classifier.h"
//Class_Cascade_Classifier model_Cascades;
////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the Face Detection Utility
///  written by Ren Haibing 
///   the face model is in the default directory
/// Acknowledge		: modified from Yimin's program
///
/// Function name	: InitProfileFaceDetector
/// Description	    : init opencv profile face detector
///
/// Argument		:	
///
/// Return type		: 
///
/// Create Time		: 2014-11-6  10:10
///
///
/// Side Effect		: 
///
///////////////////////////////////////////////////////////////////////////////////////////////
extern "C" __declspec(dllexport)
void InitProfileFaceDetector()
{
	ProfileCascade1 = (CvHaarClassifierCascade*)cvLoad("./objdet_model/ModelFiles/Haar_Cascades/haarcascade_profileface.xml");
	ProfileCascade2 = (CvHaarClassifierCascade*)cvLoad("./objdet_model/ModelFiles/Haar_Cascades/haarcascade_frontalface_alt2.xml");
}
extern "C" __declspec(dllexport)
void ReleaseProfileFaceDetector()
{
	if (ProfileCascade1 != nullptr)
		cvReleaseHaarClassifierCascade(&ProfileCascade1);
	if (ProfileCascade2 != nullptr)
		cvReleaseHaarClassifierCascade(&ProfileCascade2);
}
////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the Face Detection Utility
///  written by Ren Haibing 
/// 
/// Acknowledge		: 
///
/// Function name	: FaceRegionSuppression
/// Description	    : Remove the overlapped regions(small overlapping is acceptable)
///
/// Argument		:	
///
/// Return type		: non-overlapped face number
///
/// Create Time		: 2014-11-21  15:24
///
///
/// Side Effect		: 
///
///////////////////////////////////////////////////////////////////////////////////////////////
#define PROFILE_FACE_SIZE_THRESHOLD 80

bool FaceRegion_Overlap(Rect* r1, Rect *r2, double dError = 0.5)
{
	/// get the common area
	int nCommon_lx, nCommon_ly, nCommon_rx, nCommon_ry;
	nCommon_lx = max(r1->x, r2->x);
	nCommon_ly = max(r1->y, r2->y);
	nCommon_rx = min(r1->x + r1->width, r2->x+r2->width);
	nCommon_ry = min(r1->y + r1->height, r2->y+r2->height);
	if((nCommon_lx>nCommon_rx) || (nCommon_ly > nCommon_ry))
		return false;
	double nCommon_Area = (nCommon_rx-nCommon_lx)*(nCommon_ry - nCommon_ly);

	double f1 = nCommon_Area / (r1->width*r1->height);
	if(f1>dError) return true;
	double f2 = nCommon_Area / (r2->width*r2->height);
	if(f2>dError) return true;
	return false;
}
int FaceRegionSuppression(IplImage* gray_image, Rect *FaceRegion, int nCandidateNum)
{
	int nValidFlag[100];
	int nFaceSizeThreshold = PROFILE_FACE_SIZE_THRESHOLD;
	nFaceSizeThreshold = std::max(nFaceSizeThreshold, int(gray_image->width *0.1f));
	nFaceSizeThreshold = std::max(nFaceSizeThreshold, int(gray_image->height *0.1f));

	for(int i=0;i<nCandidateNum;i++)
	{
		if(FaceRegion[i].width>nFaceSizeThreshold)
			nValidFlag[i] = 1;
		else nValidFlag[i] = 0;
	}

	for(int i=0;i<nCandidateNum-1;i++)
	{
		if(nValidFlag[i] !=1) continue;
		for(int j=i+1;j<nCandidateNum;j++)
		{
			if(nValidFlag[j] !=1) continue;

			if(FaceRegion_Overlap(&(FaceRegion[i]), &(FaceRegion[j]))) // disable the overlapped region
			{
				if(FaceRegion[i].width>FaceRegion[j].width)
					nValidFlag[j] = 0;
				else nValidFlag[i] = 0;
			}
		}
	}
	int nSearchIndex = 0;
	int nInsertIndex = 0;
	for(int i=0;i<nCandidateNum;i++)
	{
		if(nValidFlag[i]==1)
		{
			if(nSearchIndex != nInsertIndex)
			{
				FaceRegion[nInsertIndex] = FaceRegion[nSearchIndex];
			}
			nInsertIndex++;
		}
		nSearchIndex++;
	}
	return nInsertIndex;
}

////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the Face Detection Utility
///  written by Ren Haibing 
/// 
/// Acknowledge		: 
///
/// Function name	: FaceRegionRemoveSimilar
/// Description	    : Remove the face regions which are frontal(in the frontal face region array)
///
/// Argument		:	SrcFaceRegion -- current face region array
/// Argument		:	SrcFaceRegion -- current face region array
/// Argument		:	RemoveFaceRegion -- current face region array
/// Argument		:	SrcFaceRegion -- current face region array
///
/// Return type		: non-overlapped face number
///
/// Create Time		: 2014-21-29  13:16
///
///
/// Side Effect		: 
///
///////////////////////////////////////////////////////////////////////////////////////////////
int FaceRegionRemoveSimilar(Rect *SrcFaceRegion, int nCandidateNum, Rect *RemoveFaceRegion, int nRemoveFaceNum)
{	
	int nValidFlag[100];
	for(int i=0;i<nCandidateNum;i++)
	{
		nValidFlag[i] = 1;
	}
	for(int i=0;i<nCandidateNum;i++)
		for(int j=0;j<nRemoveFaceNum;j++)
		{
			if(nValidFlag[i] ==1)
			{
				if(FaceRegion_Overlap(&(SrcFaceRegion[i]), &(RemoveFaceRegion[j]) ))
					nValidFlag[i] =0;
			}
		}
	int nRemainFaceNum = 0;
	for(int i=0;i<nCandidateNum;i++)
	{
		/*
		if(nValidFlag[i] ==1)
		{
			if(SrcFaceRegion[i].width<PROFILE_FACE_SIZE_THRESHOLD *2)  // remove small faces
				nValidFlag[i] = 0;
		}
		*/
		if(nValidFlag[i] ==1)
		{
			if(nRemainFaceNum != i)
			{
				SrcFaceRegion[nRemainFaceNum] = SrcFaceRegion[i];
			}
			nRemainFaceNum++;
		}

	}
	return nRemainFaceNum;
}
////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the Face Detection Utility
///  written by Ren Haibing 
///   the face model is in the default directory
/// Acknowledge		: modified from Yimin's program
///
/// Function name	: DetectProfileFace
/// Description	    : detect the profile file with opencv profile face detector
///
/// Argument		:	
///
/// Return type		: 
///
/// Create Time		: 2014-11-6  10:44
///
///
/// Side Effect		: 
///
///////////////////////////////////////////////////////////////////////////////////////////////
int Function_DetectProfile(IplImage* gray_image, Rect *FaceRegion)
{
	CvMemStorage* storage1 = 0;
	storage1 = cvCreateMemStorage(0);
	cvClearMemStorage(storage1);
		
	int nFaceNum = 0;
	CvSeq* DetectionResult1 = cvHaarDetectObjects(gray_image, ProfileCascade1, storage1, 1.1, 3, CV_HAAR_DO_CANNY_PRUNING, cvSize(50, 50));
	for (int i = 0; i <DetectionResult1->total; i++)
	{
		CvRect* r = (CvRect*)cvGetSeqElem(DetectionResult1, i);
		FaceRegion[nFaceNum].x = r->x;
		FaceRegion[nFaceNum].y = r->y;
		FaceRegion[nFaceNum].width = r->width;
		FaceRegion[nFaceNum].height = r->height;
		if (nFaceNum<MAX_FACE_NUMBER-1)
			nFaceNum++;
	}
	CvMemStorage* storage2 = 0;
	storage2 = cvCreateMemStorage(0);
	cvClearMemStorage(storage2);
	CvSeq* DetectionResult2 = cvHaarDetectObjects(gray_image, ProfileCascade2, storage2, 1.1, 3, CV_HAAR_DO_CANNY_PRUNING, cvSize(50, 50));
	for (int i = 0; i <DetectionResult2->total; i++)
	{
		CvRect* r = (CvRect*)cvGetSeqElem(DetectionResult2, i);
		FaceRegion[nFaceNum].x = r->x;
		FaceRegion[nFaceNum].y = r->y;
		FaceRegion[nFaceNum].width = r->width;
		FaceRegion[nFaceNum].height = r->height;
		if (nFaceNum<MAX_FACE_NUMBER - 1)
			nFaceNum++;
	}
	cvReleaseMemStorage(&storage1);
	cvReleaseMemStorage(&storage2);
	return nFaceNum;
}

extern "C" __declspec(dllexport)
int DetectProfileFace(IplImage* gray_image, Rect *FaceRegion, Rect *FrontalFaceRegion, int nFrontalFaceNum)
{
//	Mat GrayImgMat(gray_image,1);
	
//	int nCandidateNum = model_Cascades.myDetect(GrayImgMat, FaceRegion, MAX_FACE_NUMBER);	

	int nCandidateNum = Function_DetectProfile(gray_image, FaceRegion);

	int nFace_Num = FaceRegionSuppression(gray_image,FaceRegion, nCandidateNum);

	nFace_Num = FaceRegionRemoveSimilar(FaceRegion, nFace_Num, FrontalFaceRegion, nFrontalFaceNum);
	return nFace_Num;
}

