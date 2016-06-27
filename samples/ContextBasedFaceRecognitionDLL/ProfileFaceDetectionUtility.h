#ifndef  FACE_DETECTION_UTILITY_H
#define FACE_DETECTION_UTILITY_H
#include "opencv/cv.h"
#include "opencv/highgui.h"	
using namespace cv;

#include "MyType_Main.h"
// Detect profile faces
extern "C" __declspec(dllexport)
void InitProfileFaceDetector();

int DetectProfileFace(IplImage* gray_image, Face_Attribute* HumanRegion, int* Face_Valid_Flag, int nStartFaceNo);

extern "C" __declspec(dllexport)
int DetectProfileFace(IplImage* gray_image, Rect *FaceRegion, Rect *FrontalFaceRegion, int nFrontalFaceNum);
int FaceRegionRemoveSimilar(Rect *SrcFaceRegion, int nCandidateNum, Rect *RemoveFaceRegion, int nRemoveFaceNum);
#endif