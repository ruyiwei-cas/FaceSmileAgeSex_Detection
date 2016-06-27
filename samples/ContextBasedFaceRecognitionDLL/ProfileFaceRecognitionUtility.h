#ifndef  FACE_RECOGNITION_UTILITY_H
#define FACE_RECOGNITION_UTILITY_H
#include "opencv/cv.h"
#include "opencv/highgui.h"	
#include "basetypes.hpp"
using namespace cv;

extern "C" __declspec(dllexport)
int WriteProfilingFile(char *lpFilename, int nFace_Num, Face_Attribute *FaceRecognitionResult, int *Face_Valid_Flag);

extern "C" __declspec(dllexport)
int LoadProfilingFile(char *lpFilename, Face_Attribute* FaceRecognitionResult, int *Face_Valid_Flag);

//int ProfileFaceRecognition(char *sFilename, int *nFrontalFaceNum, Rect* FrontalFaceRegion);

void GetFrontalFace(Rect *FrontalFaceRegoin, int nFace_Num, Face_Attribute* FaceRecognitionResult);

#endif