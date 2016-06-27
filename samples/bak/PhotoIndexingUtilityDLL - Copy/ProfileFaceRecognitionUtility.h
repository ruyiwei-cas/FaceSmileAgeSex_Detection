#ifndef  FACE_RECOGNITION_UTILITY_H
#define FACE_RECOGNITION_UTILITY_H
#include "opencv/cv.h"
#include "opencv/highgui.h"	
#include "basetypes.hpp"
using namespace cv;

int LoadProfilingFile(char *lpFilename);

int ProfileFaceRecognition(char *sFilename, int *nFrontalFaceNum, Rect* FrontalFaceRegion);

void GetFrontalFace(Rect *FrontalFaceRegoin, int nFace_Num);

#endif