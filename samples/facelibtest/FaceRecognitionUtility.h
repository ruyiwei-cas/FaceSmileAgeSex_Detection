#ifndef  FACE_RECOGNITION_UTILITY_H
#define FACE_RECOGNITION_UTILITY_H



#include "opencv/cv.h"
#include "opencv/highgui.h"	
#include "basetypes.hpp"

void InitFaceRecognition();
const char *GetRecognizedFaceName(int nFaceSetID);
void FaceRecognitionApplication(IplImage* gray_image, int face_num,CvRectItem* rects);


int ProfileFaceRecognition(char *sFilename, int *nFrontalFaceNum, Rect* FrontalFaceRegion);

void GetFrontalFace(Rect *FrontalFaceRegoin, int nFace_Num);

#endif