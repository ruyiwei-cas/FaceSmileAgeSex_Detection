#ifndef  FACE_DETECTION_UTILITY_H
#define FACE_DETECTION_UTILITY_H

//#include "MyType.h"
//#include "FaceRecognitionUtility.h"

// Detect frontaql faces
void InitFaceDetector();
int FaceDetectionApplication(IplImage* color_image, CvRectItem* rects, int MAX_face_numBER, bool bRotateTry);

#endif