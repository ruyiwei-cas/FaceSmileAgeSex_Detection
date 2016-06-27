#ifndef  MY_TYPE_MAIN_H
#define MY_TYPE_MAIN_H

#define FACE_ID_THRESHOLD 0.1
#define MAX_FACE_ID 20
#define FACE_TEMPLATE_MAX_NUM  MAX_FACE_ID
#define MAX_FACE_NUMBER   16


#define size_smallface  64
#define size_bigface    128

#define HUMAN_CLOTHES_HIST_BIN_NUMBER  8
#define HUMAN_CLOTHES_HIST_DIM HUMAN_CLOTHES_HIST_BIN_NUMBER*HUMAN_CLOTHES_HIST_BIN_NUMBER*HUMAN_CLOTHES_HIST_BIN_NUMBER

#include "opencv/cv.h"
#include "opencv/highgui.h"
#include <opencv2/opencv.hpp> 
#include <opencv\cxcore.h>

using namespace std;
using namespace cv;

typedef struct Human_Attribute
{
	CvRect FaceRegion;
	int FaceView;

	int FaceID;
	float Prob_FaceID;

	int	   Blink;
	float  Prob_Blink;
	int	   Age;
	float  Prob_Age;
	int	   Smile;
	float  Prob_Smile;
	int	   Gender;
	float  Prob_Gender;

	double ClothesHis[HUMAN_CLOTHES_HIST_DIM];
} Human_Attribute;

#endif