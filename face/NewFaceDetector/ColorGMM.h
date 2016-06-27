#pragma once

#include <opencv2/opencv.hpp>
#include "opencv/cv.h"
#include "opencv/highgui.h"
#include <opencv2/legacy/legacy.hpp>


using namespace cv;
using namespace std;
class CColorGMM
{
public:
	int m_nClusterNum;
	int m_nLookupTable_Interval;
	int m_nLookupTalbe_Channel_Bin;
    CvEM m_Em;
    CvEMParams m_Em_params;
	unsigned char *m_lpLookupTable_Data;

public:
	int TrainLookupTalbe(Mat lpTrainingImage);
	bool SaveLookupTable(char *sFilename);
	bool LoadLookupTable(char *sFilename);
	Mat RGB_to_ColorMap(Mat ImageMat);
	void RGB_to_ColorMap(IplImage*SrcImage, IplImage *ColorMap);
	CColorGMM(void);
	~CColorGMM(void);
};

