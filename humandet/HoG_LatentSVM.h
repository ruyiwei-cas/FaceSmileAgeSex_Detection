#pragma once
#include "basic_functions.h"


class Class_HoG_LatentSVM: public LatentSvmDetector
{
public:
	Class_HoG_LatentSVM(const vector<string> &modelfiles);
	~Class_HoG_LatentSVM();

	void myDetect(const Mat &im, vector<ObjectDetection> &dets);
	void myShowDets(Mat &im, const vector<ObjectDetection> &dets);
};
