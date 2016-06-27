#pragma once
#include "basic_functions.h"


class Class_HoG_LinearSVM: public HOGDescriptor
{
public:
	struct ObjectDetection{
		Rect rect;
		float score;
	};

	Class_HoG_LinearSVM(Size win_size=Size(64,128));
	~Class_HoG_LinearSVM();

	void myDetect(const Mat &im, vector<ObjectDetection> &dets);
	void myShowDets(Mat &im, const vector<ObjectDetection> &dets);
};