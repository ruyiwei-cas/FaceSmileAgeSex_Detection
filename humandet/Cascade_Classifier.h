#pragma once
#include "basic_functions.h"


class Class_Cascade_Classifier
{
public:
	struct ObjectDetection{
		Rect rect;
		//float score;
		int classID;
	};
	
	Class_Cascade_Classifier();
	Class_Cascade_Classifier(const vector<string> &modelfiles);
	~Class_Cascade_Classifier();
	void Init(const vector<string> &modefiles);
	void myDetect(const Mat &im, vector<ObjectDetection> &dets);
	void myShowDets(Mat &im, const vector<ObjectDetection> &dets);
	int  myDetect(const Mat &im, Rect *FaceRegion, int nFaceRegionSize);
private:
	vector<CascadeClassifier *> models;
};