#include "HoG_LatentSVM.h"


// Initialization with several model files
Class_HoG_LatentSVM::Class_HoG_LatentSVM(const vector<string> &modefiles) 
	: LatentSvmDetector(modefiles)
{
}


Class_HoG_LatentSVM::~Class_HoG_LatentSVM(){
}


// Note: 'im' must be BGR-image
void Class_HoG_LatentSVM::myDetect(const Mat &im, vector<ObjectDetection> &dets)
{
	float nmsThresh = 0.4f;	
	int numThreads = -1;
	this->detect(im, dets, nmsThresh, numThreads);	// see below for detailed description of 'detect' function
	
	float scoreThresh = 0.5;
	vector<ObjectDetection>::iterator iter, iter2;
	for(iter=iter2=dets.begin(); iter!=dets.end(); ++iter)
		if(iter->score > scoreThresh)
			(*iter2++) = (*iter);	// only the detections whose scores are greater than 'scoreThresh' can be reserved.
	dets.erase(iter2, dets.end());
}
/*
void detect( const Mat& image, vector<ObjectDetection>& objectDetections,
			float overlapThreshold=0.5f, int numThreads=-1 );
Parameters:	
	image 每 An image.
	objectDetections 每 The detections: rectangle, scores and class IDs.
	overlapThreshold 每 Threshold for the non-maximum suppression algorithm.
	numThreads 每 Number of threads used in parallel version of the algorithm.					
*/


// draw the output boxes with different colors:
// 1st model: green color
// 2nd model: blue color
// 3rd model: green color
// 4th model: blue color
// ...
void Class_HoG_LatentSVM::myShowDets(Mat &im, const vector<ObjectDetection> &dets)
{
	vector<Scalar> color;
	color.push_back(Scalar(0,255,0));	// green color
	color.push_back(Scalar(255,0,0));	// blue color
	
	int num = dets.size(), colorID;
	char tmp[10];
	for(int i=0; i<num; ++i)
	{
		Rect locationRect = dets[i].rect;
		sprintf(tmp, "%.2f", dets[i].score);
		colorID = dets[i].classID % color.size();
		string scoreText = (string)tmp;
		rectangle(im, locationRect, color[colorID], 2);
		putText(im, scoreText, Point(locationRect.x+5, locationRect.y+15), 
				FONT_HERSHEY_SIMPLEX, 0.55, color[colorID], 2);
	}
}