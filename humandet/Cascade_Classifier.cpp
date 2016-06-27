#include "Cascade_Classifier.h"
 using namespace std;
 using namespace cv;



// Initialization with several model files
Class_Cascade_Classifier::Class_Cascade_Classifier() 
{
}
// Initialization with several model files
void Class_Cascade_Classifier::Init(const vector<string> &modefiles) // modified by Ren, Haibing
{
	int num = modefiles.size();
	for(int i=0; i<num; ++i)
	{
		CascadeClassifier *tmpCascades = new CascadeClassifier(modefiles[i]);
		//bool a = tmpCascades.load(modefiles[i]);
		models.push_back(tmpCascades);
	}
}
Class_Cascade_Classifier::Class_Cascade_Classifier(const vector<string> &modefiles) 
{
	int num = modefiles.size();
	for(int i=0; i<num; ++i)
	{
		CascadeClassifier *tmpCascades = new CascadeClassifier(modefiles[i]);
		models.push_back(tmpCascades);
	}
}


Class_Cascade_Classifier::~Class_Cascade_Classifier()
{
	int classNum = models.size();
	for(int i=0;i<classNum;i++)
	{
		delete models[i];
	}
}


// 'im' can be either BRG or gray image
void Class_Cascade_Classifier::myDetect(const Mat &im, vector<ObjectDetection> &dets)
{
	double scaleFactor = 1.1;
	int minNeighbors = 3; 
	int flags = 0 | CV_HAAR_SCALE_IMAGE;
	Size minSize;
	Size maxSize;

	dets.clear();
	ObjectDetection tmpdet;
	int classNum = models.size();
	vector<Rect> foundLocations;
	for(int classID=0; classID<1; ++classID)
	{
		foundLocations.clear();
		models[classID]->detectMultiScale(im, foundLocations, scaleFactor, minNeighbors, flags, minSize, maxSize);
		// see below for detailed description of 'detectMultiScale' function
		
		int num = foundLocations.size();	
		for(int i=0; i<num; ++i)
		{
			tmpdet.rect = foundLocations[i];
			//tmpdet.score = 0;
			tmpdet.classID = classID;
			dets.push_back(tmpdet);
		}
	}
}

int  Class_Cascade_Classifier::myDetect(const Mat &im, Rect *FaceRegion, int nFaceRegionSize)
{
	double scaleFactor = 1.1;
	int minNeighbors = 3; 
	int flags = 0 | CV_HAAR_SCALE_IMAGE;
	Size minSize(50,50);
	
	Size maxSize;

	int classNum = models.size();
	vector<Rect> foundLocations;
	int nFaceNum=0;
	
	//cv::HOGDescriptor hog; // 采用默认参数  
	//hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());   // 采用已经训练好的行人检测分类器  
	//std::vector<cv::Rect> regions;
	//hog.detectMultiScale(im, regions, 0, cv::Size(8, 8), cv::Size(32, 32), 1.1, 1);
	
	
	for(int classID=0; classID<classNum; ++classID)
	{
		foundLocations.clear();
		models[classID]->detectMultiScale(im, foundLocations, scaleFactor, minNeighbors, flags, minSize, maxSize);
		// see below for detailed description of 'detectMultiScale' function
		
		int num = foundLocations.size();	

		for(int i=0; i<num; ++i)
		{
			if(nFaceNum<nFaceRegionSize)
			{
				FaceRegion[nFaceNum].x = foundLocations[i].x;
				FaceRegion[nFaceNum].y = foundLocations[i].y;
				FaceRegion[nFaceNum].height = foundLocations[i].height;
				FaceRegion[nFaceNum].width = foundLocations[i].width;
				nFaceNum++;
			}
		}
		
	}

	return nFaceNum;
}
/*
void detectMultiScale(const Mat& image, CV_OUT vector<Rect>& objects, double scaleFactor=1.1,
					int minNeighbors=3, int flags=0, Size minSize=Size(), Size maxSize=Size() );
Parameters:	
	image – Matrix of the type CV_8U containing an image where objects are detected.
	objects – Vector of rectangles where each rectangle contains the detected object.
	scaleFactor – Parameter specifying how much the image size is reduced at each image scale.
	minNeighbors – Parameter specifying how many neighbors each candidate rectangle should have to retain it.
	flags – Parameter with the same meaning for an old cascade as in the function cvHaarDetectObjects. 
			It is not used for a new cascade.
			CV_HAAR_DO_CANNY_PRUNING: use Canny edge detection to skip some regions with too few or too many edges;
			CV_HAAR_SCALE_IMAGE: scale the image instead of scaling the detector;
			CV_HAAR_FIND_BIGGEST_OBJECT: only return the detection with maximum size;
			CV_HAAR_DO_ROUGH_SEARCH: only do rough search.
			example usage: CV_HAAR_DO_CANNY_PRUNING | CV_HAAR_FIND_BIGGEST_OBJECT
	minSize – Minimum possible object size. Objects smaller than that are ignored.
	maxSize – Maximum possible object size. Objects larger than that are ignored.				
*/


// draw the output boxes with different colors:
// 1st model: yellow color
// 2nd model: aqua color
// 3rd model: magenta color
// 4th model: yellow color
// 5rd model: aqua color
// 6th model: magenta color
// ...
void Class_Cascade_Classifier::myShowDets(Mat &im, const vector<ObjectDetection> &dets)
{
	vector<Scalar> color;
	color.push_back(Scalar(0,255,255));	// yellow color
	color.push_back(Scalar(255,255,0));	// aqua color
	color.push_back(Scalar(255,0,255));	// magenta color
	
	int num = dets.size(), colorID;
	char tmp[10];
	for(int i=0; i<num; ++i)
	{
		Rect locationRect = dets[i].rect;		
		//sprintf(tmp, "%.2f", dets[i].score);
		colorID = dets[i].classID % color.size();
		string scoreText = (string)tmp;
		rectangle(im, locationRect, color[colorID], 2);
		//putText(im, scoreText, Point(locationRect.x+5, locationRect.y+15), 
		//	FONT_HERSHEY_SIMPLEX, 0.55, color[colorID], 2);
	}
}