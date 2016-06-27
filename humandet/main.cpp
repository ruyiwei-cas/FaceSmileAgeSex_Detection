// demo for human body detection @Intel-Intern

#include "basic_functions.h"
#include "HoG_LinearSVM.h"
#include "HoG_LatentSVM.h"
#include "Cascade_Classifier.h"

string dirDatasets = "C:/Users/yunsheng/Documents/_JiangYS/Workstation/Datasets";

int main()
{
	// Test images and results directory ==============================================
	string dirTestImg = dirDatasets + "/ARLFacePhoto";
	string dirResults = dirDatasets + "/Results_ARLFacePhoto";	
	//string dirTestImg = dirDatasets + "/YMTestPhoto";
	//string dirResults = dirDatasets + "/Results_YMTestPhoto";


	// Declare the detection model and dets-structure =========================================

	// HoG_LinearSVM
	Class_HoG_LinearSVM model_HoG_LinearSVM(Size(64,128));
	//Class_HoG_LinearSVM model_HoG_LinearSVM(Size(48,96));
	vector<Class_HoG_LinearSVM::ObjectDetection> dets_HoG_LinearSVM;

	// HoG_LatentSVM
	vector<string> modelfiles_HoG_LatentSVM;
	modelfiles_HoG_LatentSVM.push_back("./ModelFiles/HoG_LatentSVM/person.xml");
	//modelfiles_HoG_LatentSVM.push_back("./ModelFiles/HoG_LatentSVM/upperbody_calvin.xml");
	Class_HoG_LatentSVM model_HoG_LatentSVM(modelfiles_HoG_LatentSVM);
	vector<Class_HoG_LatentSVM::ObjectDetection> dets_HoG_LatentSVM;

	// Cascade_Classifier
	vector<string> modelfiles_Cascades;
	modelfiles_Cascades.push_back("./ModelFiles/Haar_Cascades/haarcascade_frontalface_alt2.xml");
	modelfiles_Cascades.push_back("./ModelFiles/Haar_Cascades/haarcascade_profileface.xml");
	//modelfiles_Cascades.push_back("./ModelFiles/Haar_Cascades/haarcascade_mcs_upperbody.xml");	
	Class_Cascade_Classifier model_Cascades(modelfiles_Cascades);
	vector<Class_Cascade_Classifier::ObjectDetection> dets_Cascades;


	// Do body-detection for each image =========================================================
	vector<string> imgList;
	readDirectory(dirTestImg, imgList, true, true);
	TickMeter	time_HoG_LinearSVM,
				time_HoG_LatentSVM,
				time_Cascades;
	for(vector<string>::iterator iter=imgList.begin(); iter!=imgList.end(); ++iter)
	{
		string imPath = dirTestImg + "/" + (*iter);
		Mat im_BGR = imread(imPath);
		myResize(im_BGR, 500);									// optional
		Mat im_gray; cvtColor(im_BGR, im_gray, CV_BGR2GRAY);	// optional
		//equalizeHist(im_gray, im_gray);						// optional, hist-equalization is only valid for gray image.

		Mat im_BGR_flip, im_gray_flip;
		flip(im_BGR, im_BGR_flip, 1);
		flip(im_gray, im_gray_flip, 1);

		// HoG_LinearSVM
// 		time_HoG_LinearSVM.reset();	time_HoG_LinearSVM.start();
// 		model_HoG_LinearSVM.myDetect(im_gray,dets_HoG_LinearSVM);	// BGR or gray (almost the same detection time !)
// 		time_HoG_LinearSVM.stop();

		// HoG_LatentSVM
		time_HoG_LatentSVM.reset();	time_HoG_LatentSVM.start();
		model_HoG_LatentSVM.myDetect(im_BGR, dets_HoG_LatentSVM);	// input image must be BGR !
		time_HoG_LatentSVM.stop();

		// Cascade_Classifier
		time_Cascades.reset(); time_Cascades.start();
		model_Cascades.myDetect(im_gray, dets_Cascades);			// BGR or gray
		time_Cascades.start();
		

		// Show the results: ==================================================================
		cout<<*iter<<": "<<time_HoG_LinearSVM.getTimeSec()<<", "		// detection time
						 <<time_HoG_LatentSVM.getTimeSec()<<", "
						 <<time_Cascades.getTimeSec()<<endl;		
		//imshow("Original Image", im_BGR);
		model_HoG_LinearSVM.myShowDets(im_BGR, dets_HoG_LinearSVM);
		model_HoG_LatentSVM.myShowDets(im_BGR, dets_HoG_LatentSVM);
		model_Cascades.myShowDets(im_BGR, dets_Cascades);
		imshow("Detections", im_BGR);

		// Save the results
		string rstPath = dirResults + "/" + (*iter);
		saveImg(rstPath, im_BGR);
		waitKey(1);


		// Flip the image: ====================================================================
		//model_HoG_LinearSVM.myDetect(im_gray_flip, dets_HoG_LinearSVM);
		model_HoG_LatentSVM.myDetect(im_BGR_flip, dets_HoG_LatentSVM);
		model_Cascades.myDetect(im_gray_flip, dets_Cascades);
		
		for(int detNum=dets_HoG_LinearSVM.size(), i=0; i<detNum; ++i)
			dets_HoG_LinearSVM[i].rect.x = im_BGR.cols - dets_HoG_LinearSVM[i].rect.x - dets_HoG_LinearSVM[i].rect.width;
		for(int detNum=dets_HoG_LatentSVM.size(), i=0; i<detNum; ++i)
			dets_HoG_LatentSVM[i].rect.x = im_BGR.cols - dets_HoG_LatentSVM[i].rect.x - dets_HoG_LatentSVM[i].rect.width;
		for(int detNum=dets_Cascades.size(), i=0; i<detNum; ++i)
			dets_Cascades[i].rect.x = im_BGR.cols - dets_Cascades[i].rect.x - dets_Cascades[i].rect.width;

		flip(im_BGR_flip, im_BGR_flip, 1);
		model_HoG_LinearSVM.myShowDets(im_BGR_flip, dets_HoG_LinearSVM);
		model_HoG_LatentSVM.myShowDets(im_BGR_flip, dets_HoG_LatentSVM);
		model_Cascades.myShowDets(im_BGR_flip, dets_Cascades);
		
		imshow("flip", im_BGR_flip);	
		rstPath = rstPath.substr(0, rstPath.size()-4) + "_flip.jpg";
		saveImg(rstPath, im_BGR_flip);
		waitKey(1);
	}
	

	return 0;
}

