/**
*** Copyright (C) 1985-2014 Intel Corporation.  All rights reserved.
***
*** The information and source code contained herein is the exclusive
*** property of Intel Corporation and may not be disclosed, examined
*** or reproduced in whole or in part without explicit written authorization
*** from the company.
***
*** Perceptual Application Innovation Lab, Intel Labs China.
**/

/*!
*   @file        scene.cpp 
*   @brief       utility function for scene classification
*   @author      LI Xue, Intel Labs China
*   copyright reserved 2014, please do not remove this head
*/

#include <iostream>
#include <opencv2/opencv.hpp> 
#include <string>
#include <fstream>
#include <time.h>
#include <io.h>

#include "svm.h"
#include "kernel_svm.h"
#include "constant.h"
#include "transferData.h"
#include "compute_feature.h"

using namespace std;
using namespace cv;

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))




////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the Photo indexing Utility
///		written by Li, Xue
///   
/// Acknowledge     :
///
/// Function name   : compute_hik_kernels_test
/// Description     : compute HIK kernels between supporting vectors of x and the entire y
///
/// Argument        : x -- training matrix, with each row representing a feature vector 
/// Argument        : y -- test matrix, with each row representing a feature vector 
///
/// Return type     : Mat -- the obtained kernel matrix
///
/// Create Time     : 2014-12-2  17:13
///
///
/// Side Effect     : 
///
///////////////////////////////////////////////////////////////////////////////////////////////
Mat compute_hik_kernels_test(Mat x, Mat y)
{
	Mat kernel(x.rows, 1, CV_32FC1);
	kernel.setTo(0);
	int *p_test = kernel_test.ptr<int>(0);
	
	float *p, *px, *py, s;
	for(int i=0;i<x.rows;++i)
	{
		if(p_test[i] == 1)
		{
			p = kernel.ptr<float>(i);
			px = x.ptr<float>(i);
		
			py = y.ptr<float>(0);
			s = 0;
			for(int k=0;k<x.cols;++k)
			{
				
				s += px[k]<py[k]?px[k]:py[k];
				if(s<0)
				{
					int test = 1;
				}
			}
			
			p[0] = s;
		}
		
	}

	return kernel;
}




////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the Photo indexing Utility
///		written by Li, Xue
///   
/// Acknowledge     :
///
/// Function name   : test_svm_one_vs_all_weight
/// Description     : test phase for a single test sample, with class-specific weight for hog and color.
///
/// Argument        : k_test_1 -- hog2x2 kernel matrix between training samples and test samples
/// Argument        : k_test_2 -- color kernel matrix between training samples and test samples
/// Argument        : weight   -- weight for combining k_test_1 and k_test_2
///
/// Return type     : double -- the int part indicates the category index, 0-based; 
///                   and the fractional part indicates the confidence/10.
///                   e.g. 1.08 means the second category, with a confidence of 0.8.
///                   
///
/// Create Time     : 2014-12-5  12:25
///
///
/// Side Effect     : 
///
///////////////////////////////////////////////////////////////////////////////////////////////
double test_svm_one_vs_all_weight(Mat k_test_1, Mat k_test_2, double* weight)
{		
	struct path myPath = generatePath();
	fstream f1(myPath.result_path + "\\result\\score.txt",ios::app);
	//f1<<endl;
	//char name[LEN];	
	double best = -1, score, tmp;
	int class_hat = 0;
	float * pd;
	
	Mat k_test;
	
	for(int i=0;i<NUM_CLASS_USE;++i)
	{					
		addWeighted(k_test_1,weight[i],k_test_2,1-weight[i],0,k_test);

		tmp = 0;
		for(int k=0;k<svm_models[i]->l;++k)
		{
			pd = k_test.ptr<float>(svm_models[i]->SV[k]->value-1);
			tmp += svm_models[i]->sv_coef[0][k] * pd[0];			
		}
		score = tmp - svm_models[i]->rho[0];
		f1<<score<<"  ";
		if(score > best)
		{
			best = score;
			class_hat = i;
		}

		//svm_free_and_destroy_model(&model);
	}

	if(best<TH)
	{
		return -1;
	}
	f1<<endl;
	f1.close();

	double tmp1 = best>=NORM?NORM:best;
	double result = double(class_hat) + tmp1/NORM/10;
	
	return result;
	//return double(class_hat);
}




////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the Photo indexing Utility
///		written by Li, Xue
///   
/// Acknowledge     :
///
/// Function name   : test_single_img
/// Description     : input an image and return its category. This is the main interface.
///
/// Argument        : im -- the image to be tested.
///
/// Return type     : double -- the int part indicates the category index, 0-based; 
///                   and the fractional part indicates the confidence/10.
///                   e.g. 1.08 means the second category, with a confidence of 0.8.
///
/// Create Time     : 2014-12-5  12:25
///
///
/// Side Effect     : 
///
///////////////////////////////////////////////////////////////////////////////////////////////
extern "C" __declspec(dllexport)
double test_single_img(Mat im)
{
	char name[LEN];
	clock_t t1;

	// Extract feature
	//t1 = clock();	
	Mat img;
	img = img_resize(im, SIZE);		
	Mat f1 = extractFeatures_hog(img);
	//cout<<"Time for computing features = "<<clock()-t1<<"ms"<<endl;

	Mat f2 = extractFeatures_colorhist_v3(img);

	// Compute kernel
	t1 = clock();		
	Mat k_test_1 = compute_hik_kernels_test(Ftrain_hog2x2, f1);
	Mat k_test_2 = compute_hik_kernels_test(Ftrain_colorhist, f2);

	delete [] f1.data;

	double weight[NUM_CLASS_USE] = {0.5,0.5,0.5,0.5};
	double label = test_svm_one_vs_all_weight(k_test_1, k_test_2, weight);	
	
	
	return label;
}




////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the Photo indexing Utility
///		written by Li, Xue
///   
/// Acknowledge     :
///
/// Function name   : test_single_img_voting
/// Description     : input an image, extract its center, left and right parts, test these 3 sub-images, 
///                   and obtain the final category using a voting scheme. This is the main interface.
///
/// Argument        : im -- the image to be tested.
///
/// Return type     : double -- the int part indicates the category index, 0-based; 
///                   and the fractional part indicates the confidence/10.
///                   e.g. 1.08 means the second category, with a confidence of 0.8.
///
/// Create Time     : 2014-12-12  13:44
///
///
/// Side Effect     : the extracted sub-images should not be too small. The minimal size of each sub-image should 
///                   contain at least 50 pixels.
///
///////////////////////////////////////////////////////////////////////////////////////////////
double test_single_img_voting(Mat im)
{
	double eps = 1e-6;
	int height = im.rows;
	int width = im.cols;
	double score[NUM_CLASS_USE] = {0};
	int hist[NUM_CLASS_USE] = {0};
	double ratio1 = 0.1;
	double ratio2 = 0.5;
	int left1, left2, right1, right2;

	// center
	Mat im_center = im.rowRange(int(height*ratio1), int(height*(1-ratio1))).colRange(int(width*ratio1),int(width*(1-ratio1)));
	double label_center = test_single_img(im_center);
	int tag1 = int(label_center);
	if(tag1>-1)
	{
		score[tag1] += 10*(label_center-tag1);
		hist[tag1] ++;
	}
	

	// left
	if(ratio2<=0.5)
	{
		left1  = int(width*(0.5-ratio2));
		right1 = int(width*0.5);

		left2  = right1;
		right2 = int(width*(0.5+ratio2));
	}
	else
	{
		left1  = 0;
		right1 = int(width*ratio2);

		left2  = int(width*(1-ratio2));
		right2 = width;
	}
	
	Mat im_left = im.colRange(left1,right1);
	double label_left = test_single_img(im_left);
	int tag2 = int(label_left);
	if(tag2>-1)
	{
		score[tag2] += 10*(label_left-tag2);
		hist[tag2] ++;
	}	

	// right
	Mat im_right = im.colRange(left2,right2);
	double label_right = test_single_img(im_right);
	int tag3 = int(label_right);
	if(tag3>-1)
	{
		score[tag3] += 10*(label_right-tag3);
		hist[tag3] ++;
	}
	

	// voting
	for(int i=0;i<NUM_CLASS_USE;++i)
	{
		double tmp = hist[i]>eps?hist[i]:eps;
		score[i] = score[i]/tmp;
	}

	double maxval = 0;
	int tag = -1;
	for(int i=0;i<NUM_CLASS_USE;++i)
	{
		if(score[i]>maxval)
		{
			maxval = score[i];
			tag = i;
		}
	}	
	if(maxval>=0.1)
	{
		double label = double(tag) + 0.1*(maxval);
	
		return label;
	}
	else
	{
		return -1;
	}
	
}




////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the Photo indexing Utility
///		written by Li, Xue
///   
/// Acknowledge     :
///
/// Function name   : test_true
/// Description     : test all the test samples and return the average precision, designed for parameter tuning.
///
/// Argument        : none.
///
/// Return type     : double -- the average precision, which is required in parameter tuning.
///
/// Create Time     : 2014-12-5  9:10
///
///
/// Side Effect     : 
///
///////////////////////////////////////////////////////////////////////////////////////////////
/*
void test_true()
{
	char name[LEN];
	clock_t t1;
	struct path myPath = generatePath();

	// Initialize	
	if(Ftrain_hog2x2.rows == 0)
	{
		sprintf(name,"%s\\X_hog2x2_%03d.dat", myPath.feat_path.c_str(), SIZE);
		Ftrain_hog2x2 = loadFile2Mat(string(name));
	}
	if(Ftrain_colorhist.rows == 0)
	{
		sprintf(name,"%s\\X_colorhist_%03d.dat", myPath.feat_path.c_str(), SIZE);
		Ftrain_colorhist = loadFile2Mat(string(name));
	}
	if(kernel_test.rows == 0)
	{
		sprintf(name,"%s\\SV_%03d.dat", myPath.result_path.c_str(), SIZE);
		kernel_test = loadFile2Mat(string(name));
	}	
	
	//load_svm_models();
	
	string path = myPath._path + "\\image_test";
	string result_path = myPath.result_path + "\\result";
	

	// read test_num.dat
	string tmp = myPath._path + "\\test_num.dat";
	Mat test_num = loadFile2Mat(tmp);
	int * p = test_num.ptr<int>(0);

	
	double prec[TEST_NUM_CLASS_USE];
	double recall[TEST_NUM_CLASS_USE];

	int result[TEST_NUM_CLASS_USE] = {0};
	int pred[TEST_NUM_CLASS_USE] = {0};

	char mycmd[LEN];
	for (int j=0;j<TEST_NUM_CLASS_USE;++j)
	{				
		sprintf(mycmd,"%s\\%s",result_path.c_str(),class_name[j].c_str());
		if ( access(mycmd,0) ) 	
		{
			sprintf(mycmd,"mkdir %s\\%s",result_path.c_str(),class_name[j].c_str());				
			system(mycmd);
		}
	}

	// confusion matrix
	//Mat confusion(NUM_CLASS_USE,NUM_CLASS_USE,CV_64FC1);
	//confusion.setTo(0);

	double *pc;
	double label;
	
	
	for(int i=0;i<TEST_NUM_CLASS_USE;++i)
	{
		int num = p[i];	
		//pc = confusion.ptr<double>(i);
		
		for(int j=0;j<num;++j)
		{
			// test for each image
			sprintf(name,"%s\\%s\\image_%04d.jpg",path.c_str(),class_name[i].c_str(),j+1);

			t1 = clock();
			fstream f2(myPath.result_path + "\\result\\score.txt",ios::app);
			f2<<i<<"-"<<j<<"   ";
			f2.close();
			//label = test_single_img(imread(name));
			label = test_single_img_voting(imread(name));// voting
			
			//cout<<endl<<"test_image = "<<clock()-t1<<"ms"<<endl;
			
			// save image with label printed
				
			

			//if (double(label) - clabel>0)
			//{
			show_result(name, i, label, j+1);
			//}
			
		    if (label == -1)
			{
			    label = 4;
			}

			int clabel = int(label);

			// count
			if(clabel == i)
			{
				result[clabel]++;
			}
			//pc[label]++;
			if(clabel>-1)
			{
				pred[clabel]++;
			}
		}
		
		
		
		/*for(int k=0;k<NUM_CLASS_USE;++k)
		{
			pc[k] /= double(num);
		}*/

	/*}

	

	// calculate precision and recall
	double avg_p = 0, avg_r = 0;
	for(int i=0;i<TEST_NUM_CLASS_USE;++i)
	{
		prec[i] = double(result[i])/double(pred[i]);
		recall[i] = double(result[i])/double(p[i]);
		avg_p += prec[i];
		avg_r += recall[i];
	}

	// output
	tmp = myPath.result_path + "\\result\\Results.txt";
	ofstream f1(tmp);
	for(int i=0;i<TEST_NUM_CLASS_USE;++i)
	{
		f1<<class_name[i]<<" -- Precision = "<<prec[i]<<", Recall = "<<recall[i]<<endl;
	}	
	f1<<endl<<"AP = "<<avg_p/double(TEST_NUM_CLASS_USE)<<", AR = "<<avg_r/double(TEST_NUM_CLASS_USE)<<endl;
	f1.close();

	//tmp = myPath.result_path + "\\result\\Confusion.dat";
	//writeMat2File(string(tmp), confusion);

}


*/

////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the Photo indexing Utility
///		written by Li, Xue
///   
/// Acknowledge     :
///
/// Function name   : show_result
/// Description     : write category labels and confidence on to images and save them into designate paths.
///
/// Argument        : img_path -- the path to load the test image;
/// Argument        : gt       -- ground truth of the current image;
/// Argument        : label    -- the predicted label and the confidence (label + confidence/10);
/// Argument        : num      -- number used to identify the test image.
///
/// Return type     : none.
///
/// Create Time     : 2014-12-5  9:23
///
///
/// Side Effect     : 
///
///////////////////////////////////////////////////////////////////////////////////////////////
void show_result(string img_path, int gt, double label, int num)
{
	double p = 0.1,q = 1.0 - p;//0.382
	Mat im  = imread(img_path);
	
	// parsing label and confidence score 
	int clabel = int(label);
	double score = 10*(double(label) - clabel);

	struct path myPath1 = generatePath();
	string result_path = myPath1.result_path + "\\result\\"+class_name[gt];
	string text;
	char str[LEN];
	if(clabel>-1)
	{
		sprintf(str,"%s %.2f",class_name[clabel].c_str(),score);
		text = string(str);
	}
	else
	{
		text = "Denied";
	}
	 

	int fontFace = FONT_HERSHEY_SCRIPT_SIMPLEX;
	double fontScale = 0.5;
	int thickness = 1;	

	int baseline=0;
	Size textSize = getTextSize(text, fontFace, fontScale, thickness, &baseline);

	Point textOrg((im.cols - textSize.width)*p, (im.rows - textSize.height)*p);

	if(clabel>-1)
	{
		putText(im, text, textOrg, fontFace, fontScale, Scalar(0,255,0,0), thickness, 8);	
	}
	else
	{
		putText(im, text, textOrg, fontFace, fontScale, Scalar(0,0,255,0), thickness, 8);	
	}	


	char name[LEN];
	sprintf(name,"%s\\image_%04d.jpg",result_path.c_str(),num);
	imwrite(name,im);

}




////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the Photo indexing Utility
///		written by Li, Xue
///   
/// Acknowledge     :
///
/// Function name   : load_svm_models
/// Description     : load pre-trained SVM models, which enables reuse of the models.
///
/// Argument        : none.
///
/// Return type     : none.
///
/// Create Time     : 2014-12-5  9:31
///
///
/// Side Effect     : 
///
///////////////////////////////////////////////////////////////////////////////////////////////
void load_svm_models()
{		
	char name[LEN];
	struct path myPath = generatePath();

	for(int i=0;i<NUM_CLASS_USE;++i)
	{
		sprintf(name,"%s\\SVM_%03d_class_%03d.dat",myPath.result_path.c_str(), SIZE, i);
		/*struct svm_model *model = svm_load_model(name);
		svm_models[i] = model;*/
		if (!_access(name, 0))
		{
			svm_models[i] = svm_load_model(name);
		}
		else cout << "Can't Load  " << name << endl;
	}

}




////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the Photo indexing Utility
///		written by Li, Xue
///   
/// Acknowledge     :
///
/// Function name   : initialize_scene
/// Description     : load pre-saved variables and pre-trained SVM models.
///
/// Argument        : none.
///
/// Return type     : none.
///
/// Create Time     : 2014-12-5  12:41
///
///
/// Side Effect     : 
///
///////////////////////////////////////////////////////////////////////////////////////////////
extern "C" __declspec(dllexport)
void initialize_scene(char *path )
{
	root_path = path;	
	struct path myPath = generatePath();
	char name[LEN];

	class_name = get_class_names();
	class_num = get_class_num();

	// load vocabulary
	sprintf(name,"%s\\hog2x2_%d_%d.dat",myPath.vocab_path.c_str(),SIZE,VDIM);		
	if (!_access(name, 0))
	{
		Vocabulary = loadFile2Mat(string(name));
	}
	else cout << "Can't Load  " << name << endl;
	

	// load Ftrain
	sprintf(name,"%s\\X_hog2x2_%03d.dat", myPath.feat_path.c_str(), SIZE);
	if ( !_access(name,0) ) 
	{		
		Ftrain_hog2x2 = loadFile2Mat(string(name));
	}
	else cout << "Can't Load  " << name << endl;

	sprintf(name,"%s\\X_colorhist_%03d.dat", myPath.feat_path.c_str(), SIZE);
	if ( !_access(name,0) ) 
	{		
		Ftrain_colorhist = loadFile2Mat(string(name));
	}
	else cout << "Can't Load  " << name << endl;

	// load kernel_test
	sprintf(name,"%s\\SV_%03d.dat", myPath.result_path.c_str(), SIZE);
	if ( !_access(name,0) ) 
	{		
		kernel_test = loadFile2Mat(string(name));
	}
	else cout << "Can't Load  " << name << endl;
	
	// load models
	load_svm_models();	
	
}




////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the Photo indexing Utility
///		written by Li, Xue
///   
/// Acknowledge     :
///
/// Function name   : destroy_scene
/// Description     : frees dynamic memories assigned in the initialization module.
///
/// Argument        : none.
///
/// Return type     : none.
///
/// Create Time     : 2014-12-5  9:36
///
///
/// Side Effect     : 
///
///////////////////////////////////////////////////////////////////////////////////////////////
extern "C" __declspec(dllexport)
void destroy_scene()
{
	delete [] class_name;
	delete [] class_num;

	Vocabulary.release();
	Ftrain_colorhist.release();
	Ftrain_hog2x2.release();
	kernel_test.release();

	for(int i=0;i<NUM_CLASS_USE;++i)
	{
		svm_free_and_destroy_model(&svm_models[i]);

		//svm_free_and_destroy_model(&model);	
		/*svm_destroy_param(&para);
		for(int j=0;j<prob.l;++j)
		{
			delete [] prob.x[j];	
		}
		free(prob.y);
		free(prob.x);*/
	}
}

