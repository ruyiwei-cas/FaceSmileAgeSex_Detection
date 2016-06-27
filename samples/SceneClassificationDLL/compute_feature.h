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

#ifndef COMPUTE_FEATURE_H
#define COMPUTE_FEATURE_H

#include <iostream>
#include <string>
#include <opencv2/opencv.hpp> 
#include "constant.h"

using namespace std;
using namespace cv;



extern string *class_name;
extern int *class_num;
extern Mat Vocabulary;
extern string root_path;
//extern flann::Index myindex;


struct path generatePath();
string* get_class_names();
int* get_class_num();

void compute_feature_all();

void extractFeatures_hog2x2(Mat img, string feat_path);
void extractFeatures_colorhist(Mat src, string feat_path);
void extractFeatures_colorhist_v3(Mat src, string feat_path);
Mat extractFeatures_hog(Mat im);
Mat extractFeatures_colorhist(Mat src);
Mat extractFeatures_colorhist_v3(Mat src);

Mat img_resize(Mat img, int maxSize);

Mat myhog2x2_(Mat im);
Mat myhog2x2_flann(Mat im);
int computeWords_(Mat descrs,Mat V);
Mat computeWords_flann(Mat data, int m ,int n);
Mat SPMpooling_(Mat feat);

Mat compute_124d_vector(Mat im);
Mat compute_feature_train_vocabulary();


#endif