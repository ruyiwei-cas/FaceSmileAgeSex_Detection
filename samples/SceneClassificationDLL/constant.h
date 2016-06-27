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

#ifndef CONSTANT_H
#define CONSTANT_H

#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>  

using namespace std;
using namespace cv;


#define NUM_CLASS_USE 4
#define TEST_NUM_CLASS_USE 5
#define LEN           400
#define EPS           0.01
#define INF           1000000000
#define VDIM          300
#define SIZE          320
#define TH            0.0
#define LAYERS        3
#define INTERVAL      8
#define WEIGHT        0.5
#define NORM          1.8

struct path
{
	string _path;
	string img_path;
	string feat_path;
	string kernel_path;
	string result_path;
	string vocab_path;
};


#endif