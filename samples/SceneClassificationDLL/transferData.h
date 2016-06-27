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

#ifndef TRANSFERDATA_H
#define TRANSFERDATA_H

#include <opencv2/opencv.hpp>  
#include <string>
using namespace std;
using namespace cv;

Mat loadFile2Mat(string filename);
void writeMat2File(string filename, Mat data);
void display_Mat(Mat data);
#endif