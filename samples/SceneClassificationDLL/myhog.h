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

#ifndef MYHOG_H
#define MYHOG_H

#include <iostream>
#include <opencv2/opencv.hpp> 

using namespace std;
using namespace cv;


static inline double min(double x, double y) { return (x <= y ? x : y); }
static inline double max(double x, double y) { return (x <= y ? y : x); }

static inline int min(int x, int y) { return (x <= y ? x : y); }
static inline int max(int x, int y) { return (x <= y ? y : x); }


Mat features_hog_1(Mat img, int sbin);
int compute_bin(double dx, double dy);
void interpolation_bilinear(double *hist, double cx, double cy, int cell_h, int cell_w, double val, int bin);
void compute_weighted_hist(double *hist, Mat img, int sbin);
Mat compute_hog2x2_final(double *hist, int cell_h,int cell_w, double *C);

#endif