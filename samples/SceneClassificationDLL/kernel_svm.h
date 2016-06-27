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

#ifndef KERNEL_SVM_H
#define KERNEL_SVM_H

#include <iostream>
#include <opencv2/opencv.hpp> 

#include "svm.h"
#include "constant.h"

using namespace std;
using namespace cv;


extern string *class_name;
extern int *class_num;
extern Mat Ftrain_hog2x2;
extern Mat Ftrain_colorhist;
extern Mat kernel_test;
extern struct svm_model* svm_models[NUM_CLASS_USE];
extern string root_path;
//extern flann::Index myindex;


Mat compute_hik_kernels(Mat x, Mat y);
Mat compute_hik_kernels_test(Mat x, Mat y);
Mat compute_chi2_kernels_test(Mat x, Mat y);
Mat compute_chi2_kernels_train(Mat x);


struct svm_problem read_problem_x_from_mat(Mat data, bool be_precomputed);
void read_problem_y_from_mat(Mat label, struct svm_parameter &param, struct svm_problem &prob);
struct svm_parameter initialize_svm_param();


//Mat compute_kernel_train(int split_id, int num_class, vector<Mat> split);
Mat compute_kernel_train();
void train_svm_one_vs_all(Mat k_train, Mat ltrain);
void kernel_svm_train();


int test_svm_one_vs_all(Mat k_test);
double test_svm_one_vs_all_weight(Mat k_test_1, Mat k_test_2, double* weight);
//double kernel_svm_test();


void run_kernel_svm();
//double test_single_img(Mat img);
double test_single_img_voting(Mat im);

void test_true();
void show_result(string img_path, int gt, double label, int num);

void load_svm_models();
//void initialize_scene(char *path);
//void destroy_scene();
void clear_folders();

#endif