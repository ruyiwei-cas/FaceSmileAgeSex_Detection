/// ---------------------- COPYRIGHT ------------------------ 
///
/// Copyright (C) 1985-2014 Intel Corporation.  All rights reserved.
///
/// The information and source code contained herein is the exclusive
/// property of Intel Corporation and may not be disclosed, examined
/// or reproduced in whole or in part without explicit written authorization
/// from the company.
///
/// Perceptual Application Innovation Lab, Intel Labs China.
///
///
///!
///   @file        scene.cpp
///   @brief       utility function for scene classification
///   @author      LI Xue, Intel Labs China
///   copyright reserved 2014, please do not remove this head
///
// ---------------------------------------------------------- 


// --------------- MEMORY LEAK DETECTION -------------------- 

#ifdef _DEBUG
#define DEBUG_CLIENTBLOCK new( _CLIENT_BLOCK, __FILE__, __LINE__)
#else
#define DEBUG_CLIENTBLOCK
#endif
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#ifdef _DEBUG
#define new DEBUG_CLIENTBLOCK
#endif

// ---------------------------------------------------------- 


#include <string>
#include <opencv2/opencv.hpp>
#include "compute_feature.h"
#include "constant.h"
#include "myhog.h"
#include "transferData.h"
#include <time.h>
#include "kernel_svm.h"
 
using namespace cv;
using namespace std;


// ----------------------- GLOBAL --------------------------
string root_path;
Mat Ftrain_hog2x2,Ftrain_colorhist,Vocabulary,kernel_test;
struct svm_model* svm_models[NUM_CLASS_USE];
string * class_name;
int * class_num;
// ---------------------------------------------------------


int main()
{

	//--------------- MEMORY LEAKAGE DETECTION -----------------
	//
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF|_CRTDBG_LEAK_CHECK_DF);
	//
	//----------------------------------------------------------

	initialize_scene(char *path);
	clock_t t1;		
	

	// Test all the selected test samples
	test_true();

	//cout<<endl<<"Total time: "<<double(clock()-t1)<<"ms"<<endl;	
 
	destroy_scene();
	return 0;
 
}