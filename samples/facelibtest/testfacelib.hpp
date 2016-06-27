/**
*** Copyright (C) 1985-2010 Intel Corporation.  All rights reserved.
***
*** The information and source code contained herein is the exclusive
*** property of Intel Corporation and may not be disclosed, examined
*** or reproduced in whole or in part without explicit written authorization
*** from the company.
***
*** Embedded Application Lab, Intel Labs China.
**/

#ifndef _CX_TESTER_HPP_
#define _CX_TESTER_HPP_

#include "basetypes.hpp"

#define MAX_PATH 260

int consoleTestFaceLib( const char* str_video, int  trackerType = TRA_SURF, int multiviewType = VIEW_ANGLE_FRONTAL, int recognizerType = RECOGNIZER_BOOST_GB240, const char* str_facesetxml = "faceset_model.xml", int threads = 1, 
				   bool blink = true, bool smile = true, bool gender = true, bool age = true, bool recog = false, bool quiet = false, bool saveface = false, const char* sfolder = NULL, bool bEnableAutoCluster = false );

int testfaceLib_sThread( const char* str_video, int  trackerType = TRA_SURF, int multiviewType = VIEW_ANGLE_FRONTAL, int recognizerType = RECOGNIZER_BOOST_GB240, const char* str_facesetxml = "faceset_model.xml", int threads = 1, 
				   bool blink = true, bool smile = true, bool gender = true, bool age = true, bool recog = false, bool quiet = false, bool saveface = false,const char* sfolder = NULL, bool bEnableAutoCluster = false );

int testfaceLib_pThread( const char* str_video, int  trackerType = TRA_SURF, int multiviewType = VIEW_ANGLE_FRONTAL, int recognizerType = RECOGNIZER_BOOST_GB240, const char* str_facesetxml = "faceset_model.xml", int threads = 1, 
				   bool blink = true, bool smile = true, bool gender = true, bool age = true, bool recog = false, bool quiet = false, bool saveface = false, const char* sfolder = NULL, bool bEnableAutoCluster = false );

#endif // _CX_TESTER_HPP_
