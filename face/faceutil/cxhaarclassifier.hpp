/**
*** Copyright (C) 1985-2011 Intel Corporation.  All rights reserved.
***
*** The information and source code contained herein is the exclusive
*** property of Intel Corporation and may not be disclosed, examined
*** or reproduced in whole or in part without explicit written authorization
*** from the company.
***
*** Embedded Application Lab, Intel Labs China.
**/

#ifndef _CX_HAARCLASSIFIER_H_
#define _CX_HAARCLASSIFIER_H_

#include <opencv/cv.h>

void cxRunHaarClassiferCascade( CvHaarClassifierCascade* _cascade, 
                                CvRect rect, int step, int start_stage, 
                                CvSeq* seq_rect );

int cvRunHaarClassifierCascade( const CvHaarClassifierCascade* _cascade, \
							   CvPoint pt, int start_stage );

#endif // _CX_HAARCLASSIFIER_H_
