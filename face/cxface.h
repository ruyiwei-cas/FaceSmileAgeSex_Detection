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

#ifndef _CX_FACE_HPP_
#define _CX_FACE_HPP_

#include <cv.h>
//#include <cvaux.h>

#ifdef __cplusplus
extern "C" {
#endif

#include "basetypes.hpp"

#include "cxfacedetector.hpp"
#include "cxfacetracker.hpp"
#include "cxhaarfacedetector.hpp"
#include "cxhaarfacetracker.hpp"
#include "cxsurffacedetector.hpp"
#include "cxsurffacetracker.hpp"
#include "cxcompdetector.hpp"

#include "cxslidewinfea.hpp"
#include "cxboost.hpp"
#include "cxmcboost.hpp"		 // for age
#include "cxboostfacerecog.hpp"  // face recog

#include "cxfaceutil.hpp"
#include "cxvideocap.hpp"
#include "cxoptions.hpp"


#ifdef __cplusplus
}
#endif

#endif // _CX_FACEDETECTOR_HPP_


