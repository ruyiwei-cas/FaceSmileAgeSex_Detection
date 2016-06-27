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

#ifndef _CX_COMPDETECTOR_HPP_
#define _CX_COMPDETECTOR_HPP_

//#include <cvaux.h>
#include <opencv/cv.h>
//#include <cvaux.h>

#include "cxcompdetbase.hpp"

// detector of face components, such as eyes, nose and mouth
class CxCompDetector : public CxCompDetBase
{

public:
	static const int MODE_TRACK			= 0;	//track mode
	static const int MODE_DETECT		= 1;	//detect mode

    static const int LEYE_INNER			= 0;    // inner corner of left  eye
    static const int LEYE_OUTER			= 1;    // outer corner of left  eye
    static const int REYE_INNER			= 2;    // inner corner of right eye
    static const int REYE_OUTER			= 3;    // outer corner of right eye
    static const int MOUTH_LCRN			= 4;    // left  corner of mouth
    static const int MOUTH_RCRN 		= 5;    // right corner of mouth

    static const int COMP_COUNT			= 6;    // total components
    static const int NRECTBUFF   		= 16;   // tracking face rect buff
	static const int REGION_SIZE		= 5;
	

	static const int NGROUP				= 4;
	static const int NPARAMETERS		= 6;

    // constructor
    CxCompDetector( const char* xml_eye_leftcorner, 
		            const char* xml_mth_leftcorner );

    // destructor
    ~CxCompDetector( );

    // detect 6 points within 'rc_face', output to 'pt_comp'
	virtual bool detect( const IplImage* image, CvRect* rect, CvPoint2D32f points[], float parameters[]=NULL, float angle=0 );

	virtual bool track( const IplImage* image, CvRect* rect, CvPoint2D32f points[], float parameters[]=NULL, float angle=0 );
    // retrieve each comp
    virtual CvPoint2D32f getPoint( int comp ) const { return pt_dst_comps[comp]; }

protected:
	static const float FACTOR_1;
	static const float FACTOR_2;
	static const float FACTOR_3;

	static const float AVG_L1;	//average(L1)
	static const float STDVAR_L1;	//Standard deviation(L1)
	static const float AVG_L2_L1;	//average(L2/L1)
	static const float AVG_L3_L1;	//average(L3/L1)
	static const float STDVAR_L2_L1;	//Standard deviation(L2/L1)
	static const float STDVAR_L3_L1;	//Standard deviation(L3/L1)


    CvMemStorage*            storage;
    CvHaarClassifierCascade*	cascade_eye_lc;
	CvHaarClassifierCascade*	cascade_eye_rc;
    CvHaarClassifierCascade*	cascade_mth_lc;
	CvHaarClassifierCascade*	cascade_mth_rc;
	
	CvHaarClassifierCascade*    cascade_lut[COMP_COUNT];

	CvSize						cascade_size_lut[COMP_COUNT];
	CvRect						cascade_search_rect[COMP_COUNT];

    CvSize			sz_canvas;          // canvas size, being 96x96
    CvRect			rc_detect;          // detected face rect in canvas
    CvRect			rc_comps[COMP_COUNT];   // search region
	CvPoint2D32f	pt_dst_comps[COMP_COUNT];   // detected points
	CvPoint2D32f	pt_src_comps[COMP_COUNT];   // detected points
	float			para_dst[NPARAMETERS];

	cv::Mat    mat_eye_lc_dstrb;         // distribution of left eye corner
	cv::Mat    mat_eye_rc_dstrb;         // distribution of right eye corner
	cv::Mat    mat_mth_lc_dstrb;         // distribution of left mouth corner
	cv::Mat    mat_mth_rc_dstrb;         // distribution of right mouth corner
	cv::Mat*	mat_dstrb_lut[COMP_COUNT];

    cv::Mat    mat_img_c3;         // patch of color image
    cv::Mat    mat_img_c1;         // patch of grayscale image
    cv::Mat    mat_sum;            // integral (sum) image
    cv::Mat    mat_sqsum;          // squared sum image
    cv::Mat    mat_tilted;         // tilted sum image

	float		multiParameters[NGROUP][NPARAMETERS];
	int			mode;

	cv::Mat	mat_delta_parameters;
	cv::Mat	mat_jacobian;
	cv::Mat	mat_current_energy;
	cv::Mat	mat_current_temp_energy;
	cv::Mat	mat_current_parameters;
	cv::Mat	mat_new_parameters;

	cv::Mat	mat_hessian;
	cv::Mat	mat_soln;
	cv::Mat	mat_invertH;
	cv::Mat	mat_temp_calt_par;
	cv::Mat	mat_AT;

	void cxMirrorHaarClassifierCascade( CvHaarClassifierCascade* cascade );
	int  RunHaarClassifier( );
	void GenerateMultiOriginalParameters( float multiParameters[NGROUP][NPARAMETERS] );
	void CalcParametersFrom6Points( CvPoint* locations, float* parameters );
	void CalcParametersFrom6Points( CvPoint2D32f* locations, float* parameters );

	void Calc6PointsFromParameters(float* parameters, CvPoint2D32f* locations);
	bool IsWithinImage(CvPoint2D32f* points);
	bool Energy(CvPoint2D32f* cornersLocations, CvMat *result);
	bool CalcJacobian(CvMat* jacobianMat, CvMat* parametersMat);
	void CalcNewParameters( CvMat *matA, CvMat* fxMat, CvMat *oldMatX, CvMat *newMatX );
	bool NewtonIteration( float *parameters, float *energySum );
	bool IsFitShape( float *parameters );
	
	int				rect_buff_idx;
	CvRect			rect_buff[NRECTBUFF];
	CvPoint2D32f	fpt6s_buff[NRECTBUFF][COMP_COUNT];
};




#endif // _CX_COMPDETECTOR_HPP_