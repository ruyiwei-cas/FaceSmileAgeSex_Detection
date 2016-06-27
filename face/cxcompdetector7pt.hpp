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

#ifndef _CX_COMPDETECTOR7PT_HPP_
#define _CX_COMPDETECTOR7PT_HPP_

#include <opencv/cv.h>

#include "cxcompdetbase.hpp"

// detector of face components, such as eyes, nose and mouth
class CxCompDetector7pt : public CxCompDetBase
{

public:
	static const int MODE_TRACK			= 0;	//track mode
	static const int MODE_DETECT		= 1;	//detect mode

    static const int LEYE_OUTER			= 0;    // outer corner of left  eye
    static const int LEYE_INNER			= 1;    // inner of left  eye
    static const int REYE_INNER			= 2;    // inner corner of right eye
    static const int REYE_OUTER			= 3;    // outer corner of right eye
    static const int MOUTH_LCRN			= 4;    // left  corner of mouth
	static const int MOUTH_RCRN 		= 5;    // right corner of mouth
	static const int NOSE_TIPS 			= 6;    // right corner of mouth

    static const int COMP_COUNT			= 7;    // total components
    static const int NRECTBUFF   		= 16;   // tracking face rect buff
	static const int REGION_SIZE		= 7;

	static const int NGROUP				= 6;
	static const int NPARAMETERS		= 7;

    // constructor
	CxCompDetector7pt( const char* xml_eye_leftcorner, 
		const char* xml_mth_leftcorner,
		const char* xml_nose,
        const float* std_shape = NULL );

    // destructor
    ~CxCompDetector7pt( );

    // detect 7 points within 'rc_face', output to 'pt_comp'
	virtual bool detect( const IplImage* image, CvRect* rect, CvPoint2D32f points[], float parameters[]=NULL, float angle=0 );
	virtual bool track( const IplImage* image, CvRect* rect, CvPoint2D32f points[], float parameters[]=NULL, float angle=0 );

    // retrieve each comp
    virtual CvPoint2D32f getPoint( int comp ) const { return pt_dst_comps[comp]; }

    ///////////////////////////////////////////////////////////
	float energy_from_7vtx( IplImage* image, const CvPoint2D32f cornersLocations[] );
    float comp_energy( const IplImage* image, const CvPoint2D32f points[] );

protected:
	static const float FACTOR_1;
	static const float FACTOR_2;
	static const float FACTOR_3;
	static const float FACTOR_4;

	static const float AVG_PITCH_ANGLE;
	static const float STDVAR_PITCH_ANGLE;
	static const float AVG_YAW_ANGLE;
	static const float STDVAR_YAW_ANGLE;
	static const float AVG_ROLL_ANGLE;
	static const float STDVAR_ROLL_ANGLE;

	static const float AVG_MTH_L;
	static const float STDVAR_MTH_L;


    CvMemStorage*            storage;
    CvHaarClassifierCascade*	cascade_eye_lc;
	CvHaarClassifierCascade*	cascade_eye_rc;
    CvHaarClassifierCascade*	cascade_mth_lc;
	CvHaarClassifierCascade*	cascade_mth_rc;
	CvHaarClassifierCascade*	cascade_nose;
	
	CvHaarClassifierCascade*    cascade_lut[COMP_COUNT];

	CvSize						cascade_size_lut[COMP_COUNT];
	CvRect						cascade_search_rect[COMP_COUNT];

    CvSize      sz_canvas;          // canvas size, being 96x96
    CvRect      rc_detect;          // detected face rect in canvas
	CvPoint2D32f      pt_dst_comps[COMP_COUNT];   // detected points
	CvPoint2D32f      pt_src_comps[COMP_COUNT];   // detected points
	float		fl_stdshape_data[COMP_COUNT*3];


	cv::Mat    mat_eye_lc_dstrb;         // distribution of left eye corner
	cv::Mat    mat_eye_rc_dstrb;         // distribution of right eye corner
	cv::Mat    mat_mth_lc_dstrb;         // distribution of left mouth corner
	cv::Mat    mat_mth_rc_dstrb;         // distribution of right mouth corner
	cv::Mat    mat_nose_dstrb;         // distribution of nose 

	cv::Mat*	mat_dstrb_lut[COMP_COUNT];

    cv::Mat    mat_img_c3;         // patch of color image
    cv::Mat    mat_img_c1;         // patch of grayscale image
    cv::Mat    mat_sum;            // integral (sum) image
    cv::Mat    mat_sqsum;          // squared sum image
    cv::Mat    mat_tilted;         // tilted sum image

	float multiParameters[NGROUP][NPARAMETERS];
	int			mode;

	cv::Mat mat_delta_parameters;
	cv::Mat mat_jacobian;
	cv::Mat mat_current_energy;
	cv::Mat mat_current_temp_energy;
	cv::Mat mat_current_parameters;
	cv::Mat mat_new_parameters;

	cv::Mat mat_hessian;
	cv::Mat mat_soln;
	cv::Mat mat_invertH;
	cv::Mat mat_temp_calt_par;
	cv::Mat mat_AT;

	void cxMirrorHaarClassifierCascade( CvHaarClassifierCascade* cascade );
	int  RunHaarClassifier( );
	void AdjustNose(float* parameters, CvPoint2D32f pt_nose);
	void GenerateMultiOriginalParameters(float parameters[NGROUP][NPARAMETERS]);
	void GenerateOriginalParameters( float parameters[NPARAMETERS] );
	void CalcParametersFrom6Points( CvPoint2D32f * locations, float* parameters );
	void Calc7PointsFromParameters( float* parameters, CvPoint2D32f* locations );
	bool IsWithinImage( CvPoint2D32f* points );

	bool Energy( float* parameters, CvMat *result );
	//bool Energy( CvPoint* cornersLocations, CvMat *result );

	bool CalcJacobian( CvMat *jacobianMat, CvMat *parametersMat );
	void CalcNewParameters( CvMat *matA, CvMat* fxMat, CvMat *oldMatX, CvMat *newMatX );
	bool NewtonIteration( float *parameters, float *energySum );
	bool IsFitShape( float *parameters );
	
	int			rect_buff_idx;
	CvRect      rect_buff[NRECTBUFF];
	CvPoint     pt6s_buff[NRECTBUFF][COMP_COUNT];
	float		parameters_buff[NRECTBUFF][NPARAMETERS];

};




#endif // _CX_COMPDETECTOR7PT_HPP_