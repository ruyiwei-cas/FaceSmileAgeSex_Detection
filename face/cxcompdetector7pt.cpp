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

#include "cxcompdetector7pt.hpp"
#include "stdio.h"
#include "opencv/highgui.h"

#ifndef max
#define max(a,b)          (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b)          (((a) < (b)) ? (a) : (b))
#endif

//#define NO_PITCH_AND_YAW

//inline float Distance( CvPoint vertex1,CvPoint vertex2 ) 
//{
//	return sqrt((float)(vertex1.x-vertex2.x)*(vertex1.x-vertex2.x)+
//		(float)(vertex1.y-vertex2.y)*(vertex1.y-vertex2.y));
//}
inline float Distance( CvPoint2D32f vertex1,CvPoint2D32f vertex2 ) 
{
	return sqrt((float)(vertex1.x-vertex2.x)*(vertex1.x-vertex2.x)+
		(float)(vertex1.y-vertex2.y)*(vertex1.y-vertex2.y));
}

//inline float  Slope( CvPoint vertex1, CvPoint vertex2 )	
//{
//	return	(float)((vertex2.y-vertex1.y)/((float)(vertex2.x-vertex1.x)));
//}

inline float  Slope( CvPoint2D32f vertex1, CvPoint2D32f vertex2 )	
{
	return	(float)((vertex2.y-vertex1.y)/((float)(vertex2.x-vertex1.x)));
}

inline float sum_arr( float* arr, int n_count )
{
	float sum = 0.0f;
	int i = 0;
	n_count = (n_count < 0)?0:n_count;
	while ( n_count-- )
	{
		sum += arr[i++];
	}
	return sum;
}


const float CxCompDetector7pt::FACTOR_1			= 0;//float(0.3 * REGION_SIZE * REGION_SIZE);
const float CxCompDetector7pt::FACTOR_2			= 0;//float(0.2 * REGION_SIZE * REGION_SIZE);//CxCompDetector7pt::FACTOR_1;//pitch
const float CxCompDetector7pt::FACTOR_3			= 0;//float(0.2 * REGION_SIZE * REGION_SIZE);//CxCompDetector7pt::FACTOR_1;//yaw
const float CxCompDetector7pt::FACTOR_4			= float(0.5 * REGION_SIZE * REGION_SIZE);//roll



/*
const float CxCompDetector7pt::FACTOR_1			= 0;
const float CxCompDetector7pt::FACTOR_2			= 0;//CxCompDetector7pt::FACTOR_1;//pitch
const float CxCompDetector7pt::FACTOR_3			= 0;//CxCompDetector7pt::FACTOR_1;//yaw
const float CxCompDetector7pt::FACTOR_4			= 0;//roll
*/


const float CxCompDetector7pt::AVG_MTH_L			= 0.4920f;
const float CxCompDetector7pt::STDVAR_MTH_L		= 0.0880f;

const float CxCompDetector7pt::AVG_PITCH_ANGLE		= 0.0000f;
const float CxCompDetector7pt::STDVAR_PITCH_ANGLE	= float(CV_PI * 0.25);//0.5236f;
const float CxCompDetector7pt::AVG_YAW_ANGLE		= 0.0000f;
const float CxCompDetector7pt::STDVAR_YAW_ANGLE	= float(CV_PI * 0.25);//
const float CxCompDetector7pt::AVG_ROLL_ANGLE		= 0.0000f;
const float CxCompDetector7pt::STDVAR_ROLL_ANGLE	= float(CV_PI * 0.17);

CxCompDetector7pt::CxCompDetector7pt( const char* xml_eye_leftcorner, 
                                const char* xml_mth_leftcorner,
								const char* xml_nose,
                                const float* std_shape )
{
    // storage
    storage = cvCreateMemStorage( 0 );

    // cascade
    cascade_eye_lc = (CvHaarClassifierCascade*) cvLoad( xml_eye_leftcorner );
	cascade_eye_rc = (CvHaarClassifierCascade*) cvLoad( xml_eye_leftcorner );
    cascade_mth_lc = (CvHaarClassifierCascade*) cvLoad( xml_mth_leftcorner );
	cascade_mth_rc = (CvHaarClassifierCascade*) cvLoad( xml_mth_leftcorner );

	cascade_nose = (CvHaarClassifierCascade*) cvLoad( xml_nose );
    if( !cascade_eye_lc || !cascade_mth_lc )
    {
        OPENCV_ERROR( CV_StsBadArg, 
                      "CxCompDetector7pt::CxCompDetector7pt()", 
                      "Cannot load one or more cascades." );
    }

	cxMirrorHaarClassifierCascade(cascade_eye_rc);
	cxMirrorHaarClassifierCascade(cascade_mth_rc);

    cascade_lut[0] = cascade_eye_lc;
	cascade_lut[1] = cascade_eye_rc;
	cascade_lut[2] = cascade_eye_lc;
	cascade_lut[3] = cascade_eye_rc;
	cascade_lut[4] = cascade_mth_lc;
	cascade_lut[5] = cascade_mth_rc;
	cascade_lut[6] = cascade_nose;

    // definitions on canvas
    sz_canvas        = cvSize( 96, 96 );
    rc_detect        = cvRect( sz_canvas.width/6,    sz_canvas.height/6, 
                              2*sz_canvas.width/3,  2*sz_canvas.height/3 );
	// standard shape data
	float _std_shape[COMP_COUNT*3] = 
	{
		-0.460000f, 0.148000f, -0.111000f,// ×óÑÛÍâÑÛ½Ç
		-0.160000f, 0.148000f, -0.000000f,// ×óÑÛÄÚÑÛ½Ç
		0.160000f, 0.148000f, -0.000000f, // ÓÒÑÛÄÚÑÛ½Ç
		0.460000f, 0.148000f, -0.111000f,// ÓÒÑÛÍâÑÛ½Ç
		-0.246000f, -0.461000f, -0.000000f,// ×ó×ì½Ç
		0.246000f, -0.461000f, -0.000000f,// ÓÒ×ì½Ç
		0.000000f, -0.222000f, 0.210000f,// ±Ç¼â
	};

    if( std_shape == NULL )
        std_shape = _std_shape;

/*
	float std_shape[COMP_COUNT*3] = 
	{
		-0.470000f, 0.3045f, -0.111000f,// ×óÑÛÍâÑÛ½Ç
		-0.130000f, 0.3045f, 0.000000f,// ×óÑÛÄÚÑÛ½Ç
		0.130000f, 0.3045f, 0.000000f, // ÓÒÑÛÄÚÑÛ½Ç
		0.470000f, 0.3045f, -0.111000f,// ÓÒÑÛÍâÑÛ½Ç
		-0.246000f, -0.3045f, -0.000000f,// ×ó×ì½Ç
		0.246000f, -0.3045f, -0.000000f,// ÓÒ×ì½Ç
		0.000000f, -0.0655, 0.210000f,// ±Ç¼â
	};
*/
/*
	float std_shape[COMP_COUNT*3] = 
	{
		-0.470000f, 0.3045f, 0.000000f,// ×óÑÛÍâÑÛ½Ç
		-0.130000f, 0.3045f, 0.111000f,// ×óÑÛÄÚÑÛ½Ç
		0.130000f, 0.3045f, 0.111000f, // ÓÒÑÛÄÚÑÛ½Ç
		0.470000f, 0.3045f, 0.000000f,// ÓÒÑÛÍâÑÛ½Ç
		-0.246000f, -0.3045f, 0.00000f,// ×ó×ì½Ç
		0.246000f, -0.3045f, 0.00000f,// ÓÒ×ì½Ç
		0.000000f, -0.0655, 0.321000f,// ±Ç¼â
	};
*/

/*
	float std_shape[COMP_COUNT*3] = 
	{
		-0.470000f, 0.226857135f, 0.000000f,// ×óÑÛÍâÑÛ½Ç
		-0.130000f, 0.226857135f, 0.111000f,// ×óÑÛÄÚÑÛ½Ç
		0.130000f, 0.226857135f, 0.111000f, // ÓÒÑÛÄÚÑÛ½Ç
		0.470000f, 0.226857135f, 0.000000f,// ÓÒÑÛÍâÑÛ½Ç
		-0.246000f, -0.382142865f, 0.00000f,// ×ó×ì½Ç
		0.246000f, -0.382142865f, 0.00000f,// ÓÒ×ì½Ç
		0.000000f, -0.143142865f, 0.321000f,// ±Ç¼â
	};
*/


	memcpy( fl_stdshape_data, std_shape, sizeof(float)*COMP_COUNT*3 );

    // canvas
    mat_img_c1.create( sz_canvas.height, sz_canvas.width, CV_8UC1 );
    mat_img_c3.create( sz_canvas.height, sz_canvas.width, CV_8UC3 );
    mat_sum   .create( sz_canvas.height+1, sz_canvas.width+1, CV_32SC1 );
    mat_sqsum .create( sz_canvas.height+1, sz_canvas.width+1, CV_64FC1 );
    mat_tilted.create( sz_canvas.height+1, sz_canvas.width+1, CV_32SC1 );

	cvSetImagesForHaarClassifierCascade( cascade_eye_lc, &mat_sum, &mat_sqsum, NULL, 1 );
	cvSetImagesForHaarClassifierCascade( cascade_eye_rc, &mat_sum, &mat_sqsum, NULL, 1 );
	cvSetImagesForHaarClassifierCascade( cascade_mth_lc, &mat_sum, &mat_sqsum, NULL, 1 );
	cvSetImagesForHaarClassifierCascade( cascade_mth_rc, &mat_sum, &mat_sqsum, NULL, 1 );
	cvSetImagesForHaarClassifierCascade( cascade_nose, &mat_sum, &mat_sqsum, &mat_tilted, 1 );

	//init run cascade region for each component
	{
		cascade_search_rect[0] = cvRect( 14, 22, 25, 20);

		cascade_search_rect[1] = cvRect( 26, 26, 23, 15 );

		cascade_search_rect[2] = cvRect( 45, 26, 23, 14 );

		cascade_search_rect[3] = cvRect( 56, 22, 24, 20 );

		cascade_search_rect[4] = cvRect( 22, 52, 25, 24 );

		cascade_search_rect[5] = cvRect( 47, 51, 26, 24 );

		cascade_search_rect[6] = cvRect( 32, 32, 32, 32 );
	}
	mode = MODE_DETECT;


	//distribution 
	mat_eye_lc_dstrb.create( sz_canvas.height, sz_canvas.width, CV_32FC1 );
	mat_eye_rc_dstrb.create( sz_canvas.height, sz_canvas.width, CV_32FC1 );
	mat_mth_lc_dstrb.create( sz_canvas.height, sz_canvas.width, CV_32FC1 );
	mat_mth_rc_dstrb.create( sz_canvas.height, sz_canvas.width, CV_32FC1 );
	mat_nose_dstrb.create( sz_canvas.height, sz_canvas.width, CV_32FC1 );

	mat_dstrb_lut[0] = &mat_eye_lc_dstrb;
	mat_dstrb_lut[1] = &mat_eye_rc_dstrb;
	mat_dstrb_lut[2] = &mat_eye_lc_dstrb;
	mat_dstrb_lut[3] = &mat_eye_rc_dstrb;
	mat_dstrb_lut[4] = &mat_mth_lc_dstrb;
	mat_dstrb_lut[5] = &mat_mth_rc_dstrb;
	mat_dstrb_lut[6] = &mat_nose_dstrb;

	mat_delta_parameters	.create( NPARAMETERS, 1, CV_32FC1 );
	mat_jacobian			.create( REGION_SIZE*REGION_SIZE*COMP_COUNT + 4, NPARAMETERS, CV_32FC1 );
	mat_current_energy		.create( REGION_SIZE*REGION_SIZE*COMP_COUNT + 4, 1, CV_32FC1 );
	mat_current_temp_energy	.create( REGION_SIZE*REGION_SIZE*COMP_COUNT + 4, 1, CV_32FC1 );
	mat_current_parameters	.create( NPARAMETERS, 1, CV_32FC1 );
	mat_new_parameters		.create( NPARAMETERS, 1, CV_32FC1 );

	mat_hessian				.create( NPARAMETERS, NPARAMETERS, CV_32FC1 );
	mat_soln				.create( NPARAMETERS, 1, CV_32FC1 );
	mat_invertH				.create( NPARAMETERS, NPARAMETERS, CV_32FC1 );
	mat_temp_calt_par		.create( NPARAMETERS, 1, CV_32FC1 );
	mat_AT					.create( NPARAMETERS, REGION_SIZE*REGION_SIZE*COMP_COUNT + 4, CV_32FC1 );

	//init tracking face rect buff
	rect_buff_idx = 0;
	for(int i =0; i < NRECTBUFF; i++)
		rect_buff[i] = cvRect(0,0,0,0);


    memset( parameters_buff, 0, sizeof(parameters_buff) );
}

CxCompDetector7pt::~CxCompDetector7pt()
{
    if( storage )
        cvReleaseMemStorage( &storage );
    if( cascade_eye_lc )
        cvReleaseHaarClassifierCascade( &cascade_eye_lc );
    if( cascade_eye_rc )
        cvReleaseHaarClassifierCascade( &cascade_eye_rc );
	if( cascade_mth_lc )
		cvReleaseHaarClassifierCascade( &cascade_mth_lc );
	if( cascade_mth_rc )
		cvReleaseHaarClassifierCascade( &cascade_mth_rc );
}


bool CxCompDetector7pt::track( const IplImage* image, CvRect* rect, 
							CvPoint2D32f points[], float parameters[], float angle )
{
	int   face_idx;
	float ratio;
	bool  rt_found = false;

	for(int i =rect_buff_idx+NRECTBUFF-1; i >=rect_buff_idx+1 ; i--)
	{
		face_idx = i % NRECTBUFF;
		CvRect rect1   = rect_buff[face_idx];
		if(rect1.width < 5) break; //invalid rect

		CvRect rect2   = *rect;
		int l = MAX(rect1.x,rect2.x);
		int r = MIN(rect1.x+rect1.width, rect2.x+rect2.width);
		int w = r -l;
		if(w < 5) continue;

		int t = MAX(rect1.y,rect2.y);
		int b = MIN(rect1.y+rect1.height, rect2.y+rect2.height);
		int h = b -t;
		if(h < 5) continue;
		ratio = 1.0f*w*h/ MAX((rect1.width*rect1.height), (rect2.width*rect2.height));

		if(ratio > 0.80f)
		{
			rt_found = true;
			break;
		}
	}

	bool pt6_found = false;
	if(rt_found)
	{
		//if find, refine m_landmark68Buff[faceidx]
		memcpy(parameters, &parameters_buff[face_idx], sizeof(float)*NPARAMETERS);
		memcpy(points, &pt6s_buff[face_idx], sizeof(CvPoint2D32f)*COMP_COUNT);
		pt6_found = detect( image, NULL, points, parameters, angle );
		
		if(pt6_found == false)
			pt6_found = detect( image, rect, points, parameters, angle );

		if(pt6_found)
		{
			memcpy(&parameters_buff[face_idx], parameters, sizeof(float)*COMP_COUNT);
			memcpy(&pt6s_buff[face_idx], points, sizeof(CvPoint2D32f)*COMP_COUNT);
		}
	}
	else
	{
		pt6_found = detect( image, rect, points, parameters, angle );
		if(pt6_found)
		{
			memcpy(&parameters_buff[face_idx], parameters, sizeof(float)*NPARAMETERS);
			memcpy(&pt6s_buff[rect_buff_idx], points, sizeof(CvPoint2D32f)*COMP_COUNT);
			rect_buff[rect_buff_idx] = *rect;
			rect_buff_idx = (rect_buff_idx+1)%NRECTBUFF;
		}
	}

	return pt6_found;
}

bool CxCompDetector7pt::detect( const IplImage* image, CvRect* rect, 
							CvPoint2D32f points[], float parameters[], float angle )
{
    // get rc_src and rc_dst
	CvRect rc_src;
	CvRect rc_dst = rc_detect;
	bool rt_found = false;
	
	int i;

	if (rect == NULL)//track mode
	{
		mode = MODE_TRACK;
		float L1 = Distance( points[0], points[3] );
		int width_src = int(L1*64/42.475299 + 0.5f);
		int height_src = width_src;
		CvPoint2D32f pt_src_center;
		pt_src_center.x = float(points[0].x + points[3].x + points[4].x + points[5].x)*0.25f;
		pt_src_center.y = float(points[0].y + points[3].y + points[4].y + points[5].y)*0.25f;
		rc_src.x = int(pt_src_center.x - width_src*0.5 + 0.5f);
		rc_src.y = int(pt_src_center.y - height_src*0.5 + 0.5f);
		rc_src.width = width_src;
		rc_src.height = height_src;
	}
	else
	{
		mode = MODE_DETECT;
		rc_src = *rect;
	}
	// map from rc_src to rc_dst
	CvPoint pt_src = cvPoint( rc_src.x + rc_src.width /2, 
		rc_src.y + rc_src.height/2 );
	CvPoint pt_dst = cvPoint( rc_dst.x + rc_dst.width /2, 
		rc_dst.y + rc_dst.height/2 );
	float kx = float( rc_dst.width ) / float( rc_src.width );
	float bx = pt_dst.x - kx * pt_src.x;
	float ky = float( rc_dst.height ) / float( rc_src.height );
	float by = pt_dst.y - ky * pt_src.y;

	// expand rc_src
	float ex = 0.5f * (float(sz_canvas.width )/float(rc_detect.width ) - 1);
	float ey = 0.5f * (float(sz_canvas.height)/float(rc_detect.height) - 1);
	int padx = cvRound( ex * rc_src.width  );
	int pady = cvRound( ey * rc_src.height );
	rc_src.x -= padx;
	rc_src.y -= pady;
	rc_src.width  += 2 * padx;
	rc_src.height += 2 * pady;

	// expend rc_dst
	rc_dst = cvRect( 0, 0, sz_canvas.width, sz_canvas.height );

	// validate rc_src
	int left = MAX( 0, MIN( image->width  - 1, rc_src.x ) );
	int top  = MAX( 0, MIN( image->height - 1, rc_src.y ) );
	int right  = MIN( image->width , rc_src.x + rc_src.width  );
	int bottom = MIN( image->height, rc_src.y + rc_src.height );
	rc_src = cvRect( left, top, MAX(1,right-left), MAX(1,bottom-top) );

	// validate rc_dst
	left = MAX( rc_dst.x, MIN( rc_dst.x + rc_dst.width - 1, 
		cvRound( kx * left + bx )    ) );
	top  = MAX( rc_dst.y, MIN( rc_dst.y + rc_dst.height- 1, 
		cvRound( ky * top  + by )    ) );
	right  = MIN( rc_dst.width,  cvRound( kx * right  + bx ) );
	bottom = MIN( rc_dst.height, cvRound( ky * bottom + by ) );
	rc_dst = cvRect( left, top, MAX(1,right-left), MAX(1,bottom-top) );

	// do next matters on ( mat_img_c1, mat_sum, mat_sqsum, mat_tilted )
	// resize image
	cvSetZero( &mat_img_c1 );
	CvMat hmat_src;
	cvGetSubRect( image, &hmat_src, rc_src );
	CvMat hmat_dst;
	if( image->nChannels == 3 )
		cvGetSubRect( &mat_img_c3, &hmat_dst, rc_dst );
	else
		cvGetSubRect( &mat_img_c1, &hmat_dst, rc_dst );

    if( angle == 0 )
    	cvResize( &hmat_src, &hmat_dst, CV_INTER_LINEAR );
    else
    {
        float data[6];
        CvMat hmat_trans = cvMat( 2, 3, CV_32FC1, data );
        cv2DRotationMatrix( cvPointTo32f(pt_src), -angle, (kx+ky)/2.0f, &hmat_trans ); 
        CV_MAT_ELEM( hmat_trans, float, 0, 2 ) += pt_dst.x - pt_src.x;
        CV_MAT_ELEM( hmat_trans, float, 1, 2 ) += pt_dst.y - pt_src.y;
        cvWarpAffine( image, &hmat_dst, &hmat_trans );
    }

	// convert to gray image
	if( image->nChannels == 3 )
	{
		cvGetSubRect( &mat_img_c3, &hmat_src, rc_dst );
		cvGetSubRect( &mat_img_c1, &hmat_dst, rc_dst );
		cvCvtColor( &hmat_src, &hmat_dst, CV_BGR2GRAY );
	}
	// integral image1
	cvZero( &mat_sum );
	cvZero( &mat_sqsum );
	cvZero( &mat_eye_lc_dstrb );
	cvZero( &mat_eye_rc_dstrb );
	cvZero( &mat_mth_lc_dstrb );
	cvZero( &mat_mth_rc_dstrb );
	cvZero( &mat_nose_dstrb );

	cvIntegral( &mat_img_c1, &mat_sum, &mat_sqsum, &mat_tilted );

	if (rect == NULL) //track
	{
		float dst_parameters[NPARAMETERS];
		float energy = 0.0;
		
		//Modified by Nianzu, 2015-05-29
		CvMat* currentEnergy = &CvMat(mat_current_energy);

		for ( i = 0; i < COMP_COUNT; i++ )
		{
			points[i].x = float(points[i].x  - rc_src.x)*rc_dst.width/rc_src.width + 0.5f;
			points[i].y = float(points[i].y  - rc_src.y)*rc_dst.height/rc_src.height + 0.5f;
			{
				CvHaarClassifierCascade* cascade = cascade_lut[i];
				CvSize size = cascade->orig_window_size;
				int nstage = cascade->count;
				CvPoint tempPt = cvPoint( int(points[i].x - size.width/2), int(points[i].y - size.height/2) );
				int rt_stage = cvRunHaarClassifierCascade( cascade, 
					tempPt, 0 );
				rt_stage = 0 - rt_stage;
				rt_stage = (rt_stage == -1)?nstage:rt_stage;
				energy += float(nstage + 1 - rt_stage)/nstage;
			}
		}
		memcpy( pt_src_comps, points, sizeof(CvPoint2D32f)*COMP_COUNT );
		energy = energy/COMP_COUNT;

		if ( energy > 0.35f )//do newton to update
		{
			parameters[3] = float(parameters[3]  - rc_src.x)*rc_dst.width/rc_src.width + rc_dst.x;
			parameters[4] = float(parameters[4]  - rc_src.y)*rc_dst.height/rc_src.height + rc_dst.y;
			parameters[5] = parameters[5]*rc_dst.width/rc_src.width;//because width == height
			memcpy( dst_parameters, parameters, sizeof(float)*NPARAMETERS );

			Calc7PointsFromParameters( dst_parameters, pt_src_comps );

			rt_found = NewtonIteration( dst_parameters, &energy );
			//rt_found = true;
			Calc7PointsFromParameters( dst_parameters, pt_dst_comps );
			memcpy( parameters, dst_parameters, sizeof(float)*COMP_COUNT );
		}
		else//return derectly
		{
			memcpy( pt_src_comps, points, sizeof(CvPoint2D32f)*COMP_COUNT );
			memcpy( pt_dst_comps, points, sizeof(CvPoint2D32f)*COMP_COUNT );

			rt_found = true;
		}
	}
	else
	{
		float org_multi_parameters[NGROUP][NPARAMETERS];
		float dst_multi_parameters[NGROUP][NPARAMETERS];

		float energy_sum[NGROUP] = {0.0};
		float minimumEnergy = FLT_MAX;
		int minimumEnergyIndex = 0;
		bool found[NGROUP] = {false};
		int j;

		RunHaarClassifier( );
		GenerateMultiOriginalParameters( org_multi_parameters );

		memcpy( dst_multi_parameters, org_multi_parameters, sizeof(org_multi_parameters[0][0])*(NGROUP-2)*NPARAMETERS );
		//memcpy( dst_multi_parameters[4], org_multi_parameters[2], sizeof(org_multi_parameters[0][0])*(NGROUP-2)*NPARAMETERS );
		//memcpy( dst_multi_parameters[5], org_multi_parameters[3], sizeof(org_multi_parameters[0][0])*(NGROUP-2)*NPARAMETERS );
		memcpy( dst_multi_parameters[4], org_multi_parameters[2], sizeof(org_multi_parameters[0][0])*NPARAMETERS );
		memcpy( dst_multi_parameters[5], org_multi_parameters[3], sizeof(org_multi_parameters[0][0])*NPARAMETERS );

#ifdef NO_PITCH_AND_YAW
		for ( j = 0; j < NGROUP - 2; j++ )
#else
		AdjustNose( dst_multi_parameters[4], pt_src_comps[NOSE_TIPS] );
		AdjustNose( dst_multi_parameters[5], pt_src_comps[NOSE_TIPS] );
		for ( j = 0; j < NGROUP; j++ )
#endif
		{
			found[j] = NewtonIteration( dst_multi_parameters[j], &energy_sum[j] );
			if (found[j] == false)
			{
				continue;
			}
			minimumEnergyIndex = minimumEnergy>energy_sum[j]?j:minimumEnergyIndex;
			minimumEnergy = minimumEnergy>energy_sum[j]?energy_sum[j]:minimumEnergy;
		}
		//printf( "energy_sum = %f \n", energy_sum[minimumEnergyIndex] );
		//minimumEnergyIndex = 3;
		rt_found = found[minimumEnergyIndex];//rt_found = true;////
		//Calc7PointsFromParameters( org_multi_parameters[minimumEnergyIndex], pt_src_comps );
		Calc7PointsFromParameters( dst_multi_parameters[minimumEnergyIndex], pt_dst_comps );
        if( parameters != NULL )
		    memcpy( parameters, &dst_multi_parameters[minimumEnergyIndex], sizeof(float)*NPARAMETERS );
/*
		float org_parameters[NPARAMETERS];
		float dst_parameters[NPARAMETERS];
		float energy;
		RunHaarClassifier( );
		GenerateOriginalParameters( org_parameters );
		//Calc7PointsFromParameters( org_parameters, pt_src_comps );
		//org_parameters[2] -= 0.5;
		Calc7PointsFromParameters( org_parameters, pt_dst_comps );
		memcpy( dst_parameters, org_parameters, sizeof(float)*NPARAMETERS );
		//rt_found = NewtonIteration( dst_parameters, &energy );
		rt_found = true;
		Calc7PointsFromParameters( dst_parameters, pt_dst_comps );
		memcpy( parameters, dst_parameters, sizeof(float)*NPARAMETERS );

	}
*/
	}

        float data[6];
        CvMat hmat_trans = cvMat( 2, 3, CV_32FC1, data );
        cv2DRotationMatrix( cvPointTo32f(pt_dst), +angle, 2.0f/(kx+ky), &hmat_trans ); 
        CV_MAT_ELEM( hmat_trans, float, 0, 2 ) -= pt_dst.x - pt_src.x;
        CV_MAT_ELEM( hmat_trans, float, 1, 2 ) -= pt_dst.y - pt_src.y;
    
    //mapping
	for ( i = 0; i < COMP_COUNT; i++ )
	{

        if( angle == 0 )
        {
		    pt_src_comps[i].x = (float((pt_src_comps[i].x - rc_dst.x)*rc_src.width)/rc_dst.width    + rc_src.x + 0.5f);
		    pt_src_comps[i].y = (float((pt_src_comps[i].y - rc_dst.y)*rc_src.height)/rc_dst.height  + rc_src.y + 0.5f);

		    pt_dst_comps[i].x = (float((pt_dst_comps[i].x - rc_dst.x)*rc_src.width)/rc_dst.width    + rc_src.x + 0.5f);
		    pt_dst_comps[i].y = (float((pt_dst_comps[i].y - rc_dst.y)*rc_src.height)/rc_dst.height  + rc_src.y + 0.5f);
        }
        else
        {
            float x, y;

            x = data[0] * pt_src_comps[i].x + data[1] * pt_src_comps[i].y + data[2];
            y = data[3] * pt_src_comps[i].x + data[4] * pt_src_comps[i].y + data[5];
            pt_src_comps[i].x = x;
            pt_src_comps[i].y = y;

            x = data[0] * pt_dst_comps[i].x + data[1] * pt_dst_comps[i].y + data[2];
            y = data[3] * pt_dst_comps[i].x + data[4] * pt_dst_comps[i].y + data[5];
            pt_dst_comps[i].x = x;
            pt_dst_comps[i].y = y;
        }
    
    }
	
    if( parameters != NULL )
    {
	    parameters[3] = float((parameters[3] - rc_dst.x)*rc_src.width)/rc_dst.width  + rc_src.x ;
	    parameters[4] = float((parameters[4] - rc_dst.y)*rc_src.height)/rc_dst.height  + rc_src.y;
	    parameters[5] = parameters[5]*rc_src.width/rc_dst.width;//because width == height
    }

	memcpy( points, pt_dst_comps, sizeof(CvPoint2D32f)*COMP_COUNT );

    return rt_found;
}

void CxCompDetector7pt::cxMirrorHaarClassifierCascade( CvHaarClassifierCascade* cascade )
{
	CvSize win_size = cascade->orig_window_size;

	int i, j, l, k;
	for( i = 0; i < cascade->count; i++ )
	{
		for( j = 0; j < cascade->stage_classifier[i].count; j++ )
		{
			for( l = 0; l < cascade->stage_classifier[i].classifier[j].count; l++ )
			{
				CvHaarFeature* feature =
					&cascade->stage_classifier[i].classifier[j].haar_feature[l];
				for( k = 0; k < CV_HAAR_FEATURE_MAX; k++ )
					if( feature->rect[k].weight != 0 )
					{
						CvRect* pr = &feature->rect[k].r;
						pr->x = win_size.width  - pr->x - pr->width;
					}
			}
		}
	}
}


int CxCompDetector7pt::RunHaarClassifier()
{
	int i;
	int x, y;
	CvMat* sum = 0,  *sqsum = 0;
	float temp_scale = 0.0;

	for ( i = 0; i < COMP_COUNT; i++ )
	{
		CvHaarClassifierCascade* cascade = cascade_lut[i];
		cv::Mat* mat_dstrb = mat_dstrb_lut[i];
		CvRect search_rect = cascade_search_rect[i];
		int nstage = cascade->count;

		CvSize size = cascade->orig_window_size;

		for ( y = search_rect.y ; y < search_rect.height + search_rect.y ; y++ )
		{
			float* mat_data = (float*)(mat_dstrb->row(y).data); //Modified by Nianzu

			for ( x = search_rect.x ; x < search_rect.width + search_rect.x ; x++ )
			{
				if ( x%2 == 0 && y%2 == 0 )
				{
					CvPoint tempPt = cvPoint( x - (size.width/2), y - (size.height/2) );
					int rt_stage = cvRunHaarClassifierCascade( cascade, 
						tempPt, 0 );
					rt_stage = 0 - rt_stage;
					rt_stage = (rt_stage == -1)?nstage:rt_stage;
					*(mat_data + x) = float(rt_stage);
				}
			}
		}

		for ( y = search_rect.y ; y < search_rect.height + search_rect.y ; y++ )
		{
			float* mat_data = (float*)(mat_dstrb->row(y).data);	//Modified by Nianzu
			int step = mat_dstrb->rows;

			for ( x = search_rect.x ; x < search_rect.width + search_rect.x ; x++ )
			{
				if ( x%2 == 1 && y%2 == 0 )
				{
					*(mat_data + x) = float(int((*(mat_data + x -1) + *(mat_data + x + 1))*0.5+0.5));
				}else
					if ( x%2 != 1 && y%2 == 1 )
					{
						*(mat_data + x) = float(int((*(mat_data + x - step) + *(mat_data + x + step))*0.5+0.5));
					}else
						if ( x%2 == 1 && y%2 == 1 )
						{
							*(mat_data + x) = float(int((*(mat_data + x - step-1) + *(mat_data + x - step + 1) + 
								*(mat_data + x + step -1) + *(mat_data + x + step + 1))*0.25+0.5));
						}
			}
		}
	}


/*
	temp_scale = float(1./cascade_eye_lc->count);
	cvSubRS( mat_eye_lc_dstrb, cvScalar( cascade_eye_lc->count + 1 ), mat_eye_lc_dstrb);
	cvScale( mat_eye_lc_dstrb, mat_eye_lc_dstrb, temp_scale );

	cvSubRS( mat_eye_rc_dstrb, cvScalar( cascade_eye_lc->count + 1 ), mat_eye_rc_dstrb);
	cvScale( mat_eye_rc_dstrb, mat_eye_rc_dstrb, temp_scale );

	temp_scale = float(1./cascade_mth_lc->count);
	cvSubRS( mat_mth_lc_dstrb, cvScalar( cascade_mth_lc->count + 1 ), mat_mth_lc_dstrb);
	cvScale( mat_mth_lc_dstrb, mat_mth_lc_dstrb, temp_scale );

	cvSubRS( mat_mth_rc_dstrb, cvScalar( cascade_mth_lc->count + 1 ), mat_mth_rc_dstrb);
	cvScale( mat_mth_rc_dstrb, mat_mth_rc_dstrb, temp_scale );

	temp_scale = float(1./cascade_nose->count);
	cvSubRS( mat_nose_dstrb, cvScalar( cascade_nose->count + 1 ), mat_nose_dstrb);
	cvScale( mat_nose_dstrb, mat_nose_dstrb, temp_scale );
*/


	temp_scale = float(1./cascade_eye_lc->count);
	cvSubRS( &mat_eye_lc_dstrb, cvScalar( cascade_eye_lc->count + 1 ), &mat_eye_lc_dstrb);
	cvScale( &mat_eye_lc_dstrb, &mat_eye_lc_dstrb, temp_scale );

	cvSubRS( &mat_eye_rc_dstrb, cvScalar( cascade_eye_lc->count + 1 ), &mat_eye_rc_dstrb);
	cvScale( &mat_eye_rc_dstrb, &mat_eye_rc_dstrb, temp_scale );

	temp_scale = float(1./cascade_mth_lc->count);
	cvSubRS( &mat_mth_lc_dstrb, cvScalar( cascade_mth_lc->count + 1 ), &mat_mth_lc_dstrb);
	cvScale( &mat_mth_lc_dstrb, &mat_mth_lc_dstrb, temp_scale );

	cvSubRS( &mat_mth_rc_dstrb, cvScalar( cascade_mth_lc->count + 1 ), &mat_mth_rc_dstrb);
	cvScale( &mat_mth_rc_dstrb, &mat_mth_rc_dstrb, temp_scale );

	temp_scale = float(1./cascade_nose->count);
	cvSubRS( &mat_nose_dstrb, cvScalar( cascade_nose->count + 1 ), &mat_nose_dstrb);
	cvScale( &mat_nose_dstrb, &mat_nose_dstrb, temp_scale );

	//cvNamedWindow("test1");
	//cvShowImage("test1", mat_eye_lc_dstrb);
	//cvNamedWindow("test2");
	//cvShowImage("test2", mat_eye_rc_dstrb);
	//cvNamedWindow("test3");
	//cvShowImage("test3", mat_mth_lc_dstrb);
	//cvNamedWindow("test4");
	//cvShowImage("test4", mat_mth_rc_dstrb);
	//cvNamedWindow("test5");
	//cvShowImage("test5", mat_nose_dstrb);


	return 0;

}

//----------------------------------------------------------------------
// Shape = (c00, c01; c10, c11) * Shape
//----------------------------------------------------------------------
void Transform(CvMat &oneShape, CvMat &resultShape, float c00, float c01, float c10, float c11)
{
	float x, y;
	int shapeSize = oneShape.rows*oneShape.cols;
	shapeSize = shapeSize>>1;
	for(int i = 0, size = shapeSize; i < size; i++)
	{
		x = (float) cvmGet(&oneShape,0,i*2);
		y = (float) cvmGet(&oneShape,0,i*2+1);

		cvmSet(&resultShape,0,i*2,c00*x+c01*y);
		cvmSet(&resultShape,0,i*2+1,c10*x+c11*y);
	}

}


//---------------------------------------------------------------------
// Translate Shape using (x, y) 
//----------------------------------------------------------------------
void Translate(CvMat &resultShape, float x, float y)
{
	int shapeSize = resultShape.rows*resultShape.cols;
	float xx,yy;
	shapeSize = shapeSize>>1;

	for(int i = 0, size = shapeSize; i < size; i++)
	{
		xx = (float) cvmGet(&resultShape,0,i*2)+x;
		yy = (float) cvmGet(&resultShape,0,i*2+1)+y;
		cvmSet(&resultShape,0,i*2,xx);
		cvmSet(&resultShape,0,i*2+1,yy);
	}
}



void AlignTransformation(CvMat &refShape, CvMat &oneShape, CvMat &resultShape, float &a_, float &b_, float &tx_, float &ty_)
{

	float X1 = 0, Y1 = 0, X2 = 0, Y2 = 0, Z = 0, C1 = 0, C2 = 0;
	int shapeSize = (refShape.rows*refShape.cols)>>1;

	float W = (float)shapeSize;
	float x1, y1, x2, y2;
	float a, b, tx, ty;

	for(int i = 0, size = shapeSize; i < size; i++)
	{
		x1 = (float) cvmGet(&refShape, 0, i*2);
		y1 = (float) cvmGet(&refShape, 0, i*2+1);
		x2 = (float) cvmGet(&oneShape, 0, i*2);
		y2 = (float) cvmGet(&oneShape, 0, i*2+1);

		Z  += x2 * x2 + y2 * y2;
		X1 += x1;
		Y1 += y1;
		X2 += x2;
		Y2 += y2;
		C1 += x1 * x2 + y1 * y2;
		C2 += y1 * x2 - x1 * y2;
	}

	{
		float SolnA[] = {X2, -Y2, W, 0, Y2, X2, 0, W, Z, 0, X2, Y2, 0, Z, -Y2, X2};
		CvMat A = cvMat(4, 4, CV_32FC1, SolnA);
		float SolnB[] = {X1, Y1, C1, C2};
		CvMat B = cvMat(4, 1, CV_32FC1, SolnB);

		static CvMat* Soln = cvCreateMat(4, 1, CV_32FC1);
		cvSolve(&A, &B, Soln, CV_SVD);

		a	= CV_MAT_ELEM(*Soln, float, 0, 0);  b	= CV_MAT_ELEM(*Soln, float, 1, 0);
		tx	= CV_MAT_ELEM(*Soln, float, 2, 0);	 ty	= CV_MAT_ELEM(*Soln, float, 3, 0);
	}

	// Explained by YAO Wei, 2008.01.29.
	// It is equivalent as follows, but the former method of is more robust.
	/************************************************************************/
	/*		a	=	C1 / Z			b	=	C2 / Z							*/
	/*		tx	=	X1 / W			ty	=	Y1 / W							*/
	/************************************************************************/	

	float norm = a*a + b*b;
	a_ = a / norm, b_ = -b / norm;
	tx_ = (-a*tx - b*ty) / norm, ty_ = (b*tx - a*ty) / norm;

}

void CenterofShape(const CvMat &oneShape, float &cx, float &cy)
{
	cx = 0.0;
	cy = 0.0;
	int size = (oneShape.rows*oneShape.cols)>>1;

	for(int i = 0; i < size; i++)
	{
		cx += (float) cvmGet(&oneShape, 0, i*2);
		cy += (float) cvmGet(&oneShape, 0, i*2+1);
	}
	cx /= size;
	cy /= size;
}



void CxCompDetector7pt::GenerateOriginalParameters(float parameters[NPARAMETERS])
{
	CvPoint minPoints[COMP_COUNT];

	CvMat searchMat;
	const int gauss_size_lut[COMP_COUNT] = {7,7,7,7,17,17,7};
	//const int gauss_size_lut[COMP_COUNT] = {17,17,17,17,17,17,7};
	int i;

	for ( i = 0;i < COMP_COUNT;i++)
	{
		double minValue, maxValue;
		cv::Mat matrix = (*mat_dstrb_lut[i]).clone();
		cvGetSubRect(&matrix, &searchMat, cascade_search_rect[i]);
		cvSmooth(&searchMat, &searchMat, CV_GAUSSIAN, gauss_size_lut[i]);
		cvMinMaxLoc(&searchMat, &minValue, &maxValue, &minPoints[i], NULL);
		minPoints[i].x += cascade_search_rect[i].x;
		minPoints[i].y += cascade_search_rect[i].y;
	}

	float	std_2d_shapedata[COMP_COUNT*2];
	float	dst_2d_shapedata[COMP_COUNT*2];
	float	rt_2d_shapedata[COMP_COUNT*2];

	for ( i = 0; i < COMP_COUNT; i++ )
	{
		std_2d_shapedata[i*2] = fl_stdshape_data[i*3];
		std_2d_shapedata[i*2 + 1] = -fl_stdshape_data[i*3 + 1];

		dst_2d_shapedata[i*2] = float(minPoints[i].x);
		dst_2d_shapedata[i*2 + 1] = float(minPoints[i].y);
		//printf( "(%d,%d)  ", minPoints[i].x, minPoints[i].y );
	}

	CvMat std_2d_Shape		= cvMat( 1, COMP_COUNT*2,CV_32FC1, std_2d_shapedata );
	CvMat dst_2d_Shape		= cvMat( 1, COMP_COUNT*2,CV_32FC1, dst_2d_shapedata );
	CvMat rt_2d_Shape		= cvMat( 1, COMP_COUNT*2,CV_32FC1, rt_2d_shapedata );
	float center_std_shape_x;
	float center_std_shape_y;

	float a			= 0.0f;
	float b			= 0.0f;
	float tx		= 0.0f;
	float ty		= 0.0f;
	float s			= 0.0f;
	float theta		= 0.0f;

	CenterofShape( std_2d_Shape, center_std_shape_x, center_std_shape_y );
	Translate( std_2d_Shape, -center_std_shape_x, -center_std_shape_y );
	AlignTransformation( std_2d_Shape, dst_2d_Shape, rt_2d_Shape, a, b, tx, ty );

	Transform(std_2d_Shape, rt_2d_Shape, a, -b, b, a);
	Translate(rt_2d_Shape,tx,ty);

	s = sqrt(a*a+b*b);
	theta = atan(b/a);

	for ( i = 0; i < COMP_COUNT; i++ )
	{
		minPoints[i].x = (int)rt_2d_shapedata[i*2];
		minPoints[i].y = (int)rt_2d_shapedata[i*2 + 1];
	}
	parameters[0] = 0;
	parameters[1] = 0;
	parameters[2] = theta;//atan(Slope(minPoints[0],minPoints[3]))
	//parameters[3] = float(minPoints[0].x + minPoints[3].x + minPoints[4].x + minPoints[5].x)*0.25f;
	//parameters[4] = float(minPoints[0].y + minPoints[3].y + minPoints[4].y + minPoints[5].y)*0.25f;
	parameters[3] = tx - s*(cos(theta)*center_std_shape_x + sin(theta)*center_std_shape_y);
	parameters[4] = ty - s*(-sin(theta)*center_std_shape_x + cos(theta)*center_std_shape_y);
	parameters[5] = s;
	parameters[6] = 0.492f;

    for( i = 0; i < COMP_COUNT; i ++ )
    {
        pt_src_comps[i].x = (float)minPoints[i].x;
        pt_src_comps[i].y = (float)minPoints[i].y;
    }

	return;
}


void CxCompDetector7pt::GenerateMultiOriginalParameters(float parameters[NGROUP][NPARAMETERS])
{
	const int NPOINT_ORIGINAL = 4;

	CvPoint minPoints[COMP_COUNT];
	CvMat searchMat;
	//const int gauss_size_lut[COMP_COUNT] = {7,7,7,7,17,17,17};
	const int gauss_size_lut[COMP_COUNT] = {17,17,17,17,17,17,17};

	int i;

	for ( i = 0; i < COMP_COUNT; i++ )
	{
		double minValue, maxValue;
		cv::Mat matrix = (*mat_dstrb_lut[i]).clone();
		cvGetSubRect(&matrix, &searchMat, cascade_search_rect[i]);
		cvSmooth(&searchMat, &searchMat, CV_GAUSSIAN, gauss_size_lut[i]);
		cvMinMaxLoc(&searchMat, &minValue, &maxValue, &minPoints[i], NULL);
		minPoints[i].x += cascade_search_rect[i].x;
		minPoints[i].y += cascade_search_rect[i].y;
	}

	float	std_2d_shapedata[NPOINT_ORIGINAL*2];
	float	dst_2d_shapedata[NPOINT_ORIGINAL*2];
	float	rt_2d_shapedata[NPOINT_ORIGINAL*2];

	CvMat std_2d_Shape		= cvMat( 1, NPOINT_ORIGINAL*2,CV_32FC1, std_2d_shapedata );
	CvMat dst_2d_Shape		= cvMat( 1, NPOINT_ORIGINAL*2,CV_32FC1, dst_2d_shapedata );
	CvMat rt_2d_Shape		= cvMat( 1, NPOINT_ORIGINAL*2,CV_32FC1, rt_2d_shapedata );
	float center_std_shape_x;
	float center_std_shape_y;
	//float center_dst_shape_x;
	//float center_dst_shape_y;
	float a			= 0.0f;
	float b			= 0.0f;
	float tx		= 0.0f;
	float ty		= 0.0f;
	float s			= 0.0f;
	float theta		= 0.0f;

	//first group
	std_2d_shapedata[0] =  fl_stdshape_data[LEYE_OUTER*3];
	std_2d_shapedata[1] = -fl_stdshape_data[LEYE_OUTER*3 + 1];
	std_2d_shapedata[2] =  fl_stdshape_data[REYE_INNER*3];
	std_2d_shapedata[3] = -fl_stdshape_data[REYE_INNER*3 + 1];
	std_2d_shapedata[4] =  fl_stdshape_data[MOUTH_RCRN*3];
	std_2d_shapedata[5] = -fl_stdshape_data[MOUTH_RCRN*3 + 1];
	std_2d_shapedata[6] =  fl_stdshape_data[NOSE_TIPS*3];
	std_2d_shapedata[7] = -fl_stdshape_data[NOSE_TIPS*3 + 1];

	dst_2d_shapedata[0] = float(minPoints[LEYE_OUTER].x);
	dst_2d_shapedata[1] = float(minPoints[LEYE_OUTER].y);
	dst_2d_shapedata[2] = float(minPoints[REYE_INNER].x);
	dst_2d_shapedata[3] = float(minPoints[REYE_INNER].y);
	dst_2d_shapedata[4] = float(minPoints[MOUTH_RCRN].x);
	dst_2d_shapedata[5] = float(minPoints[MOUTH_RCRN].y);
	dst_2d_shapedata[6] = float(minPoints[NOSE_TIPS].x);
	dst_2d_shapedata[7] = float(minPoints[NOSE_TIPS].y);

	CenterofShape( std_2d_Shape, center_std_shape_x, center_std_shape_y );
	Translate( std_2d_Shape, -center_std_shape_x, -center_std_shape_y );

	AlignTransformation( std_2d_Shape, dst_2d_Shape, rt_2d_Shape, a, b, tx, ty );

	Transform(std_2d_Shape, rt_2d_Shape, a, -b, b, a);
	Translate(rt_2d_Shape,tx,ty);

	s = sqrt(a*a+b*b);
	theta = atan(b/a);

	parameters[0][0] = 0;
	parameters[0][1] = 0;
	parameters[0][2] = theta;//atan(Slope(minPoints[0],minPoints[3]))
	parameters[0][3] = tx - s*(cos(-theta)*center_std_shape_x + sin(-theta)*center_std_shape_y);
	parameters[0][4] = ty - s*(-sin(-theta)*center_std_shape_x + cos(-theta)*center_std_shape_y);

	parameters[0][5] = s;
	parameters[0][6] = 0.492f;
/*

	float test_pt_data[COMP_COUNT*2];
	float std_test_2d_data[COMP_COUNT*2];

	for ( i = 0; i < COMP_COUNT; i++ )
	{
		std_test_2d_data[i*2] = fl_stdshape_data[i*3];
		std_test_2d_data[i*2 + 1] = -fl_stdshape_data[i*3 + 1];
	}
	IplImage *test_img = cvCreateImage(cvSize(96,96),8,3);
	cvZero(test_img);
	theta = -theta;
	tx = tx - s*(cos(theta)*center_std_shape_x + sin(theta)*center_std_shape_y);
	ty = ty - s*(-sin(theta)*center_std_shape_x + cos(theta)*center_std_shape_y);

	for (i = 0;i < 7;i++)
	{
		test_pt_data[i*2] = std_test_2d_data[i*2]*cos(theta) + sin(theta)*std_test_2d_data[i*2+1];
		test_pt_data[i*2+1] = -std_test_2d_data[i*2]*sin(theta) + cos(theta)*std_test_2d_data[i*2+1];
		test_pt_data[i*2] = test_pt_data[i*2]*s;
		test_pt_data[i*2+1] = test_pt_data[i*2+1]*s;
		test_pt_data[i*2] = tx + test_pt_data[i*2];
		test_pt_data[i*2+1] = ty + test_pt_data[i*2+1];


		cvCircle(test_img,cvPoint(test_pt_data[i*2],test_pt_data[i*2+1]),2,cvScalar(0,0,255));
	}
	for (i = 0;i < 4;i++)
	{
		printf("(%f,%f)--(%f,%f)  \n", rt_2d_shapedata[i*2],rt_2d_shapedata[i*2+1],
			test_pt_data[i*2],test_pt_data[i*2+1]);
		cvCircle(test_img,cvPoint(rt_2d_shapedata[i*2],rt_2d_shapedata[i*2+1]),2,cvScalar(0,255,0));

	}
	cvNamedWindow("test");
	cvShowImage("test",test_img);
*/




	//second group
	std_2d_shapedata[0] =  fl_stdshape_data[LEYE_INNER*3];
	std_2d_shapedata[1] = -fl_stdshape_data[LEYE_INNER*3 + 1];
	std_2d_shapedata[2] =  fl_stdshape_data[REYE_OUTER*3];
	std_2d_shapedata[3] = -fl_stdshape_data[REYE_OUTER*3 + 1];
	std_2d_shapedata[4] =  fl_stdshape_data[MOUTH_LCRN*3];
	std_2d_shapedata[5] = -fl_stdshape_data[MOUTH_LCRN*3 + 1];
	std_2d_shapedata[6] =  fl_stdshape_data[NOSE_TIPS*3];
	std_2d_shapedata[7] = -fl_stdshape_data[NOSE_TIPS*3 + 1];

	dst_2d_shapedata[0] = float(minPoints[LEYE_INNER].x);
	dst_2d_shapedata[1] = float(minPoints[LEYE_INNER].y);
	dst_2d_shapedata[2] = float(minPoints[REYE_OUTER].x);
	dst_2d_shapedata[3] = float(minPoints[REYE_OUTER].y);
	dst_2d_shapedata[4] = float(minPoints[MOUTH_LCRN].x);
	dst_2d_shapedata[5] = float(minPoints[MOUTH_LCRN].y);
	dst_2d_shapedata[6] = float(minPoints[NOSE_TIPS].x);
	dst_2d_shapedata[7] = float(minPoints[NOSE_TIPS].y);

	CenterofShape( std_2d_Shape, center_std_shape_x, center_std_shape_y );
	Translate( std_2d_Shape, -center_std_shape_x, -center_std_shape_y );

	AlignTransformation( std_2d_Shape, dst_2d_Shape, rt_2d_Shape, a, b, tx, ty );

	Transform(std_2d_Shape, rt_2d_Shape, a, -b, b, a);
	Translate(rt_2d_Shape,tx,ty);

	s = sqrt(a*a+b*b);
	theta = atan(b/a);

	parameters[1][0] = 0;
	parameters[1][1] = 0;
	parameters[1][2] = theta;//atan(Slope(minPoints[0],minPoints[3]))
	parameters[1][3] = tx - s*(cos(-theta)*center_std_shape_x + sin(-theta)*center_std_shape_y);
	parameters[1][4] = ty - s*(-sin(-theta)*center_std_shape_x + cos(-theta)*center_std_shape_y);
	parameters[1][5] = s;
	parameters[1][6] = 0.492f;


	//third group

	std_2d_shapedata[0] =  fl_stdshape_data[LEYE_INNER*3];
	std_2d_shapedata[1] = -fl_stdshape_data[LEYE_INNER*3 + 1];
	std_2d_shapedata[2] =  fl_stdshape_data[REYE_INNER*3];
	std_2d_shapedata[3] = -fl_stdshape_data[REYE_INNER*3 + 1];
	std_2d_shapedata[4] =  fl_stdshape_data[MOUTH_LCRN*3];
	std_2d_shapedata[5] = -fl_stdshape_data[MOUTH_LCRN*3 + 1];
	std_2d_shapedata[6] =  fl_stdshape_data[MOUTH_RCRN*3];
	std_2d_shapedata[7] = -fl_stdshape_data[MOUTH_RCRN*3 + 1];


	dst_2d_shapedata[0] = float(minPoints[LEYE_INNER].x);
	dst_2d_shapedata[1] = float(minPoints[LEYE_INNER].y);
	dst_2d_shapedata[2] = float(minPoints[REYE_INNER].x);
	dst_2d_shapedata[3] = float(minPoints[REYE_INNER].y);
	dst_2d_shapedata[4] = float(minPoints[MOUTH_LCRN].x);
	dst_2d_shapedata[5] = float(minPoints[MOUTH_LCRN].y);
	dst_2d_shapedata[6] = float(minPoints[MOUTH_RCRN].x);
	dst_2d_shapedata[7] = float(minPoints[MOUTH_RCRN].y);


	CenterofShape( std_2d_Shape, center_std_shape_x, center_std_shape_y );
	Translate( std_2d_Shape, -center_std_shape_x, -center_std_shape_y );

	AlignTransformation( std_2d_Shape, dst_2d_Shape, rt_2d_Shape, a, b, tx, ty );

	Transform(std_2d_Shape, rt_2d_Shape, a, -b, b, a);
	Translate(rt_2d_Shape,tx,ty);

	s = sqrt(a*a+b*b);
	theta = atan(b/a);

	parameters[2][0] = 0;
	parameters[2][1] = 0;
	parameters[2][2] = theta;//atan(Slope(minPoints[0],minPoints[3]))
	parameters[2][3] = tx - s*(cos(-theta)*center_std_shape_x + sin(-theta)*center_std_shape_y);
	parameters[2][4] = ty - s*(-sin(-theta)*center_std_shape_x + cos(-theta)*center_std_shape_y);
	parameters[2][5] = s;
	parameters[2][6] = 0.492f;


	//fourth group

	std_2d_shapedata[0] =  fl_stdshape_data[LEYE_OUTER*3];
	std_2d_shapedata[1] = -fl_stdshape_data[LEYE_OUTER*3 + 1];
	std_2d_shapedata[2] =  fl_stdshape_data[REYE_OUTER*3];
	std_2d_shapedata[3] = -fl_stdshape_data[REYE_OUTER*3 + 1];
	std_2d_shapedata[4] =  fl_stdshape_data[MOUTH_LCRN*3];
	std_2d_shapedata[5] = -fl_stdshape_data[MOUTH_LCRN*3 + 1];
	std_2d_shapedata[6] =  fl_stdshape_data[MOUTH_RCRN*3];
	std_2d_shapedata[7] = -fl_stdshape_data[MOUTH_RCRN*3 + 1];

	dst_2d_shapedata[0] = float(minPoints[LEYE_OUTER].x);
	dst_2d_shapedata[1] = float(minPoints[LEYE_OUTER].y);
	dst_2d_shapedata[2] = float(minPoints[REYE_OUTER].x);
	dst_2d_shapedata[3] = float(minPoints[REYE_OUTER].y);
	dst_2d_shapedata[4] = float(minPoints[MOUTH_LCRN].x);
	dst_2d_shapedata[5] = float(minPoints[MOUTH_LCRN].y);
	dst_2d_shapedata[6] = float(minPoints[MOUTH_RCRN].x);
	dst_2d_shapedata[7] = float(minPoints[MOUTH_RCRN].y);


	CenterofShape( std_2d_Shape, center_std_shape_x, center_std_shape_y );
	Translate( std_2d_Shape, -center_std_shape_x, -center_std_shape_y );

	AlignTransformation( std_2d_Shape, dst_2d_Shape, rt_2d_Shape, a, b, tx, ty );

	Transform(std_2d_Shape, rt_2d_Shape, a, -b, b, a);
	Translate(rt_2d_Shape,tx,ty);

	s = sqrt(a*a+b*b);
	theta = atan(b/a);

	parameters[3][0] = 0;
	parameters[3][1] = 0;
	parameters[3][2] = theta;//atan(Slope(minPoints[0],minPoints[3]))
	parameters[3][3] = tx - s*(cos(-theta)*center_std_shape_x + sin(-theta)*center_std_shape_y);
	parameters[3][4] = ty - s*(-sin(-theta)*center_std_shape_x + cos(-theta)*center_std_shape_y);
	parameters[3][5] = s;
	parameters[3][6] = 0.492f;
/*
	for ( i = 0; i < NPOINT_ORIGINAL; i++ )
	{
		minPoints[i].x = int(rt_2d_shapedata[i*2] + 0.5);
		minPoints[i].y = int(rt_2d_shapedata[i*2 + 1] + 0.5);
	}
*/

    for( i = 0; i < COMP_COUNT; i ++ )
    {
        pt_src_comps[i].x = (float)minPoints[i].x;
        pt_src_comps[i].y = (float)minPoints[i].y;
    }

	return;
}




void CxCompDetector7pt::CalcParametersFrom6Points( CvPoint2D32f* locations, float* parameters )
{
	CvPoint2D32f pt_eye_center = cvPoint2D32f( (locations[0].x + locations[3].x)/2., 
		(locations[0].y + locations[3].y)/2. );
	CvPoint2D32f pt_mth_center = cvPoint2D32f( (locations[4].x + locations[5].x)/2., 
		(locations[4].y + locations[5].y)/2. );

	parameters[0] = float(pt_eye_center.x);
	parameters[1] = float(pt_eye_center.y);
	parameters[2] = float(pt_mth_center.x);
	parameters[3] = float(pt_mth_center.y);
	parameters[4] = Distance( locations[0], pt_eye_center );
	parameters[5] = Distance( locations[4], pt_mth_center );
}
void CxCompDetector7pt::AdjustNose(float* parameters, CvPoint2D32f pt_nose)
{
	CvPoint2D32f pt_comps[COMP_COUNT];
	float l = fl_stdshape_data[NOSE_TIPS*3 + 2]*parameters[5];

	Calc7PointsFromParameters( parameters, pt_comps );
	parameters[0] = asin(( pt_comps[NOSE_TIPS].y - pt_nose.y )/l)*0.5f;//multi 0.5f for smooth
	parameters[1] = asin(( pt_comps[NOSE_TIPS].x - pt_nose.x )/l)*0.5f;
}

void CxCompDetector7pt::Calc7PointsFromParameters(float* parameters, CvPoint2D32f* locations)
{
	int i;
	float gamma		 = parameters[0];
	float beta		 = parameters[1];
	float alpha		 = parameters[2];
	float tx		 = parameters[3];
	float ty		 = parameters[4];
	float s			 = parameters[5];
	float l			 = parameters[6];


	float data_temp[COMP_COUNT*3];
	float data_result[COMP_COUNT*3];

	CvMat std_shape_t	 = cvMat( COMP_COUNT, 3, CV_32FC1, fl_stdshape_data ); 
	CvMat std_shape	= cvMat( 3, COMP_COUNT, CV_32FC1, data_temp ); 
	CvMat rt_shape	= cvMat( 3, COMP_COUNT, CV_32FC1, data_result ); 

	fl_stdshape_data[12] = -l/2;
	fl_stdshape_data[15] = l/2;

	cvTranspose( &std_shape_t, &std_shape );

	//ang2mat
	float rx[3*3] = 
	{
		1,0,0,
		0,cos(gamma),sin(gamma),
		0,-sin(gamma),cos(gamma)
	};
	float ry[3*3] = 
	{
		cos(beta),0,-sin(beta),
		0,1,0,
		sin(beta),0,cos(beta)
	};
	float rz[3*3] = 
	{
		cos(alpha),sin(alpha),0,
		-sin(alpha),cos(alpha),0,
		0,0,1
	};
	float r[3*3];

	CvMat Rx = cvMat( 3, 3, CV_32FC1, rx );
	CvMat Ry = cvMat( 3, 3, CV_32FC1, ry );
	CvMat Rz = cvMat( 3, 3, CV_32FC1, rz );
	CvMat R = cvMat( 3, 3, CV_32FC1, r );

//	cvmMul( &Ry, &Rz, &R);
	cvMatMulAdd( &Ry, &Rz, 0, &R );
//	cvmMul( &Rx, &R, &R);
    cvMatMulAdd( &Rx, &R, 0, &R );

//	cvmMul( &R, &std_shape, &rt_shape );
    cvMatMulAdd( &R, &std_shape, 0, &rt_shape );
	cvScale( &rt_shape, &rt_shape, s);

	for ( i = 0; i < COMP_COUNT; i++ )
	{
		cvmSet( &rt_shape, 0, i, tx + cvmGet(&rt_shape, 0, i) );
		cvmSet( &rt_shape, 1, i, ty - cvmGet(&rt_shape, 1, i) );
	}

	for ( i = 0; i < COMP_COUNT; i++ )
	{
		locations[i].x = (float)cvmGet(&rt_shape, 0, i);
		locations[i].y = (float)cvmGet(&rt_shape, 1, i);
	}
}

bool CxCompDetector7pt::IsWithinImage(CvPoint2D32f* points)
{
	for (int i = 0;i < COMP_COUNT;i++)
	{
		int range_x = (sz_canvas.width - (REGION_SIZE/2));
		int range_y = (sz_canvas.height - (REGION_SIZE/2));
		if ((int)points[i].x < 0||(int)points[i].x > (range_x - 1)||
			(int)points[i].y < 0||(int)points[i].y > (range_y - 1))
		{
			return false;
		}
	}
	return true;
}


//calculate average energy from components locations
float CxCompDetector7pt::energy_from_7vtx( IplImage* image, const CvPoint2D32f cornersLocations[] )
{
	CvPoint2D32f points[COMP_COUNT];
	memcpy( points, cornersLocations, sizeof(CvPoint2D32f)*COMP_COUNT );
	CvRect rc_src;
	CvRect rc_dst = rc_detect;
	float L1 = Distance( cornersLocations[0], cornersLocations[3] );
	int width_src = int(L1*64/42.475299 + 0.5f);
	int height_src = width_src;
	CvPoint pt_src_center;
	pt_src_center.x = int(float(points[0].x + points[3].x + points[4].x + points[5].x)*0.25 + 0.5f);
	pt_src_center.y = int(float(points[0].y + points[3].y + points[4].y + points[5].y)*0.25 + 0.5f);
	rc_src.x = int(pt_src_center.x - width_src*0.5 + 0.5f);
	rc_src.y = int(pt_src_center.y - height_src*0.5 + 0.5f);
	rc_src.width = width_src;
	rc_src.height = height_src;

	// map from rc_src to rc_dst
	CvPoint pt_src = cvPoint( rc_src.x + rc_src.width /2, 
		rc_src.y + rc_src.height/2 );
	CvPoint pt_dst = cvPoint( rc_dst.x + rc_dst.width /2, 
		rc_dst.y + rc_dst.height/2 );
	float kx = float( rc_dst.width ) / float( rc_src.width );
	float bx = pt_dst.x - kx * pt_src.x;
	float ky = float( rc_dst.height ) / float( rc_src.height );
	float by = pt_dst.y - ky * pt_src.y;

	// expand rc_src
	float ex = 0.5f * (float(sz_canvas.width )/float(rc_detect.width ) - 1);
	float ey = 0.5f * (float(sz_canvas.height)/float(rc_detect.height) - 1);
	int padx = cvRound( ex * rc_src.width  );
	int pady = cvRound( ey * rc_src.height );
	rc_src.x -= padx;
	rc_src.y -= pady;
	rc_src.width  += 2 * padx;
	rc_src.height += 2 * pady;

	// expend rc_dst
	rc_dst = cvRect( 0, 0, sz_canvas.width, sz_canvas.height );

	// validate rc_src
	int left = MAX( 0, MIN( image->width  - 1, rc_src.x ) );
	int top  = MAX( 0, MIN( image->height - 1, rc_src.y ) );
	int right  = MIN( image->width , rc_src.x + rc_src.width  );
	int bottom = MIN( image->height, rc_src.y + rc_src.height );
	rc_src = cvRect( left, top, MAX(1,right-left), MAX(1,bottom-top) );

	// validate rc_dst
	left = MAX( rc_dst.x, MIN( rc_dst.x + rc_dst.width - 1, 
		cvRound( kx * left + bx )    ) );
	top  = MAX( rc_dst.y, MIN( rc_dst.y + rc_dst.height- 1, 
		cvRound( ky * top  + by )    ) );
	right  = MIN( rc_dst.width,  cvRound( kx * right  + bx ) );
	bottom = MIN( rc_dst.height, cvRound( ky * bottom + by ) );
	rc_dst = cvRect( left, top, MAX(1,right-left), MAX(1,bottom-top) );

	cvIntegral( &mat_img_c1, &mat_sum, &mat_sqsum, &mat_tilted );
	int i;
	float avg_energy = 0.0f;
	for ( i = 0; i < COMP_COUNT; i++ )
	{
		points[i].x = (float(points[i].x  - rc_src.x)*rc_dst.width/rc_src.width   + 0.5f);
		points[i].y = (float(points[i].y  - rc_src.y)*rc_dst.height/rc_src.height + 0.5f);
		{
			CvHaarClassifierCascade* cascade = cascade_lut[i];
			CvSize size = cascade->orig_window_size;
			int nstage = cascade->count;
			CvPoint tempPt = cvPoint( int(points[i].x - size.width/2), int(points[i].y - size.height/2) );
			int rt_stage = cvRunHaarClassifierCascade( cascade, 
				tempPt, 0 );
			rt_stage = 0 - rt_stage;
			rt_stage = (rt_stage == -1)?nstage:rt_stage;
			//printf( "%d:%f ", i, float(nstage + 1 - rt_stage)/nstage);
			avg_energy += float(nstage + 1 - rt_stage)/nstage;
		}
	}
	avg_energy = avg_energy/COMP_COUNT;
	//printf("\n");
	for ( i = 0; i < COMP_COUNT; i++ )
	{
		points[i].x = (float((points[i].x - rc_dst.x)*rc_src.width)/rc_dst.width  + rc_src.x + 0.5f);
		points[i].y = (float((points[i].y - rc_dst.y)*rc_src.height)/rc_dst.height+ rc_src.y + 0.5f);
	}
	
	for ( i = 0; i < COMP_COUNT; i++ )
	{
		//cvCircle( image, points[i], 2, CV_RGB(255,255,255), 1, CV_AA );
	}

	return avg_energy;
}

bool CxCompDetector7pt::Energy( float* parameters, CvMat *result )
{
	int i, j, x, y;
	int yEnd, xEnd;
	float resultEnergy = 0;

	int matWidth = sz_canvas.width;
	int matHeight = sz_canvas.height;

	CvPoint2D32f cornersLocations[COMP_COUNT];
	Calc7PointsFromParameters( parameters, cornersLocations);

	for (i = 0;i < COMP_COUNT;i++)
	{
		const CvHaarClassifierCascade* cascade = cascade_lut[i];
		CvSize size = cascade->orig_window_size;

		int region_begin_x = size.width/2 + 1;
		int region_end_x = sz_canvas.width - size.width/2 - 1;
		int region_begin_y = size.height/2 + 1;
		int region_end_y = sz_canvas.height - size.height/2 - 1;

		for ( y  = (int)cornersLocations[i].y - (REGION_SIZE/2), 
			yEnd = (int)cornersLocations[i].y + (REGION_SIZE/2);
			y <= yEnd;y++ )
		{
			if ( y > region_end_y || y < region_begin_y)
			{
				return false;
			}
			for (x   = (int)cornersLocations[i].x - (REGION_SIZE/2),
				xEnd = (int)cornersLocations[i].x + (REGION_SIZE/2);
				x <= xEnd;x++)
			{
				if ( x > region_end_x || x < region_begin_x)
				{
					return false;
				}
			}
		}
	}

	if ( mode == MODE_DETECT )
	{
		for (i = 0, j = 0;i < COMP_COUNT;i++)
		{
			const CvHaarClassifierCascade* cascade = cascade_lut[i];
			cv::Mat* mat_dstrb = mat_dstrb_lut[i];
			CvRect search_rect = cascade_search_rect[i];
			float nstage = float(cascade->count);
			CvSize size = cascade->orig_window_size;
			j = i*REGION_SIZE*REGION_SIZE;

			for ( y  = (int)cornersLocations[i].y - (REGION_SIZE/2), 
				yEnd = (int)cornersLocations[i].y + (REGION_SIZE/2);
				y <= yEnd;y++ )
			{
				float *mat_dstrb_data = (float*)(mat_dstrb->row(y).data);	//Modified by Nianzu
				for (x   = (int)cornersLocations[i].x - (REGION_SIZE/2),
					xEnd = (int)cornersLocations[i].x + (REGION_SIZE/2);
					x <= xEnd;x++)
				{
					cvmSet(result, j, 0, *(mat_dstrb_data + x));
					j++;
				}
			}
		}
	}
	else//track mode
	{
		for (i = 0, j = 0;i < COMP_COUNT;i++)
		{
			CvHaarClassifierCascade* cascade = cascade_lut[i];
			cv::Mat* mat_dstrb = mat_dstrb_lut[i];
			CvRect search_rect = cascade_search_rect[i];
			float nstage = float(cascade->count);
			CvSize size = cascade->orig_window_size;
			int matWidth = sz_canvas.width - size.width/2;
			int matHeight = sz_canvas.height - size.height/2;

			int step = mat_dstrb->rows;

			j = i*REGION_SIZE*REGION_SIZE;
			for ( y  = (int)cornersLocations[i].y - (REGION_SIZE/2), 
				yEnd = (int)cornersLocations[i].y + (REGION_SIZE/2);
				y <= yEnd;y++ )
			{
				float *mat_dstrb_data = (float*)(mat_dstrb->row(y).data);	//Modified by Nianzu

				for (x   = (int)cornersLocations[i].x - (REGION_SIZE/2),
					xEnd = (int)cornersLocations[i].x + (REGION_SIZE/2);
					x <= xEnd;x++)
				{
					float marks = *(mat_dstrb_data + x);
					if (marks == 0.0)
					{
						if ( x%2 == 0 && y%2 == 0 )
						{
							CvPoint tempPt = cvPoint( x - (size.width/2), y - (size.height/2) );
							int rt_stage = cvRunHaarClassifierCascade( cascade, 
								tempPt, 0 );
							rt_stage = 0 - rt_stage;
							rt_stage = (rt_stage == -1)?int(nstage):rt_stage;
							*(mat_dstrb_data + x) = float((nstage + 1 - rt_stage)/nstage);
						}//end of if ( x%2 == 0 && y%2 == 0 )
						else
							if ( x%2 == 1 && y%2 == 0 )
							{
								marks = *(mat_dstrb_data + x + 1);
								if (marks == 0.0)
								{
									int _x = x+1;
									int _y = y;
									CvPoint tempPt = cvPoint( _x - (size.width/2), _y - (size.height/2) );
									int rt_stage = cvRunHaarClassifierCascade( cascade, 
										tempPt, 0 );
									rt_stage = 0 - rt_stage;
									rt_stage = (rt_stage == -1)?int(nstage):rt_stage;
									*(mat_dstrb_data + _x) = float((nstage + 1 - rt_stage)/nstage);

								}
								marks = *(mat_dstrb_data + x - 1);
								if (marks == 0.0)
								{
									int _x = x-1;
									int _y = y;
									CvPoint tempPt = cvPoint( _x - (size.width/2), _y - (size.height/2) );
									int rt_stage = cvRunHaarClassifierCascade( cascade, 
										tempPt, 0 );
									rt_stage = 0 - rt_stage;
									rt_stage = (rt_stage == -1)?int(nstage):rt_stage;
									*(mat_dstrb_data + _x) = float((nstage + 1 - rt_stage)/nstage);
								}
								*(mat_dstrb_data + x) = float((*(mat_dstrb_data + x -1) + *(mat_dstrb_data + x + 1))*0.5+0.5);
							}else//end of if ( x%2 == 1 && y%2 == 0 )
								if ( x%2 == 0 && y%2 == 1 )
								{
									marks = *(mat_dstrb_data + x + step);
									if (marks == 0.0)
									{
										int _x = x;
										int _y = y+1;
										CvPoint tempPt = cvPoint( _x - (size.width/2), _y - (size.height/2) );
										int rt_stage = cvRunHaarClassifierCascade( cascade, 
											tempPt, 0 );
										rt_stage = 0 - rt_stage;
										rt_stage = (rt_stage == -1)?int(nstage):rt_stage;
										*(mat_dstrb_data + _x + step) = float((nstage + 1 - rt_stage)/nstage);
									}

									marks = *(mat_dstrb_data + x -step );
									if (marks == 0.0)
									{
										int _x = x;
										int _y = y-1;
										CvPoint tempPt = cvPoint( _x - (size.width/2), _y - (size.height/2) );
										int rt_stage = cvRunHaarClassifierCascade( cascade, 
											tempPt, 0 );
										rt_stage = 0 - rt_stage;
										rt_stage = (rt_stage == -1)?int(nstage):rt_stage;
										*(mat_dstrb_data + _x - step) = float((nstage + 1 - rt_stage)/nstage);
									}
									*(mat_dstrb_data + x) = float((*(mat_dstrb_data + x - step) + *(mat_dstrb_data + x + step))*0.5+0.5);
								}else//end of if ( x%2 != 1 && y%2 == 1 )
									if ( x%2 == 1 && y%2 == 1 )
									{
										marks = *(mat_dstrb_data + x + 1);
										if (marks == 0.0)
										{
											int _x = x+1;
											int _y = y;
											CvPoint tempPt = cvPoint( _x - (size.width/2), _y - (size.height/2) );
											int rt_stage = cvRunHaarClassifierCascade( cascade, 
												tempPt, 0 );
											rt_stage = 0 - rt_stage;
											rt_stage = (rt_stage == -1)?int(nstage):rt_stage;
											*(mat_dstrb_data + _x) = float((nstage + 1 - rt_stage)/nstage);
										}
										marks = *(mat_dstrb_data + x - 1);
										if (marks == 0.0)
										{
											int _x = x-1;
											int _y = y;
											CvPoint tempPt = cvPoint( _x - (size.width/2), _y - (size.height/2) );
											int rt_stage = cvRunHaarClassifierCascade( cascade, 
												tempPt, 0 );
											rt_stage = 0 - rt_stage;
											rt_stage = (rt_stage == -1)?int(nstage):rt_stage;
											*(mat_dstrb_data + _x) = float((nstage + 1 - rt_stage)/nstage);
										}
										marks = *(mat_dstrb_data + x + step);
										if (marks == 0.0)
										{
											int _x = x;
											int _y = y+1;
											CvPoint tempPt = cvPoint( _x - (size.width/2), _y - (size.height/2) );
											int rt_stage = cvRunHaarClassifierCascade( cascade, 
												tempPt, 0 );
											rt_stage = 0 - rt_stage;
											rt_stage = (rt_stage == -1)?int(nstage):rt_stage;
											*(mat_dstrb_data + _x + step) = float((nstage + 1 - rt_stage)/nstage);
										}
										marks = *(mat_dstrb_data + x - step );
										if (marks == 0.0)
										{
											int _x = x;
											int _y = y-1;
											CvPoint tempPt = cvPoint( _x - (size.width/2), _y - (size.height/2) );
											int rt_stage = cvRunHaarClassifierCascade( cascade, 
												tempPt, 0 );
											rt_stage = 0 - rt_stage;
											rt_stage = (rt_stage == -1)?int(nstage):rt_stage;
											*(mat_dstrb_data + _x - step) = float((nstage + 1 - rt_stage)/nstage);
										}
										*(mat_dstrb_data + x) = float((*(mat_dstrb_data + x - step-1) + *(mat_dstrb_data + x - step + 1) + 
											*(mat_dstrb_data + x + step -1) + *(mat_dstrb_data + x + step + 1))*0.25+0.5);
									}////end of if ( x%2 == 1 && y%2 == 1 )
					}
					cvmSet(result, j, 0, *(mat_dstrb_data + x));
					j++;
				}
			}
		}
	}

	float l = parameters[6];//length of mouth
	float gamma		 = parameters[0];//radian of pitch
	float beta		 = parameters[1];//radian of yaw
	float alpha		 = parameters[2];//radian of roll

	cvmSet( result, j++, 0, FACTOR_1*fabs(l - AVG_MTH_L)/STDVAR_MTH_L );
	cvmSet( result, j++, 0, FACTOR_2*fabs(gamma - AVG_PITCH_ANGLE)/STDVAR_PITCH_ANGLE );
	cvmSet( result, j++, 0, FACTOR_3*fabs(beta - AVG_YAW_ANGLE)/STDVAR_YAW_ANGLE );
	cvmSet( result, j, 0, FACTOR_4*fabs(alpha - AVG_ROLL_ANGLE)/STDVAR_ROLL_ANGLE );


	return true;
};



bool CxCompDetector7pt::CalcJacobian(CvMat* jacobianMat, CvMat* parametersMat)
{
	//const float diff_step_lut[NPARAMETERS] = { 0.3f, 0.12f, 0.12f, 2.f, 2.f, 10.f, 0.1f};
	const float diff_step_lut[NPARAMETERS] = { 0.15f, 0.06f, 0.06f, 1.f, 1.f, 5.f, 0.05f};

	float diffParameters[NPARAMETERS];
	CvSize disrtibuteMatSize = sz_canvas;
	//Modified by Nianzu
	CvMat* currentEnergyMat = &CvMat(mat_current_temp_energy);	

	CvMat diffEnergyMat;
	float* parameters = parametersMat->data.fl;
	float* currentEnergy = currentEnergyMat->data.fl;
	float tempVal = 0.0;

	memcpy( diffParameters, parameters, NPARAMETERS*sizeof( parameters[0] ) );

	if ( false == Energy( parameters, currentEnergyMat ) )
	{
		return false;
	}

#ifdef NO_PITCH_AND_YAW
	cvGetCol( jacobianMat, &diffEnergyMat, 0 );
	cvSetZero( &diffEnergyMat );
	cvGetCol( jacobianMat, &diffEnergyMat, 1 );
	cvSetZero( &diffEnergyMat );
	for ( int i = 2; i < NPARAMETERS; i++ )
#else
	for ( int i = 0; i < NPARAMETERS; i++ )
#endif
	{
		memcpy(diffParameters, parameters, NPARAMETERS*sizeof(parameters[0]));
		diffParameters[i] =diffParameters[i] + diff_step_lut[i];
		cvGetCol( jacobianMat, &diffEnergyMat, i );
		if ( false == Energy( diffParameters, &diffEnergyMat ) )
		{
			return false;
		}
		cvSub( &diffEnergyMat, currentEnergyMat, &diffEnergyMat );

		tempVal = float(1./(diffParameters[i] - parameters[i]));
		cvScale(&diffEnergyMat, &diffEnergyMat, tempVal);
	}
	return true;
}


void CxCompDetector7pt::CalcNewParameters( CvMat *matA, CvMat* fxMat, CvMat *oldMatX, CvMat *newMatX )
{
	cvTranspose( matA, &mat_AT );
	cvMulTransposed( &mat_AT, &mat_hessian, 0 );
//	cvmMul( mat_AT, fxMat, mat_temp_calt_par );
    cvMatMulAdd( &mat_AT, fxMat, 0, &mat_temp_calt_par );
	cvSolve( &mat_hessian, &mat_temp_calt_par, &mat_soln, CV_LU );
	cvSub( oldMatX, &mat_soln, newMatX );
}

bool CxCompDetector7pt::IsFitShape( float* parameters )
{
	return true;
}

bool CxCompDetector7pt::NewtonIteration( float *parameters, float *energySum )
{
	const int MAXITER = 12;

	//Modified by Nianzu
	//Start
	CvMat* currentParametersMat = &CvMat(mat_current_parameters);
	CvMat* newParametersMat = &CvMat(mat_new_parameters);

	CvMat* deltaParametersMat = &CvMat(mat_delta_parameters);
	CvMat* jacobianMat = &CvMat(mat_jacobian);
	CvMat* currentEnergy = &CvMat(mat_current_energy);
	//End

	float* currentParameters = currentParametersMat->data.fl;
	float* newParameters = newParametersMat->data.fl;

	CvPoint2D32f currentPoints[COMP_COUNT];
	CvPoint2D32f newPoints[COMP_COUNT];

	float e0 = 0.0;
	int iter,i;
	bool bl_stop_iter = false;

	float sum_energy_current = 0.0f;
	float sum_energy_new = 0.0f;

	memcpy(currentParameters, parameters, NPARAMETERS*sizeof(parameters[0]));

	Calc7PointsFromParameters(currentParameters, currentPoints);
	if (false == Energy( currentParameters, currentEnergy))
	{
		//fprintf( stderr, "NewtonIteration ERROR: Points out of image!!\n" );
		*energySum = FLT_MAX;
		return false;
	}
	//sum_energy_current = float(cvSum(currentEnergy).val[0]);
	sum_energy_current = sum_arr( currentEnergy->data.fl, REGION_SIZE*REGION_SIZE*COMP_COUNT );

	for (iter = 0;iter < MAXITER;iter++)
	{
		bl_stop_iter = false;

		if (false == CalcJacobian( jacobianMat, currentParametersMat ))
		{
			//fprintf( stderr, "NewtonIteration ERROR: Points out of image!!\n" );
			*energySum = FLT_MAX;
			return false;
		}
		CalcNewParameters(jacobianMat, currentEnergy, currentParametersMat, newParametersMat);
		Calc7PointsFromParameters(newParameters, newPoints);
		if (!IsWithinImage( newPoints ))
		{
			memcpy(parameters, currentParameters, NPARAMETERS*sizeof(parameters[0]));
			//fprintf( stderr, "NewtonIteration ERROR: Points out of image!!\n" );
			*energySum = FLT_MAX;
			return false;
		}
		for ( i = 0; i < COMP_COUNT; i++ )
		{
			if ( abs(currentPoints[i].x - newPoints[i].x) > 1.0f || 
				abs(currentPoints[i].y - newPoints[i].y) > 1.0f )
			{
				bl_stop_iter = true;
				break;
			}
		}
		if ( false == bl_stop_iter )
		{
			memcpy(currentParameters, newParameters, NPARAMETERS*sizeof(parameters[0]));
			break;
		}
		if (false == Energy( newParameters, currentEnergy))
		{
			//fprintf( stderr, "NewtonIteration ERROR: Points out of image!!\n" );
			memcpy(parameters, currentParameters, NPARAMETERS*sizeof(parameters[0]));
			*energySum = FLT_MAX;
			return false;
		}
		//sum_energy_new = float(cvSum(currentEnergy).val[0]);
		sum_energy_new = sum_arr( currentEnergy->data.fl, REGION_SIZE*REGION_SIZE*COMP_COUNT );
		
		if (sum_energy_new > sum_energy_current)
		{
			//memcpy(parameters, currentParameters, NPARAMETERS*sizeof(parameters[0]));
			//memcpy(currentParameters, newParameters, NPARAMETERS*sizeof(currentParameters[0]));
			//printf("wrong iteration !\n");
			break;
		}
		sum_energy_current = sum_energy_new;
		memcpy(currentParameters, newParameters, NPARAMETERS*sizeof(currentParameters[0]));
		memcpy(currentPoints, newPoints, COMP_COUNT*sizeof(currentPoints[0]));
	}
	memcpy(parameters, currentParameters, NPARAMETERS*sizeof(parameters[0]));
	//printf("iter = %d \n", iter);

	int ncount = 0;
	*energySum = 0;
	for (i = REGION_SIZE*REGION_SIZE/2;i < COMP_COUNT*REGION_SIZE*REGION_SIZE;i += REGION_SIZE*REGION_SIZE)
	{
		ncount++;
		*energySum += float(cvmGet(currentEnergy, i, 0));
	}
	*energySum = *energySum/ncount;
	//printf( "NewtonIteration energy = %f    !\n  ",  *energySum);
	if (iter == MAXITER)
	{
		*energySum += 0.2f;
	}

	if (*energySum > 0.48f)
	{
		//fprintf( stderr, "failed::energy = %f   surpass !\n  ",  *energySum);
		//*energySum = FLT_MAX;
		return false;
	}

	//printf("iter = %d , energy = %f \n", iter,*energySum);
	return true;
}




// calculate average energy from components locations
float CxCompDetector7pt::comp_energy( const IplImage* image, const CvPoint2D32f points[] )
{
    CvPoint pnt;
	pnt.x  = (int)( points[0].x + points[3].x + points[4].x + points[5].x ) / 4;
	pnt.y  = (int)( points[0].y + points[3].y + points[4].y + points[5].y ) / 4;
    int dx = (int)( points[0].x + points[1].x ) / 2 - (int)( points[2].x + points[3].x ) / 2;
    int dy = (int)( points[0].y + points[1].y ) / 2 - (int)( points[2].y + points[3].y ) / 2;
    int len= cvRound( 2.0 * sqrt( 1.0*dx*dx + 1.0*dy*dy ) );
    CvRect rc_src = cvRect( pnt.x-len/2, pnt.y-len/2, len, len );
	CvRect rc_dst = rc_detect;

	// map from rc_src to rc_dst
	CvPoint pt_src = cvPoint( rc_src.x + rc_src.width /2, 
		                      rc_src.y + rc_src.height/2 );
	CvPoint pt_dst = cvPoint( rc_dst.x + rc_dst.width /2, 
		                      rc_dst.y + rc_dst.height/2 );
	float kx = float( rc_dst.width ) / float( rc_src.width );
	float bx = pt_dst.x - kx * pt_src.x;
	float ky = float( rc_dst.height ) / float( rc_src.height );
	float by = pt_dst.y - ky * pt_src.y;

	// expand rc_src
	float ex = 0.5f * (float(sz_canvas.width )/float(rc_detect.width ) - 1);
	float ey = 0.5f * (float(sz_canvas.height)/float(rc_detect.height) - 1);
	int padx = cvRound( ex * rc_src.width  );
	int pady = cvRound( ey * rc_src.height );
	rc_src.x -= padx;
	rc_src.y -= pady;
	rc_src.width  += 2 * padx;
	rc_src.height += 2 * pady;

	// expend rc_dst
	rc_dst = cvRect( 0, 0, sz_canvas.width, sz_canvas.height );

	// validate rc_src
	int left = MAX( 0, MIN( image->width  - 1, rc_src.x ) );
	int top  = MAX( 0, MIN( image->height - 1, rc_src.y ) );
	int right  = MIN( image->width , rc_src.x + rc_src.width  );
	int bottom = MIN( image->height, rc_src.y + rc_src.height );
	rc_src = cvRect( left, top, MAX(1,right-left), MAX(1,bottom-top) );

	// validate rc_dst
	left = MAX( rc_dst.x, MIN( rc_dst.x + rc_dst.width - 1, 
                               cvRound( kx * left + bx )    ) );
	top  = MAX( rc_dst.y, MIN( rc_dst.y + rc_dst.height- 1, 
		                       cvRound( ky * top  + by )    ) );
	right  = MIN( rc_dst.width,  cvRound( kx * right  + bx ) );
	bottom = MIN( rc_dst.height, cvRound( ky * bottom + by ) );
	rc_dst = cvRect( left, top, MAX(1,right-left), MAX(1,bottom-top) );

    // resize image
    CvMat hmat_src;
    cvGetSubRect( image, &hmat_src, rc_src );
    CvMat hmat_dst;
    if( image->nChannels == 3 )
        cvGetSubRect( &mat_img_c3, &hmat_dst, rc_dst );
    else
        cvGetSubRect( &mat_img_c1, &hmat_dst, rc_dst );
    cvResize( &hmat_src, &hmat_dst, CV_INTER_LINEAR );

    // convert to gray image
    if( image->nChannels == 3 )
    {
        cvGetSubRect( &mat_img_c3, &hmat_src, rc_dst );
        cvGetSubRect( &mat_img_c1, &hmat_dst, rc_dst );
        cvCvtColor( &hmat_src, &hmat_dst, CV_BGR2GRAY );
    }

    // integral image
	cvIntegral( &mat_img_c1, &mat_sum, &mat_sqsum, &mat_tilted );

	int i;
	float sum = 0.0f;
	for ( i = 0; i < COMP_COUNT; i++ )
	{
		CvSize size = cascade_lut[i]->orig_window_size;
		int x = cvRound(kx * points[i].x + bx) - size.width/2;
		int y = cvRound(ky * points[i].y + by) - size.height/2;
        int ret = cvRunHaarClassifierCascade( cascade_lut[i], cvPoint(x,y), 0 );
        if( ret <= 0 )
        {
            sum += 1.0f + float(ret) / cascade_lut[i]->count;
//            printf( "%4.3f  ", 1.0f + float(ret) / cascade_lut[i]->count);
        }
//        else
//            printf( "%4.3f  ", 0 );
	}
//    printf("\n");

    return sum / COMP_COUNT;
}

