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

#include "cxcompdetector.hpp"
#include "stdio.h"
#include "opencv/highgui.h"

#ifndef max
#define max(a,b)          (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b)          (((a) < (b)) ? (a) : (b))
#endif

#ifdef _OPENMP
#include <omp.h>
#endif //_OPENMP

inline float Distance( CvPoint vertex1,CvPoint vertex2 ) 
{ 
	return sqrt((float)(vertex1.x-vertex2.x)*(vertex1.x-vertex2.x)+
		(float)(vertex1.y-vertex2.y)*(vertex1.y-vertex2.y));
}
inline float Distance( CvPoint2D32f vertex1,CvPoint2D32f vertex2 ) 
{ 
	return sqrt((float)(vertex1.x-vertex2.x)*(vertex1.x-vertex2.x)+
		(float)(vertex1.y-vertex2.y)*(vertex1.y-vertex2.y));
}

inline float  Slope( CvPoint vertex1, CvPoint vertex2 )	
{ 
	return	(float)((vertex2.y-vertex1.y)/((float)(vertex2.x-vertex1.x)));
}

inline float  Slope( CvPoint2D32f vertex1, CvPoint2D32f vertex2 )	
{ 
	return	(float)((vertex2.y-vertex1.y)/((float)(vertex2.x-vertex1.x)));
}

inline float cxmGetSubPixel32F( const CvMat *mat, CvPoint2D32f pt )
{
	int type;
	type = CV_MAT_TYPE(mat->type);
	assert( (unsigned)pt.x < (unsigned)mat->rows &&
		(unsigned)pt.y < (unsigned)mat->cols && 
		type == CV_32FC1 ) ;

	int nx	= int(pt.x);
	int nx1 = nx + 1;
	int ny	= int(pt.y);
	int ny1 = ny + 1;

	float alfa = pt.x - nx;
	float beta = pt.y - ny;
	const float TH1 = (float)1e-4;
	const float TH2 = 1.0f - TH1;

	float fxy	= CV_MAT_ELEM(*mat, float, ny, nx);
	float fx1y	= CV_MAT_ELEM(*mat, float, ny, nx1);
	float fxy1	= CV_MAT_ELEM(*mat, float, ny1, nx);
	float fx1y1 = CV_MAT_ELEM(*mat, float, ny1, nx1);

	if( alfa < TH1 )
		return float(beta*fxy1 + (1-beta)*fxy);

	if( alfa > TH2 )
		return float(beta*fx1y1 + (1-beta)*fx1y);

	if( beta < TH1 ) 
		return float(alfa*fx1y + (1-alfa)*fxy);	

	if( beta > TH2 )
		return float(alfa*fx1y1 + (1-alfa)*fxy1);

	return float( beta*(alfa * fx1y1 + (1-alfa)*fxy1) + (1-beta)*(alfa*fx1y + (1-alfa)*fxy) );
}

//bilinear interpolation. alfa = nx1 - pt.x; (nx1 = int(pt.x) + 1;) the same as beta
inline float cxBilinear( float fxy, float fx1y, float fxy1, float fx1y1, float alfa, float beta )
{
	const float TH1 = (float)1e-4;
	const float TH2 = 1.0f - TH1;
	if( alfa < TH1 ) 
		return float(beta * fxy1 + (1-beta) * fxy);

	if( alfa > TH2 )
		return float(beta * fx1y1 + (1-beta) * fx1y);

	if( beta < TH1 ) 
		return float(alfa * fx1y + (1-alfa) * fxy);	

	if( beta > TH2 )
		return float(alfa * fx1y1 + (1-alfa) * fxy1);

	return float( beta*(alfa * fx1y1 + (1-alfa)*fxy1) + (1-beta)*(alfa*fx1y + (1-alfa)*fxy) );
}


const float CxCompDetector::FACTOR_1		= float(0.3 * REGION_SIZE * REGION_SIZE);
const float CxCompDetector::FACTOR_2		= FACTOR_1;
const float CxCompDetector::FACTOR_3		= 0.0f;
const float CxCompDetector::AVG_L1			= 42.475299f;	//average(L1)
const float CxCompDetector::STDVAR_L1		= 4.7354597f;	//Standard deviation(L1)
const float CxCompDetector::AVG_L2_L1		= 0.770145f;	//average(L2/L1)
const float CxCompDetector::AVG_L3_L1		= 0.588204f;	//average(L3/L1)
const float CxCompDetector::STDVAR_L2_L1	= 0.087113f;	//Standard deviation(L2/L1)
const float CxCompDetector::STDVAR_L3_L1	= 0.090698f;	//Standard deviation(L3/L1)

CxCompDetector::CxCompDetector( const char* xml_eye_leftcorner, 
                                const char* xml_mth_leftcorner )
{

    // storage
    storage = cvCreateMemStorage( 0 );

    // cascade
	// cascade_eye_lc = (CvHaarClassifierCascade*) cvLoad( xml_eye_leftcorner );
	// cascade_eye_rc = (CvHaarClassifierCascade*) cvLoad( xml_eye_leftcorner );
	// cascade_mth_lc = (CvHaarClassifierCascade*) cvLoad( xml_mth_leftcorner );
	// cascade_mth_rc = (CvHaarClassifierCascade*) cvLoad( xml_mth_leftcorner );


	cascade_eye_lc = (CvHaarClassifierCascade*)cvLoadHaarClassifierCascade(xml_eye_leftcorner, cvSize(16, 16));
	cascade_eye_rc = (CvHaarClassifierCascade*)cvLoadHaarClassifierCascade(xml_eye_leftcorner, cvSize(16, 16));
	cascade_mth_lc = (CvHaarClassifierCascade*)cvLoadHaarClassifierCascade(xml_mth_leftcorner, cvSize(20, 20));
	cascade_mth_rc = (CvHaarClassifierCascade*)cvLoadHaarClassifierCascade(xml_mth_leftcorner, cvSize(20, 20));

    if( !cascade_eye_lc || !cascade_mth_lc )
    {
        OPENCV_ERROR( CV_StsBadArg, 
                      "CxCompDetector::CxCompDetector()", 
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

    // definitions on canvas
    sz_canvas        = cvSize( 96, 96 );
    rc_detect        = cvRect( sz_canvas.width/6,    sz_canvas.height/6, 
                              2*sz_canvas.width/3,  2*sz_canvas.height/3 );

    // canvas
    mat_img_c1.create( sz_canvas.height, sz_canvas.width, CV_8UC1 );
    mat_img_c3.create( sz_canvas.height, sz_canvas.width, CV_8UC3 );
    mat_sum   .create( sz_canvas.height+1, sz_canvas.width+1, CV_32SC1 );
    mat_sqsum .create( sz_canvas.height+1, sz_canvas.width+1, CV_64FC1 );
    mat_tilted.create( sz_canvas.height+1, sz_canvas.width+1, CV_32SC1 );

	cvSetImagesForHaarClassifierCascade( cascade_eye_lc, &(CvMat)mat_sum, &(CvMat)mat_sqsum, NULL, 1);	//Modified by Nianzu
	cvSetImagesForHaarClassifierCascade( cascade_eye_rc, &(CvMat)mat_sum, &(CvMat)mat_sqsum, NULL, 1 );	//Modified by Nianzu
	cvSetImagesForHaarClassifierCascade( cascade_mth_lc, &(CvMat)mat_sum, &(CvMat)mat_sqsum, NULL, 1);	//Modified by Nianzu
	cvSetImagesForHaarClassifierCascade( cascade_mth_rc, &(CvMat)mat_sum, &(CvMat)mat_sqsum, NULL, 1);	//Modified by Nianzu

	//init searching region of generating parameters for each component
	{
		rc_comps[0] = cvRect(sz_canvas.width/8, sz_canvas.height/8, 
			(sz_canvas.width/2)-(sz_canvas.width/4), (sz_canvas.height/2)-(sz_canvas.height/8));

		rc_comps[1] = cvRect(sz_canvas.width/4, sz_canvas.height/8, 
			(sz_canvas.width/2)-(sz_canvas.width/4), (sz_canvas.height/2)-(sz_canvas.height/8));

		rc_comps[2] = cvRect((sz_canvas.width/2), sz_canvas.height/8, 
			(sz_canvas.width/2)-(sz_canvas.width/4), (sz_canvas.height/2)-(sz_canvas.height/8));

		rc_comps[3] = cvRect((sz_canvas.width/2) + (sz_canvas.width/4) - (sz_canvas.width/8), sz_canvas.height/8, 
			(sz_canvas.width/2)-(sz_canvas.width/4), (sz_canvas.height/2)-(sz_canvas.height/8));

		rc_comps[4] = cvRect(sz_canvas.width/8, (sz_canvas.height/2), 
			(sz_canvas.width/2)-(sz_canvas.width/8), (sz_canvas.height/2));

		rc_comps[5] = cvRect((sz_canvas.width/2), (sz_canvas.height/2), 
			(sz_canvas.width/2)-(sz_canvas.width/8), (sz_canvas.height/2));
	}
	//init run cascade region for each component

	{
		cascade_search_rect[0] = cvRect( 14, 22, 25, 20);

		cascade_search_rect[1] = cvRect( 26, 26, 23, 15 );

		cascade_search_rect[2] = cvRect( 45, 26, 23, 14 );

		cascade_search_rect[3] = cvRect( 56, 22, 24, 20 );

		cascade_search_rect[4] = cvRect( 22, 52, 25, 24 );

		cascade_search_rect[5] = cvRect( 47, 51, 26, 24 );
	}
	mode = MODE_DETECT;

	//distribution 
	mat_eye_lc_dstrb.create( sz_canvas.height, sz_canvas.width, CV_32FC1 );
	mat_eye_rc_dstrb.create( sz_canvas.height, sz_canvas.width, CV_32FC1 );
	mat_mth_lc_dstrb.create( sz_canvas.height, sz_canvas.width, CV_32FC1 );
	mat_mth_rc_dstrb.create( sz_canvas.height, sz_canvas.width, CV_32FC1 );

	mat_dstrb_lut[0] = &mat_eye_lc_dstrb;
	mat_dstrb_lut[1] = &mat_eye_rc_dstrb;
	mat_dstrb_lut[2] = &mat_eye_lc_dstrb;
	mat_dstrb_lut[3] = &mat_eye_rc_dstrb;
	mat_dstrb_lut[4] = &mat_mth_lc_dstrb;
	mat_dstrb_lut[5] = &mat_mth_rc_dstrb;

	mat_delta_parameters	.create( NPARAMETERS, 1, CV_32FC1 );
	mat_jacobian			.create( REGION_SIZE*REGION_SIZE*COMP_COUNT+3, NPARAMETERS, CV_32FC1 );
	mat_current_energy		.create( REGION_SIZE*REGION_SIZE*COMP_COUNT+3, 1, CV_32FC1 );
	mat_current_temp_energy	.create( REGION_SIZE*REGION_SIZE*COMP_COUNT+3, 1, CV_32FC1 );
	mat_current_parameters	.create( NPARAMETERS, 1, CV_32FC1 );
	mat_new_parameters		.create( NPARAMETERS, 1, CV_32FC1 );

	mat_hessian				.create( NPARAMETERS, NPARAMETERS, CV_32FC1 );
	mat_soln				.create( NPARAMETERS, 1, CV_32FC1 );
	mat_invertH				.create( NPARAMETERS, NPARAMETERS, CV_32FC1 );
	mat_temp_calt_par		.create( NPARAMETERS, 1, CV_32FC1 );
	mat_AT					.create( NPARAMETERS, REGION_SIZE*REGION_SIZE*COMP_COUNT+3, CV_32FC1 );

	//init tracking face rect buff
	rect_buff_idx = 0;
	for(int i =0; i < NRECTBUFF; i++)
		rect_buff[i] = cvRect(0,0,0,0);
}

CxCompDetector::~CxCompDetector()
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

bool CxCompDetector::track( const IplImage* image, CvRect* rect, 
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
		memcpy(pt_src_comps, &fpt6s_buff[face_idx], sizeof(CvPoint2D32f)*COMP_COUNT);

		pt6_found = detect(image, NULL, points, NULL, angle);

		if(pt6_found == false)
			pt6_found = detect(image, rect, points, NULL, angle);

		if(pt6_found)
		{
			memcpy(&fpt6s_buff[face_idx], pt_dst_comps, sizeof(CvPoint2D32f)*COMP_COUNT);
		}
	}
	else
	{
		pt6_found = detect(image, rect, points, NULL, angle);
		if(pt6_found)
		{
			memcpy(&fpt6s_buff[rect_buff_idx], pt_dst_comps, sizeof(CvPoint)*COMP_COUNT);
			rect_buff[rect_buff_idx] = *rect;
			rect_buff_idx = (rect_buff_idx+1)%NRECTBUFF;
		}
	}

	return pt6_found;
}


bool CxCompDetector::detect( const IplImage* image, CvRect* rect, 
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
		int width_src = int(L1*64/42.475299);
		int height_src = width_src;
		CvPoint pt_src_center;
		pt_src_center.x = int(points[0].x + points[3].x + points[4].x + points[5].x)/4;
		pt_src_center.y = int(points[0].y + points[3].y + points[4].y + points[5].y)/4;
		rc_src.x = pt_src_center.x - width_src/2;
		rc_src.y = pt_src_center.y - height_src/2;
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
	cvSetZero( &(CvMat)mat_img_c1 );
	CvMat hmat_src;
	cvGetSubRect( image, &hmat_src, rc_src );
	CvMat hmat_dst;
	if( image->nChannels == 3 )
		cvGetSubRect( &(CvMat)mat_img_c3, &hmat_dst, rc_dst );
	else
		cvGetSubRect( &(CvMat)mat_img_c1, &hmat_dst, rc_dst );

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
		cvGetSubRect( &(CvMat)mat_img_c3, &hmat_src, rc_dst);
		cvGetSubRect( &(CvMat)mat_img_c1, &hmat_dst, rc_dst );
		cvCvtColor( &hmat_src, &hmat_dst, CV_BGR2GRAY );
	}
	// integral image1
	cvZero( &(CvMat)mat_sum );
	cvZero( &(CvMat)mat_sqsum );
	cvZero( &(CvMat)mat_eye_lc_dstrb );
	cvZero( &(CvMat)mat_eye_rc_dstrb );
	cvZero( &(CvMat)mat_mth_lc_dstrb );
	cvZero( &(CvMat)mat_mth_rc_dstrb );
	cvIntegral( &(CvMat)mat_img_c1, &(CvMat)mat_sum, &(CvMat)mat_sqsum, NULL );

	if (rect == NULL) //track
	{
		float energy = 0.0;
		//printf("track=================\n");
		for ( i = 0; i < COMP_COUNT; i++ )
		{
			pt_src_comps[i].x = float(pt_src_comps[i].x  - rc_src.x)*rc_dst.width/rc_src.width + 0.5f;
			pt_src_comps[i].y = float(pt_src_comps[i].y  - rc_src.y)*rc_dst.height/rc_src.height + 0.5f;
		}
		CalcParametersFrom6Points(pt_src_comps, para_dst);
		rt_found = NewtonIteration( para_dst, &energy );
		Calc6PointsFromParameters( para_dst, pt_dst_comps );		
	}
	else
	{
		float org_multi_parameters[NGROUP][NPARAMETERS];
		float dst_multi_parameters[NGROUP][NPARAMETERS];
		//printf("detect!!!!!!!!!!!!\n");
		RunHaarClassifier( );
		GenerateMultiOriginalParameters( org_multi_parameters );
		memcpy( dst_multi_parameters, org_multi_parameters, sizeof(org_multi_parameters[0][0])*NGROUP*NPARAMETERS );
		float energy_sum[NGROUP] = {0.0};
		float minimumEnergy = FLT_MAX;
		int minimumEnergyIndex = 0;
		bool found[NGROUP] = {false};
		int j;
		for ( j = 0; j < NGROUP; j++ )
		{
			found[j] = NewtonIteration( dst_multi_parameters[j], &energy_sum[j] );
			if (found[j] == false)
			{
				continue;
			}
			minimumEnergyIndex = minimumEnergy>energy_sum[j]?j:minimumEnergyIndex;
			minimumEnergy = minimumEnergy>energy_sum[j]?energy_sum[j]:minimumEnergy;
		}
		rt_found = found[minimumEnergyIndex];
		Calc6PointsFromParameters( org_multi_parameters[minimumEnergyIndex], pt_src_comps );
		memcpy(para_dst, dst_multi_parameters[minimumEnergyIndex], sizeof(float)*NPARAMETERS);
		Calc6PointsFromParameters( dst_multi_parameters[minimumEnergyIndex], pt_dst_comps );
	}

        float data[6];
        CvMat hmat_trans = cvMat( 2, 3, CV_32FC1, data );
        cv2DRotationMatrix( cvPointTo32f(pt_dst), +angle, 2.0f/(kx+ky), &hmat_trans ); 
        CV_MAT_ELEM( hmat_trans, float, 0, 2 ) -= pt_dst.x - pt_src.x;
        CV_MAT_ELEM( hmat_trans, float, 1, 2 ) -= pt_dst.y - pt_src.y;

    for ( i = 0;i < COMP_COUNT;i++ )
	{
        if( angle == 0 )
        {
    		pt_dst_comps[i].x = (float((pt_dst_comps[i].x - rc_dst.x)*rc_src.width)/rc_dst.width  + rc_src.x);
	    	pt_dst_comps[i].y = (float((pt_dst_comps[i].y - rc_dst.y)*rc_src.height)/rc_dst.height  + rc_src.y);
        }
        else
        {
            float x = data[0] * pt_dst_comps[i].x + data[1] * pt_dst_comps[i].y + data[2];
            float y = data[3] * pt_dst_comps[i].x + data[4] * pt_dst_comps[i].y + data[5];
            pt_dst_comps[i].x = x;
            pt_dst_comps[i].y = y;
        }
		
		//points[i] = cvPointFrom32f(pt_dst_comps[i]);
		points[i]  = pt_dst_comps[i];
	}

	memcpy(pt_src_comps, pt_dst_comps, sizeof(CvPoint2D32f)*COMP_COUNT);
	return rt_found;
}

void CxCompDetector::cxMirrorHaarClassifierCascade( CvHaarClassifierCascade* cascade )
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


int CxCompDetector::RunHaarClassifier()
{
	int i;
	int x, y;
	CvMat* sum = 0,  *sqsum = 0;
	float temp_scale = 0.0;

#ifdef _OPENMP
#pragma omp parallel for private(x,y)
#endif
	for ( i = 0; i < COMP_COUNT; i++ )
	{
		CvHaarClassifierCascade* cascade = cascade_lut[i];
		cv::Mat* mat_dstrb = mat_dstrb_lut[i];
		CvRect search_rect = cascade_search_rect[i];
		int nstage = cascade->count;

		CvSize size = cascade->orig_window_size;

		for ( y = search_rect.y ; y < search_rect.height + search_rect.y ; y++ )
		{
			float* mat_data = (float*)(mat_dstrb->row(y).data); // Modified by Nianzu

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
			float* mat_data = (float*)(mat_dstrb->row(y).data);	// Modified by Nianzu
			int step = mat_dstrb->rows;

			for ( x = search_rect.x ; x < search_rect.width + search_rect.x ; x++ )
			{
				if ( x%2 == 1 && y%2 == 0 )
				{
					*(mat_data + x) = float(int((*(mat_data + x -1) + *(mat_data + x + 1))*0.5+0.5));
				}
				else if ( x%2 != 1 && y%2 == 1 )
				{
					*(mat_data + x) = float(int((*(mat_data + x - step) + *(mat_data + x + step))*0.5+0.5));
					
				}
				else if ( x%2 == 1 && y%2 == 1 )
				{
					*(mat_data + x) = float(int((*(mat_data + x - step-1) + *(mat_data + x - step + 1) + 
						*(mat_data + x + step -1) + *(mat_data + x + step + 1))*0.25+0.5));
				}
			}
		}
	}

	temp_scale = float(1./cascade_eye_lc->count);
	//Modified by Nianzu
	//Start
	cvSubRS( &(CvMat)mat_eye_lc_dstrb, cvScalar( cascade_eye_lc->count + 1 ), &(CvMat)mat_eye_lc_dstrb);
	cvScale( &(CvMat)mat_eye_lc_dstrb, &(CvMat)mat_eye_lc_dstrb, temp_scale );

	cvSubRS( &(CvMat)mat_eye_rc_dstrb, cvScalar( cascade_eye_lc->count + 1 ), &(CvMat)mat_eye_rc_dstrb);
	cvScale( &(CvMat)mat_eye_rc_dstrb, &(CvMat)mat_eye_rc_dstrb, temp_scale );

	temp_scale = float(1./cascade_mth_lc->count);
	cvSubRS( &(CvMat)mat_mth_lc_dstrb, cvScalar(cascade_mth_lc->count + 1), &(CvMat)mat_mth_lc_dstrb);
	cvScale( &(CvMat)mat_mth_lc_dstrb, &(CvMat)mat_mth_lc_dstrb, temp_scale);

	cvSubRS( &(CvMat)mat_mth_rc_dstrb, cvScalar( cascade_mth_lc->count + 1 ), &(CvMat)mat_mth_rc_dstrb);
	cvScale( &(CvMat)mat_mth_rc_dstrb, &(CvMat)mat_mth_rc_dstrb, temp_scale );
	//End

	return 0;

}


void CxCompDetector::GenerateMultiOriginalParameters(float multiParameters[NGROUP][NPARAMETERS])
{
	CvPoint minPoints[COMP_COUNT];
	

	CvPoint locationPt[4][COMP_COUNT];
	//int gauss_size_lut[COMP_COUNT] = {7,7,7,7,17,17};
	int gauss_size_lut[COMP_COUNT] = {5,5,5,5,7,7};
	for (int i = 0;i < COMP_COUNT;i++)
	{
		double minValue, maxValue;
		cv::Mat matrix = (*mat_dstrb_lut[i]).clone();//*mat_dstrb_lut[i];//;//; //(*mat_dstrb_lut[i]).clone();
		CvMat searchMat;
		cvGetSubRect(&(CvMat)matrix, &searchMat, rc_comps[i]);
		cvSmooth(&searchMat, &searchMat, CV_GAUSSIAN, gauss_size_lut[i]);
		cvMinMaxLoc(&searchMat, &minValue, &maxValue, &minPoints[i], NULL);
		minPoints[i].x += rc_comps[i].x;
		minPoints[i].y += rc_comps[i].y;
	}

	locationPt[0][0].x = minPoints[0].x;
	locationPt[0][0].y = minPoints[0].y;//first group-Left_left	
	locationPt[1][0].x = minPoints[0].x;
	locationPt[1][0].y = minPoints[0].y;//second group-Left_left

	locationPt[2][1].x = minPoints[1].x;
	locationPt[2][1].y = minPoints[1].y;//third group-Left_right
	locationPt[3][1].x = minPoints[1].x;
	locationPt[3][1].y = minPoints[1].y;//fourth group-Left_right


	locationPt[1][2].x = minPoints[2].x;
	locationPt[1][2].y = minPoints[2].y;//second group-Right_left
	locationPt[3][2].x = minPoints[2].x;
	locationPt[3][2].y = minPoints[2].y;//fourth group-Right_left

	locationPt[0][3].x = minPoints[3].x;
	locationPt[0][3].y = minPoints[3].y;//first group-Right_right
	locationPt[2][3].x = minPoints[3].x;
	locationPt[2][3].y = minPoints[3].y;//third group-Right_right


	//first group
	locationPt[0][1].x = int(locationPt[0][0].x + (locationPt[0][3].x - locationPt[0][0].x)*0.27907 + 0.5);
	locationPt[0][1].y = int(locationPt[0][0].y + (locationPt[0][3].y - locationPt[0][0].y)*0.27907 + 0.5);
	locationPt[0][2].x = int(locationPt[0][0].x + (locationPt[0][3].x - locationPt[0][0].x)*0.72093 + 0.5);
	locationPt[0][2].y = int(locationPt[0][0].y + (locationPt[0][3].y - locationPt[0][0].y)*0.72093 + 0.5);

	//second group
	locationPt[1][1].x = int(locationPt[1][0].x + (locationPt[1][2].x - locationPt[1][0].x)*0.38710 + 0.5);
	locationPt[1][1].y = int(locationPt[1][0].y + (locationPt[1][2].y - locationPt[1][0].y)*0.38710 + 0.5);
	locationPt[1][3].x = int(locationPt[1][0].x + (locationPt[1][2].x - locationPt[1][0].x)*1.38710 + 0.5);
	locationPt[1][3].y = int(locationPt[1][0].y + (locationPt[1][2].y - locationPt[1][0].y)*1.38710 + 0.5);

	//third group
	locationPt[2][0].x = int(locationPt[2][1].x - (locationPt[2][3].x - locationPt[2][1].x)*0.38710 + 0.5);
	locationPt[2][0].y = int(locationPt[2][1].y - (locationPt[2][3].y - locationPt[2][1].y)*0.38710 + 0.5);
	locationPt[2][2].x = int(locationPt[2][1].x + (locationPt[2][3].x - locationPt[2][1].x)*0.61290 + 0.5);
	locationPt[2][2].y = int(locationPt[2][1].y + (locationPt[2][3].y - locationPt[2][1].y)*0.61290 + 0.5);

	//fourth group
	locationPt[3][0].x = int(locationPt[3][1].x-(locationPt[3][2].x - locationPt[3][1].x)*0.63158);
	locationPt[3][0].y = int(locationPt[3][1].y-(locationPt[3][2].y - locationPt[3][1].y)*0.63158);
	locationPt[3][3].x = int(locationPt[3][2].x+(locationPt[3][2].x - locationPt[3][1].x)*0.63158);
	locationPt[3][3].y = int(locationPt[3][2].y+(locationPt[3][2].y - locationPt[3][1].y)*0.63158);


	for (int i = 0;i < 4;i++)
	{
		locationPt[i][4] = minPoints[4];
		locationPt[i][5] = minPoints[5];
	}

	for (int i = 0;i < 4;i++)
	{
		float eyecenterX = float(0.5*(locationPt[i][0].x + locationPt[i][3].x));
		float eyecenterY = float(0.5*(locationPt[i][0].y + locationPt[i][3].y));
		float lefteyecenterX = float(0.5*(locationPt[i][0].x + locationPt[i][1].x));
		float lefteyecenterY = float(0.5*(locationPt[i][0].y + locationPt[i][1].y));
		float righteyecenterX = float(0.5*(locationPt[i][2].x + locationPt[i][3].x));
		float righteyecenterY = float(0.5*(locationPt[i][2].y + locationPt[i][3].y));

		float mouthcenterX = float(0.5*(locationPt[i][4].x + locationPt[i][5].x));
		float mouthcenterY = float(0.5*(locationPt[i][4].y + locationPt[i][5].y));

		float L1 = Distance(locationPt[i][0], locationPt[i][3]);
		float L2 = Distance(cvPoint2D32f(eyecenterX, eyecenterY), cvPoint2D32f(mouthcenterX, mouthcenterY));
		float L3 = Distance(locationPt[i][4] , locationPt[i][5]);

		float eyeSlope1 = Slope(locationPt[i][3], locationPt[i][0]);
		float eyeSlope2 = Slope(cvPoint2D32f(lefteyecenterX, lefteyecenterY), 
			cvPoint2D32f(righteyecenterX, righteyecenterY));
		float eyeSlope = Slope(locationPt[i][3], locationPt[i][0]);
		float mouthSlope = Slope(locationPt[i][3], locationPt[i][0]);
		float dist = float(0.75*Distance(locationPt[i][0], locationPt[i][3]));

		if ((((fabs(eyeSlope)) > 0.7) || (dist > L2))&&((L2/L1)>(AVG_L2_L1 - STDVAR_L2_L1*3))&&((L3/L1)>(AVG_L3_L1 - STDVAR_L3_L1*3))&&
			(fabs(eyeSlope1 - eyeSlope2)<0.18)&&fabs(eyeSlope1 - mouthSlope) < 0.18)
		{
			multiParameters[i][0] = float((locationPt[i][0].x + locationPt[i][3].x)/2);
			multiParameters[i][1] = float((locationPt[i][0].y + locationPt[i][3].y)/2);
			multiParameters[i][2] = mouthcenterX;
			multiParameters[i][3] = mouthcenterY;
			multiParameters[i][4] = float(L1*0.5);
			multiParameters[i][5] = float(L3*0.5);
			//L1 = L1/0;
		}
		else
		{
			multiParameters[i][0] = float((locationPt[i][0].x + locationPt[i][3].x)/2);
			multiParameters[i][1] = float((locationPt[i][0].y + locationPt[i][3].y)/2);
			multiParameters[i][4] = float(Distance(locationPt[i][0], locationPt[i][3])/2);
			multiParameters[i][2] = (dist*(locationPt[i][0].y - eyecenterY)+(eyecenterX*multiParameters[i][4]))/multiParameters[i][4];
			multiParameters[i][3] = (dist*(eyecenterX - locationPt[i][0].x)+(eyecenterY*multiParameters[i][4]))/multiParameters[i][4];
			multiParameters[i][5] = float(multiParameters[i][4]*0.58);
		}
	}
}


void CxCompDetector::CalcParametersFrom6Points( CvPoint* locations, float* parameters )
{
	CvPoint2D32f pt_eye_center = cvPoint2D32f( (locations[0].x + locations[3].x)/2., 
		(locations[0].y + locations[3].y)/2. );
	CvPoint2D32f pt_mth_center = cvPoint2D32f( (locations[4].x + locations[5].x)/2., 
		(locations[4].y + locations[5].y)/2. );

	parameters[0] = float(pt_eye_center.x);
	parameters[1] = float(pt_eye_center.y);
	parameters[2] = float(pt_mth_center.x);
	parameters[3] = float(pt_mth_center.y);
	parameters[4] = Distance( cvPointTo32f(locations[0]), pt_eye_center );
	parameters[5] = Distance( cvPointTo32f(locations[4]), pt_mth_center );

}
void CxCompDetector::CalcParametersFrom6Points( CvPoint2D32f* locations, float* parameters )
{
	CvPoint2D32f pt_eye_center = cvPoint2D32f((locations[0].x + locations[3].x)/2.f, 
		(locations[0].y + locations[3].y)/2.f) ;
	CvPoint2D32f pt_mth_center = cvPoint2D32f( (locations[4].x + locations[5].x)/2.f, 
		(locations[4].y + locations[5].y)/2.f );

	parameters[0] = float(pt_eye_center.x);
	parameters[1] = float(pt_eye_center.y);
	parameters[2] = float(pt_mth_center.x);
	parameters[3] = float(pt_mth_center.y);
	parameters[4] = Distance( locations[0], pt_eye_center );
	parameters[5] = Distance( locations[4], pt_mth_center );

}

void CxCompDetector::Calc6PointsFromParameters(float* parameters, CvPoint2D32f* locations)
{
	float x1, y1, x2, y2, l1, l2;
	float dist;
	float dy1, dx1, dy2, dx2;
	CvPoint2D32f temp6Pt[6] = {0.};
	int i;

	x1 = parameters[0];
	y1 = parameters[1];
	x2 = parameters[2];
	y2 = parameters[3];
	l1 = parameters[4];
	l2 = parameters[5];
	dist = Distance(cvPoint2D32f(x1, y1), 
		cvPoint2D32f(x2, y2));

	if ((parameters[1] != parameters[3]))//if x1 == x2 or y1 == y2
	{
		//4points of eyes
		dy1 = (x2 - x1)*l1/(dist + FLT_MIN);
		dx1 = (y1 - y2)*l1/(dist + FLT_MIN);//sign !!
		temp6Pt[0].x = x1 + dx1;
		temp6Pt[0].y = y1 + dy1;
		temp6Pt[1].x = temp6Pt[0].x + ((x1 - temp6Pt[0].x)*0.55814f);
		temp6Pt[1].y = temp6Pt[0].y + ((y1 - temp6Pt[0].y)*0.55814f);
		temp6Pt[2].x = temp6Pt[0].x + ((x1 - temp6Pt[0].x)*1.48837f);
		temp6Pt[2].y = temp6Pt[0].y + ((y1 - temp6Pt[0].y)*1.48837f);
		temp6Pt[3].x = temp6Pt[0].x + ((x1 - temp6Pt[0].x)*2);
		temp6Pt[3].y = temp6Pt[0].y + ((y1 - temp6Pt[0].y)*2);

		//2points of mouth
		dy2 = (x2 - x1)*l2/(dist + FLT_MIN);
		dx2 = (y1 - y2)*l2/(dist + FLT_MIN);//sign !!
		temp6Pt[4].x = x2 + dx2;
		temp6Pt[4].y = y2 + dy2;
		temp6Pt[5].x = temp6Pt[4].x + ((x2 - temp6Pt[4].x)*2);
		temp6Pt[5].y = temp6Pt[4].y + ((y2 - temp6Pt[4].y)*2);
	}

	for (i = 0;i < COMP_COUNT;i++)
	{
		locations[i].x = temp6Pt[i].x;
		locations[i].y = temp6Pt[i].y;
	}
}

bool CxCompDetector::IsWithinImage(CvPoint2D32f* points)
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

//calculate energy for current shape
bool CxCompDetector::Energy(CvPoint2D32f* cornersLocations, CvMat *result)
{
	int i, j;
	float x, y;
	float yEnd, xEnd;

	for (i = 0;i < COMP_COUNT;i++)
	{
		const CvHaarClassifierCascade* cascade = cascade_lut[i];
		CvSize size = cascade->orig_window_size;

		int region_begin_x = size.width/2 + 1;
		int region_end_x = sz_canvas.width - size.width/2 - 1;
		int region_begin_y = size.height/2 + 1;
		int region_end_y = sz_canvas.height - size.height/2 - 1;

		for ( y = cornersLocations[i].y - (REGION_SIZE/2), 
			yEnd = (cornersLocations[i].y + (REGION_SIZE/2));
			y <= yEnd;y++ )
		{
			if ( y > region_end_y || y < region_begin_y)
			{
				return false;
			}
			for (x = ((cornersLocations[i].x - (REGION_SIZE/2))),
				xEnd = cornersLocations[i].x + (REGION_SIZE/2);
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
		for (i = 0;i < COMP_COUNT;i++)
		{
			const CvHaarClassifierCascade* cascade = cascade_lut[i];
			cv::Mat* mat_dstrb = mat_dstrb_lut[i];
			CvRect search_rect = cascade_search_rect[i];
			float nstage = float(cascade->count);
			CvSize size = cascade->orig_window_size;
			int j = i*REGION_SIZE*REGION_SIZE;

			float x, y;
			float yEnd, xEnd;

			for ( y = cornersLocations[i].y - (REGION_SIZE/2), 
				yEnd = (cornersLocations[i].y + (REGION_SIZE/2) + 0.00001f);
				y < yEnd;y+=1 )
			{
				int ny	= int(y);
				int ny1 = ny + 1;
				float beta = y - ny;

				for (x = ((cornersLocations[i].x - (REGION_SIZE/2))),
					xEnd = cornersLocations[i].x + (REGION_SIZE/2) + 0.00001f;
					x < xEnd;x+=1)
				{
					float v = 0.f;
					int nx	= int(x);
					int nx1 = nx + 1;
					float alfa = x - nx;

					//Modified by Nianzu
					//Start
					CvMat* cvmat_dstrb = &CvMat(*mat_dstrb);

					//float fxy	= CV_MAT_ELEM(*(new CvMat(*mat_dstrb)), float, ny, nx);					
					//float fx1y	= CV_MAT_ELEM(*(new CvMat(*mat_dstrb)), float, ny, nx1);
					//float fxy1	= CV_MAT_ELEM(*(new CvMat(*mat_dstrb)), float, ny1, nx);
					//float fx1y1 = CV_MAT_ELEM(*(new CvMat(*mat_dstrb)), float, ny1, nx1);

					float fxy = CV_MAT_ELEM(*cvmat_dstrb, float, ny, nx);
					float fx1y = CV_MAT_ELEM(*cvmat_dstrb, float, ny, nx1);
					float fxy1 = CV_MAT_ELEM(*cvmat_dstrb, float, ny1, nx);
					float fx1y1 = CV_MAT_ELEM(*cvmat_dstrb, float, ny1, nx1);

					//cvReleaseMat(&cvmat_dstrb);
					//End
					v = cxBilinear(fxy, fx1y, fxy1, fx1y1, alfa, beta);
					cvmSet(result, j++, 0, v);
				}
			}
		}
	}
	else//track mode
	{
#ifdef _OPENMP
#pragma omp parallel for schedule(guided) 
#endif
		for (i = 0;i < COMP_COUNT;i++)
		{
			CvHaarClassifierCascade* cascade = cascade_lut[i];
			cv::Mat* mat_dstrb = mat_dstrb_lut[i];
			CvRect search_rect = cascade_search_rect[i];
			float nstage = float(cascade->count);
			CvSize size = cascade->orig_window_size;

			int j = i*REGION_SIZE*REGION_SIZE;
			float x, y;
			float yEnd, xEnd;

			for ( y = cornersLocations[i].y - (REGION_SIZE/2), 
				yEnd = (cornersLocations[i].y + (REGION_SIZE/2) + 0.00001f);
				y < yEnd;y+=1 )
			{
				int iy = int(y);
				int ey = iy&~1;
				int ey2 = ey + 2;
				float *mat_dstrb_data1 = (float*)(mat_dstrb->row(ey).data);	//Modified by Nianzu
				float *mat_dstrb_data2 = (float*)(mat_dstrb->row(ey + 2).data);	//Modified by Nianzu

				for (x = ((cornersLocations[i].x - (REGION_SIZE/2))),
					xEnd = cornersLocations[i].x + (REGION_SIZE/2) + 0.00001f;
					x < xEnd;x+=1)
				{
					int ix = int(x);
					int ex = ix&~1;
					int ex2 = ex + 2;
					float vl = 0.0f;
					
					//(ex, ey)
					if (mat_dstrb_data1[ex] == 0.0f)
					{
						CvPoint tempPt = cvPoint( ex - (size.width/2), ey - (size.height/2) );
						int rt_stage = cvRunHaarClassifierCascade( cascade, 
							tempPt, 0 );
						rt_stage = 0 - rt_stage;
						rt_stage = (rt_stage == -1)?int(nstage):rt_stage;
						mat_dstrb_data1[ex] = float((nstage + 1 - rt_stage)/nstage);
					}
					//(ex+2, ey)
					if (mat_dstrb_data1[ex2] == 0.0f)
					{
						CvPoint tempPt = cvPoint( ex2 - (size.width/2), ey - (size.height/2) );
						int rt_stage = cvRunHaarClassifierCascade( cascade, 
							tempPt, 0 );
						rt_stage = 0 - rt_stage;
						rt_stage = (rt_stage == -1)?int(nstage):rt_stage;
						mat_dstrb_data1[ex2] = float((nstage + 1 - rt_stage)/nstage);
					}
					//(ex, ey+2)
					if (mat_dstrb_data2[ex] == 0.0f)
					{
						CvPoint tempPt = cvPoint( ex - (size.width/2), ey2 - (size.height/2) );
						int rt_stage = cvRunHaarClassifierCascade( cascade, 
							tempPt, 0 );
						rt_stage = 0 - rt_stage;
						rt_stage = (rt_stage == -1)?int(nstage):rt_stage;
						mat_dstrb_data2[ex] = float((nstage + 1 - rt_stage)/nstage);
					}
					//(ex+2, ey+2)
					if (mat_dstrb_data2[ex2] == 0.0f)
					{
						CvPoint tempPt = cvPoint( ex2 - (size.width/2), ey2 - (size.height/2) );
						int rt_stage = cvRunHaarClassifierCascade( cascade, 
							tempPt, 0 );
						rt_stage = 0 - rt_stage;
						rt_stage = (rt_stage == -1)?int(nstage):rt_stage;
						mat_dstrb_data2[ex2] = float((nstage + 1 - rt_stage)/nstage);
					}
					{
						float fxy	= mat_dstrb_data1[ex];
						float fxy1	= mat_dstrb_data2[ex];
						float fx1y	= mat_dstrb_data1[ex2];
						float fx1y1 = mat_dstrb_data2[ex2];
						float alfa = 0.5f*(x - ex);
						float beta = 0.5f*(y - ey);
						vl = cxBilinear(fxy, fxy1, fx1y, fx1y1, alfa, beta);
					}
					cvmSet(result, j++, 0, vl);
				}
			}
		}
	}
	float L1 = Distance(cornersLocations[0], cornersLocations[3]);
	float L2 = Distance(cvPoint2D32f((cornersLocations[0].x+cornersLocations[3].x)/2., 
		(cornersLocations[0].y+cornersLocations[3].y)/2.),
		cvPoint2D32f((cornersLocations[4].x+cornersLocations[5].x)/2., 
		(cornersLocations[4].y+cornersLocations[5].y)/2.));
	float L3 = Distance( cornersLocations[4], cornersLocations[5] );
	j = COMP_COUNT*REGION_SIZE*REGION_SIZE;
	cvmSet( result, j++, 0, FACTOR_1*(L2/L1-AVG_L2_L1)/STDVAR_L2_L1 );
	cvmSet( result, j++, 0, FACTOR_2*(L3/L1-AVG_L3_L1)/STDVAR_L3_L1 );
	cvmSet( result, j, 0, FACTOR_3*(L1-AVG_L1)/STDVAR_L1 );

	return true;
};


bool CxCompDetector::CalcJacobian(CvMat* jacobianMat, CvMat* parametersMat)
{
	const float DIFFSTEP = 1.f;
	static bool jitter = true;
	float diffStep1[NPARAMETERS]	= {1.f, 1.f, 1.f, 1.f, -1.f, -1.f};
	float diffStep2[NPARAMETERS]	= {-1.f, -1.f, -1.f, -1.f, 1.f, 1.f};
	float diffStep[NPARAMETERS]		= {1.f, 1.f, 1.f, 1.f, 1.f, 1.f};
	if (mode == MODE_TRACK)
	{
		if (jitter == true)
		{
			memcpy(diffStep, diffStep1, NPARAMETERS*sizeof(float));
		}
		else
		{
			memcpy(diffStep, diffStep2, NPARAMETERS*sizeof(float));
		}
		jitter = !jitter;
	}
	
	CvPoint2D32f currentCorners[COMP_COUNT];
	CvPoint2D32f diffCorners[COMP_COUNT];
	float diffParameters[COMP_COUNT];
	CvSize disrtibuteMatSize = sz_canvas;
	CvMat* currentEnergyMat = &CvMat(mat_current_temp_energy); //Modified by Nianzu

	CvMat diffEnergyMat;
	float* parameters = parametersMat->data.fl;
	float* currentEnergy = currentEnergyMat->data.fl;
	float tempVal = 0.0;

	memcpy( diffParameters, parameters, NPARAMETERS*sizeof( parameters[0] ) );

	Calc6PointsFromParameters( parameters,currentCorners );

	if ( false == Energy( currentCorners, currentEnergyMat ) )
	{
		return false;
	}

	for ( int i = 0; i < NPARAMETERS; i++ )
	{
		int range = (i%2 == 1)?disrtibuteMatSize.height:disrtibuteMatSize.width;
		memcpy(diffParameters, parameters, NPARAMETERS*sizeof(parameters[0]));
		diffParameters[i] = (diffParameters[i] + diffStep[i] < range)?(diffParameters[i] + diffStep[i]):(range - 1);
		Calc6PointsFromParameters( diffParameters, diffCorners );
		cvGetCol( jacobianMat, &diffEnergyMat, i );
		if ( false == Energy( diffCorners, &diffEnergyMat ) )
		{
			return false;
		}
		cvSub( &diffEnergyMat, currentEnergyMat, &diffEnergyMat );
		tempVal = float(1./(diffParameters[i] - parameters[i]));
		cvScale(&diffEnergyMat, &diffEnergyMat, tempVal);
	}
	return true;
}
void CxCompDetector::CalcNewParameters( CvMat *matA, CvMat* fxMat, CvMat *oldMatX, CvMat *newMatX )
{
	////////////////////////////////////////////////////////////
	//Modified by Nianzu
	cvTranspose( matA, &(CvMat)mat_AT );
	cvMulTransposed( &(CvMat)mat_AT, &(CvMat)mat_hessian, 0 );
	cvMatMul( &(CvMat)mat_AT, fxMat, &(CvMat)mat_temp_calt_par );
	cvSolve( &(CvMat)mat_hessian, &(CvMat)mat_temp_calt_par, &(CvMat)mat_soln, CV_LU );
	cvSub( oldMatX, &(CvMat)mat_soln, newMatX );
	//////////////////////////////////////////////////////////
}

bool CxCompDetector::IsFitShape( float* parameters )
{
	const float RANGE_L2_L1_LOW		= 0.508806f;
	const float RANGE_L2_L1_HIGH	= 1.031484f;
	const float RANGE_L3_L1_LOW		= 0.31611f;
	const float RANGE_L3_L1_HIGH	= 0.860298f;

	float L1 = parameters[4]*2;
	float L2 = Distance( cvPoint2D32f( parameters[0], parameters[1] ),
		cvPoint2D32f( parameters[2], parameters[3] ) );
	float L3 = parameters[5]*2;

	if ( (L2/L1) < RANGE_L2_L1_LOW || (L2/L1) > RANGE_L2_L1_HIGH )
	{
		return false;
	}
	if ( (L3/L1) < RANGE_L3_L1_LOW || (L3/L1) > RANGE_L3_L1_HIGH )
	{
		return false;
	}
	
	if ( L1 < sz_canvas.width/2 )
	{
		return false;
	}

	return true;
}

bool CxCompDetector::NewtonIteration( float *parameters, float *energySum )
{
	const int MAXITER = 6;

	//Modified by Nianzu
	//Start
	CvMat* currentParametersMat = &CvMat(mat_current_parameters);
	CvMat* newParametersMat = &CvMat(mat_new_parameters);

	CvMat* deltaParametersMat = &CvMat(mat_delta_parameters);
	CvMat* jacobianMat = &CvMat(mat_jacobian);
	CvMat* currentEnergy = &CvMat(mat_current_energy);

	float* currentParameters = currentParametersMat->data.fl;
	float* newParameters = newParametersMat->data.fl;

	CvPoint2D32f currentPoints[COMP_COUNT];
	CvPoint2D32f newPoints[COMP_COUNT];
	float e0 = 0.0;
	int iter,i;
	memcpy(currentParameters, parameters, NPARAMETERS*sizeof(parameters[0]));
	float sum_energy_current = 0.0f;
	float sum_energy_new = 0.0f;

	Calc6PointsFromParameters(currentParameters, currentPoints);
	if (false == Energy( currentPoints, currentEnergy))
	{
		//fprintf( stderr, "NewtonIteration ERROR: Points out of image!!\n" );
		*energySum = FLT_MAX;
		return false;
	}

	sum_energy_current = (float)cvSum(currentEnergy).val[0];

	for (iter = 0;iter < MAXITER;iter++)
	{

		if (false == CalcJacobian( jacobianMat, currentParametersMat ))
		{
			//fprintf( stderr, "NewtonIteration ERROR: Points out of image!!\n" );
			*energySum = FLT_MAX;
			return false;
		}

		CalcNewParameters(jacobianMat, currentEnergy, currentParametersMat, newParametersMat);
		Calc6PointsFromParameters(newParameters, newPoints);
		if (!IsWithinImage( newPoints ))
		{
			memcpy(parameters, currentParameters, NPARAMETERS*sizeof(parameters[0]));
			//fprintf( stderr, "NewtonIteration ERROR: Points out of image!!\n" );
			*energySum = FLT_MAX;
			return false;
		}

		if (false == Energy( newPoints, currentEnergy))
		{
			//fprintf( stderr, "NewtonIteration ERROR: Points out of image!!\n" );
			memcpy(parameters, currentParameters, NPARAMETERS*sizeof(parameters[0]));
			*energySum = FLT_MAX;
			return false;
		}

		if (mode == MODE_TRACK)
		{
			sum_energy_new = (float)cvSum( currentEnergy ).val[0];
			if (sum_energy_new > sum_energy_current)
			{
				break;
			}
			sum_energy_current = sum_energy_new;
		}

		cvSub( currentParametersMat, newParametersMat, deltaParametersMat );
		e0 = float(cvNorm(deltaParametersMat, 0, CV_L1 ));
		if (e0 < 0.25f*COMP_COUNT)
		{
			memcpy(currentParameters, newParameters, NPARAMETERS*sizeof(currentParameters[0]));
			memcpy(currentPoints, newPoints, COMP_COUNT*sizeof(currentPoints[0]));
			break;
		}
		memcpy(currentParameters, newParameters, NPARAMETERS*sizeof(currentParameters[0]));
		memcpy(currentPoints, newPoints, COMP_COUNT*sizeof(currentPoints[0]));
	}

	memcpy(parameters, newParameters, NPARAMETERS*sizeof(parameters[0]));
	//printf("iter = %d \n", iter);
	int ncount = 0;
	for (i = REGION_SIZE*REGION_SIZE/2;i < COMP_COUNT*REGION_SIZE*REGION_SIZE;i += REGION_SIZE*REGION_SIZE)
	{
		ncount++;
		*energySum += float(cvmGet(currentEnergy, i, 0));
	}
	*energySum = *energySum/ncount;

	if (*energySum > 0.65f)
	{ 
		//printf("failed::energy = %f   surpass !\n  ",  *energySum);
		*energySum = FLT_MAX;
		return false;
	}

	if (false == IsFitShape(newParameters))
	{
		*energySum += 0.2f;
	}

	memcpy(parameters, newParameters, NPARAMETERS*sizeof(parameters[0]));
	return true;
}

