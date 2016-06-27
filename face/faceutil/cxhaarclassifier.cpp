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

#include "cxhaarfacetracker.hpp"
#include "cxhaarclassifier.hpp"

//#define __SSE2__

#ifdef __SSE2__
#include <emmintrin.h>
#ifndef SIMD_WIDTH
#define SIMD_WIDTH 4
#endif

#if defined __SSE4__
#include <smmintrin.h>
#else
#define _mm_blendv_pd(a, b, m) _mm_xor_pd(a, _mm_and_pd(_mm_xor_pd(b, a), m))
#endif

#if defined(__GNUC__) && !defined(_MM_ALIGN16)
#define _MM_ALIGN16 __attribute__ ((aligned(16)))
#endif
#endif

#define calc_sum(rect,offset) \
    ((rect).p0[offset] - (rect).p1[offset] - (rect).p2[offset] + (rect).p3[offset])

#ifdef __SSE2__
void cxRunHaarClassiferCascade( CvHaarClassifierCascade* _cascade, 
                                CvRect rect, int step, int start_stage, 
                                CvSeq* seq_rect )
{
    int x, y, i, j, k;
    int count = 0, passed = 0, end;
    CvHidHaarClassifierCascade* cascade = _cascade->hid_cascade;

    int size = (rect.width+2) * (rect.height+2);
    int*   win_offset = (int *) _mm_malloc (size * sizeof(int), 64);       // offset of the pixels for detection
    float* win_factor = (float *) _mm_malloc (size * sizeof(float), 64);   // their factor values
    
    for( k = 0, y = rect.y; y < rect.y+rect.height; y+=step )
    {
        for( x = rect.x; x < rect.x+rect.width; x+=step, k ++ )
        {
            int pq_offset;
            float mean, variance_norm_factor;

            win_offset[k] = y * (cascade->sum.step/sizeof(sumtype)) + x;
            pq_offset = y * (cascade->sqsum.step/sizeof(sqsumtype)) + x;
            mean = (float)(calc_sum(*cascade, win_offset[k])*cascade->inv_window_area);
            variance_norm_factor = (float)(cascade->pq0[pq_offset] - cascade->pq1[pq_offset] -
                cascade->pq2[pq_offset] + cascade->pq3[pq_offset]);
            variance_norm_factor = variance_norm_factor*(float)cascade->inv_window_area - mean*mean;
            if( variance_norm_factor >= 0. )
                win_factor[k] = sqrt(variance_norm_factor);
            else
                win_factor[k] = 1.;
        }
    }

    count = k;

    for( i = start_stage; i < cascade->count; i++ )
    {
        passed = 0;
        end = count/SIMD_WIDTH*SIMD_WIDTH;
                
        for(k = 0; k < end; k+=SIMD_WIDTH )
        {
            __m128 xmm_pixel, xmm_value, xmm_alpha1, xmm_alpha2, xmm_sum, xmm_stagesum;
            __m128i xmm_ivalue, xmm_ipixel;
            _MM_ALIGN16 float stage_sum[SIMD_WIDTH] = {0.0};

            xmm_stagesum = _mm_setzero_ps();
            for( j = 0; j < cascade->stage_classifier[i].count; j++ )
            {
                CvHidHaarClassifier* classifier = cascade->stage_classifier[i].classifier + j;
                CvHidHaarTreeNode* node = classifier->node;

                /* sum[0] =  calc_sum(node->feature.rect[0],win_offset[k]) * node->feature.rect[0].weight;
                sum[1] =  calc_sum(node->feature.rect[0],win_offset[k+1]) * node->feature.rect[0].weight;
                sum[2] =  calc_sum(node->feature.rect[0],win_offset[k+2]) * node->feature.rect[0].weight;
                sum[3] =  calc_sum(node->feature.rect[0],win_offset[k+3]) * node->feature.rect[0].weight;

                sum[0] += calc_sum(node->feature.rect[1],win_offset[k]) * node->feature.rect[1].weight;
                sum[1] += calc_sum(node->feature.rect[1],win_offset[k+1]) * node->feature.rect[1].weight;
                sum[2] += calc_sum(node->feature.rect[1],win_offset[k+2]) * node->feature.rect[1].weight;
                sum[3] += calc_sum(node->feature.rect[1],win_offset[k+3]) * node->feature.rect[1].weight;
                */

                // first feature
                xmm_ivalue = _mm_setr_epi32 (node->feature.rect[0].p0[win_offset[k]], 
                    node->feature.rect[0].p0[win_offset[k+1]], 
                    node->feature.rect[0].p0[win_offset[k+2]],
                    node->feature.rect[0].p0[win_offset[k+3]]);
                xmm_ipixel = _mm_setr_epi32 (node->feature.rect[0].p1[win_offset[k]], 
                    node->feature.rect[0].p1[win_offset[k+1]], 
                    node->feature.rect[0].p1[win_offset[k+2]],
                    node->feature.rect[0].p1[win_offset[k+3]]);
                xmm_ivalue = _mm_sub_epi32 (xmm_ivalue, xmm_ipixel);
                xmm_ipixel = _mm_setr_epi32 (node->feature.rect[0].p2[win_offset[k]], 
                    node->feature.rect[0].p2[win_offset[k+1]], 
                    node->feature.rect[0].p2[win_offset[k+2]],
                    node->feature.rect[0].p2[win_offset[k+3]]);
                xmm_ivalue = _mm_sub_epi32 (xmm_ivalue, xmm_ipixel);
                xmm_ipixel = _mm_setr_epi32 (node->feature.rect[0].p3[win_offset[k]], 
                    node->feature.rect[0].p3[win_offset[k+1]], 
                    node->feature.rect[0].p3[win_offset[k+2]],
                    node->feature.rect[0].p3[win_offset[k+3]]);
                xmm_ivalue = _mm_add_epi32 (xmm_ivalue, xmm_ipixel);
                xmm_value = _mm_cvtepi32_ps (xmm_ivalue);

                xmm_pixel = _mm_set_ps1 (node->feature.rect[0].weight);
                xmm_sum   = _mm_mul_ps (xmm_value, xmm_pixel);

                // second feature 
                xmm_ivalue = _mm_setr_epi32 (node->feature.rect[1].p0[win_offset[k]], 
                    node->feature.rect[1].p0[win_offset[k+1]], 
                    node->feature.rect[1].p0[win_offset[k+2]],
                    node->feature.rect[1].p0[win_offset[k+3]]);
                xmm_ipixel = _mm_setr_epi32 (node->feature.rect[1].p1[win_offset[k]], 
                    node->feature.rect[1].p1[win_offset[k+1]], 
                    node->feature.rect[1].p1[win_offset[k+2]],
                    node->feature.rect[1].p1[win_offset[k+3]]);
                xmm_ivalue = _mm_sub_epi32 (xmm_ivalue, xmm_ipixel);
                xmm_ipixel = _mm_setr_epi32 (node->feature.rect[1].p2[win_offset[k]], 
                    node->feature.rect[1].p2[win_offset[k+1]], 
                    node->feature.rect[1].p2[win_offset[k+2]],
                    node->feature.rect[1].p2[win_offset[k+3]]);
                xmm_ivalue = _mm_sub_epi32 (xmm_ivalue, xmm_ipixel);
                xmm_ipixel = _mm_setr_epi32 (node->feature.rect[1].p3[win_offset[k]], 
                    node->feature.rect[1].p3[win_offset[k+1]], 
                    node->feature.rect[1].p3[win_offset[k+2]],
                    node->feature.rect[1].p3[win_offset[k+3]]);
                xmm_ivalue = _mm_add_epi32 (xmm_ivalue, xmm_ipixel);
                xmm_value = _mm_cvtepi32_ps (xmm_ivalue);

                xmm_pixel = _mm_set_ps1 (node->feature.rect[1].weight);
                xmm_value = _mm_mul_ps (xmm_value, xmm_pixel);
                xmm_sum   = _mm_add_ps (xmm_sum, xmm_value);

                // third feature when available
                if( node->feature.rect[2].p0 )
                {
                    // sum[0] += calc_sum(node->feature.rect[2],win_offset[k]  ) * node->feature.rect[2].weight;
                    // sum[1] += calc_sum(node->feature.rect[2],win_offset[k+1]) * node->feature.rect[2].weight;
                    // sum[2] += calc_sum(node->feature.rect[2],win_offset[k+2]) * node->feature.rect[2].weight;
                    // sum[3] += calc_sum(node->feature.rect[2],win_offset[k+3]) * node->feature.rect[2].weight;

                    xmm_ivalue = _mm_setr_epi32 (node->feature.rect[2].p0[win_offset[k]], 
                        node->feature.rect[2].p0[win_offset[k+1]], 
                        node->feature.rect[2].p0[win_offset[k+2]],
                        node->feature.rect[2].p0[win_offset[k+3]]);
                    xmm_ipixel = _mm_setr_epi32 (node->feature.rect[2].p1[win_offset[k]], 
                        node->feature.rect[2].p1[win_offset[k+1]], 
                        node->feature.rect[2].p1[win_offset[k+2]],
                        node->feature.rect[2].p1[win_offset[k+3]]);
                    xmm_ivalue = _mm_sub_epi32 (xmm_ivalue, xmm_ipixel);
                    xmm_ipixel = _mm_setr_epi32 (node->feature.rect[2].p2[win_offset[k]], 
                        node->feature.rect[2].p2[win_offset[k+1]], 
                        node->feature.rect[2].p2[win_offset[k+2]],
                        node->feature.rect[2].p2[win_offset[k+3]]);
                    xmm_ivalue = _mm_sub_epi32 (xmm_ivalue, xmm_ipixel);
                    xmm_ipixel = _mm_setr_epi32 (node->feature.rect[2].p3[win_offset[k]], 
                        node->feature.rect[2].p3[win_offset[k+1]], 
                        node->feature.rect[2].p3[win_offset[k+2]],
                        node->feature.rect[2].p3[win_offset[k+3]]);
                    xmm_ivalue = _mm_add_epi32 (xmm_ivalue, xmm_ipixel);
                    xmm_value = _mm_cvtepi32_ps (xmm_ivalue);

                    xmm_pixel = _mm_set_ps1 (node->feature.rect[2].weight);
                    xmm_value = _mm_mul_ps (xmm_value, xmm_pixel);
                    xmm_sum   = _mm_add_ps (xmm_sum, xmm_value);
                }

                /* a = classifier->alpha[0];
                b = classifier->alpha[1];

                stage_sum[0] += sum[0] < node->threshold*factor[k] ? a : b;
                stage_sum[1] += sum[1] < node->threshold*factor[k+1] ? a : b;
                stage_sum[2] += sum[2] < node->threshold*factor[k+2] ? a : b;
                stage_sum[3] += sum[3] < node->threshold*factor[k+3] ? a : b; */

                xmm_value = _mm_load_ps (win_factor+k);
                xmm_pixel = _mm_set_ps1 (node->threshold);
                xmm_value = _mm_mul_ps  (xmm_value, xmm_pixel);

                xmm_alpha1 = _mm_set_ps1 (classifier->alpha[0]);
                xmm_alpha2 = _mm_set_ps1 (classifier->alpha[1]);
                xmm_pixel  = _mm_cmplt_ps (xmm_sum, xmm_value);
                xmm_pixel  = _mm_and_ps (xmm_pixel, xmm_alpha1);
                xmm_value  = _mm_cmpge_ps (xmm_sum, xmm_value);
                xmm_value  = _mm_and_ps (xmm_value, xmm_alpha2);
                xmm_value  = _mm_or_ps (xmm_pixel, xmm_value);

                xmm_stagesum = _mm_add_ps (xmm_stagesum, xmm_value);
            }

            _mm_store_ps (stage_sum, xmm_stagesum);
            for (j = 0; j < SIMD_WIDTH; j++)
            {
                if( stage_sum[j] >= cascade->stage_classifier[i].threshold )
                {
                    win_offset[passed] = win_offset[k+j];
                    win_factor[passed] = win_factor[k+j];
                    passed ++;
                }
            }
        }

        // printf ("passed = %d\n", passed);
        for( k = end; k < count; k++ )
        {
            float stage_sum = 0;
            for( j = 0; j < cascade->stage_classifier[i].count; j++ )
            {
                CvHidHaarClassifier* classifier = cascade->stage_classifier[i].classifier + j;
                CvHidHaarTreeNode* node = classifier->node;
                float sum, t = node->threshold*win_factor[k], a, b;

                sum = calc_sum(node->feature.rect[0],win_offset[k]) * node->feature.rect[0].weight;
                sum += calc_sum(node->feature.rect[1],win_offset[k]) * node->feature.rect[1].weight;

                if( node->feature.rect[2].p0 )
                    sum += calc_sum(node->feature.rect[2],win_offset[k]) * node->feature.rect[2].weight;

                a = classifier->alpha[0];
                b = classifier->alpha[1];
                stage_sum += sum < t ? a : b;
            }

            if( stage_sum >= cascade->stage_classifier[i].threshold )
            {
                win_offset[passed] = win_offset[k];
                win_factor[passed] = win_factor[k];
                passed ++;
            }
        }
        count = passed;
        
        if (count == 0)
            break;
    }

    count = passed;
    
    for( k = 0; k < count; k++ )
    {
        y = win_offset[k] / (cascade->sum.step/sizeof(sumtype));
        x = win_offset[k] - y * (cascade->sum.step/sizeof(sumtype));
        CvRect rc = cvRect( x, y, 
            _cascade->real_window_size.width, _cascade->real_window_size.height );
        cvSeqPush( seq_rect, &rc );
    }

    if( win_offset )
        _mm_free (win_offset);
    if( win_factor )
        _mm_free (win_factor);
}

#else

void cxRunHaarClassiferCascade( CvHaarClassifierCascade* _cascade, 
							   CvRect rect, int step, int start_stage, 
							   CvSeq* seq_rect )
{
	int x, y, i, j, k;
	int count = 0, passed = 0;
	CvHidHaarClassifierCascade* cascade = _cascade->hid_cascade;

	int size = (rect.width+2) * (rect.height+2);
	int*   win_offset = (int *) malloc (size * sizeof(int));       // offset of the pixels for detection
	float* win_factor = (float *) malloc (size * sizeof(float));   // their factor values

	for( k = 0, y = rect.y; y < rect.y+rect.height; y+=step )
	{
		for( x = rect.x; x < rect.x+rect.width; x+=step, k ++ )
		{
			int pq_offset;
			float mean, variance_norm_factor;

			win_offset[k] = y * (cascade->sum.step/sizeof(sumtype)) + x;
			pq_offset = y * (cascade->sqsum.step/sizeof(sqsumtype)) + x;
			mean = (float)(calc_sum(*cascade, win_offset[k])*cascade->inv_window_area);
			variance_norm_factor = (float)(cascade->pq0[pq_offset] - cascade->pq1[pq_offset] -
				cascade->pq2[pq_offset] + cascade->pq3[pq_offset]);
			variance_norm_factor = variance_norm_factor*(float)cascade->inv_window_area - mean*mean;
			if( variance_norm_factor >= 0. )
				win_factor[k] = sqrt(variance_norm_factor);
			else
				win_factor[k] = 1.;
		}
	}

	count = k;

	for( i = start_stage; i < cascade->count; i++ )
	{
		passed = 0;
		
		for( k = 0; k < count; k++ )
		{
			float stage_sum = 0;
			for( j = 0; j < cascade->stage_classifier[i].count; j++ )
			{
				CvHidHaarClassifier* classifier = cascade->stage_classifier[i].classifier + j;
				CvHidHaarTreeNode* node = classifier->node;
				float sum, t = node->threshold*win_factor[k], a, b;

				sum = calc_sum(node->feature.rect[0],win_offset[k]) * node->feature.rect[0].weight;
				sum += calc_sum(node->feature.rect[1],win_offset[k]) * node->feature.rect[1].weight;

				if( node->feature.rect[2].p0 )
					sum += calc_sum(node->feature.rect[2],win_offset[k]) * node->feature.rect[2].weight;

				a = classifier->alpha[0];
				b = classifier->alpha[1];
				stage_sum += sum < t ? a : b;
			}

			if( stage_sum >= cascade->stage_classifier[i].threshold )
			{
				win_offset[passed] = win_offset[k];
				win_factor[passed] = win_factor[k];
				passed ++;
			}
		}
		count = passed;

		if (count == 0)
			break;
	}

	count = passed;

	for( k = 0; k < count; k++ )
	{
		y = win_offset[k] / (cascade->sum.step/sizeof(sumtype));
		x = win_offset[k] - y * (cascade->sum.step/sizeof(sumtype));
		CvRect rc = cvRect( x, y, 
			_cascade->real_window_size.width, _cascade->real_window_size.height );
		cvSeqPush( seq_rect, &rc );
	}

	if( win_offset )
		free (win_offset);
	if( win_factor )
		free (win_factor);
}

#endif

CV_INLINE
double icvEvalHidHaarClassifier( CvHidHaarClassifier* classifier,
								double variance_norm_factor,
								size_t p_offset )
{
	int idx = 0;
	do
	{
		CvHidHaarTreeNode* node = classifier->node + idx;
		double t = node->threshold * variance_norm_factor;

		double sum = calc_sum(node->feature.rect[0],p_offset) * node->feature.rect[0].weight;
		sum += calc_sum(node->feature.rect[1],p_offset) * node->feature.rect[1].weight;

		if( node->feature.rect[2].p0 )
			sum += calc_sum(node->feature.rect[2],p_offset) * node->feature.rect[2].weight;

		idx = sum < t ? node->left : node->right;
	}
	while( idx > 0 );
	return classifier->alpha[-idx];
}


int cvRunHaarClassifierCascade( const CvHaarClassifierCascade* _cascade,
						   CvPoint pt, int start_stage )
{
	int result = -1;
	CV_FUNCNAME("cvRunHaarClassifierCascade");

	__BEGIN__;

	int p_offset, pq_offset;
	int i, j;
	double mean, variance_norm_factor;
	CvHidHaarClassifierCascade* cascade;

	if( !CV_IS_HAAR_CLASSIFIER(_cascade) )
		CV_ERROR( !_cascade ? CV_StsNullPtr : CV_StsBadArg, "Invalid cascade pointer" );

	cascade = _cascade->hid_cascade;
	if( !cascade )
		CV_ERROR( CV_StsNullPtr, "Hidden cascade has not been created.\n"
		"Use cvSetImagesForHaarClassifierCascade" );

	if( pt.x < 0 || pt.y < 0 ||
		pt.x + _cascade->real_window_size.width >= cascade->sum.width-2 ||
		pt.y + _cascade->real_window_size.height >= cascade->sum.height-2 )
		EXIT;

	p_offset = pt.y * (cascade->sum.step/sizeof(sumtype)) + pt.x;
	pq_offset = pt.y * (cascade->sqsum.step/sizeof(sqsumtype)) + pt.x;
	mean = calc_sum(*cascade,p_offset)*cascade->inv_window_area;
	variance_norm_factor = cascade->pq0[pq_offset] - cascade->pq1[pq_offset] -
		cascade->pq2[pq_offset] + cascade->pq3[pq_offset];
	variance_norm_factor = variance_norm_factor*cascade->inv_window_area - mean*mean;
	if( variance_norm_factor >= 0. )
		variance_norm_factor = sqrt(variance_norm_factor);
	else
		variance_norm_factor = 1.;

	if( cascade->is_tree )
	{
		CvHidHaarStageClassifier* ptr;
		assert( start_stage == 0 );

		result = 1;
		ptr = cascade->stage_classifier;

		while( ptr )
		{
			double stage_sum = 0;

			for( j = 0; j < ptr->count; j++ )
			{
				stage_sum += icvEvalHidHaarClassifier( ptr->classifier + j,
					variance_norm_factor, p_offset );
			}

			if( stage_sum >= ptr->threshold )
			{
				ptr = ptr->child;
			}
			else
			{
				while( ptr && ptr->next == NULL ) ptr = ptr->parent;
				if( ptr == NULL )
				{
					result = 0;
					EXIT;
				}
				ptr = ptr->next;
			}
		}
	}
	else if( cascade->is_stump_based )
	{
		for( i = start_stage; i < cascade->count; i++ )
		{
#ifndef __SSE2__
			double stage_sum = 0;
#else
			__m128d stage_sum = _mm_setzero_pd();
#endif

			if( cascade->stage_classifier[i].two_rects )
			{
				for( j = 0; j < cascade->stage_classifier[i].count; j++ )
				{
					CvHidHaarClassifier* classifier = cascade->stage_classifier[i].classifier + j;
					CvHidHaarTreeNode* node = classifier->node;
#ifndef __SSE2__
					double t = node->threshold*variance_norm_factor;
					double sum = calc_sum(node->feature.rect[0],p_offset) * node->feature.rect[0].weight;
					sum += calc_sum(node->feature.rect[1],p_offset) * node->feature.rect[1].weight;
					stage_sum += classifier->alpha[sum >= t];
#else
					// ayasin - NHM perf optim. Avoid use of costly flaky jcc
					__m128d t = _mm_set_sd(node->threshold*variance_norm_factor);
					__m128d a = _mm_set_sd(classifier->alpha[0]);
					__m128d b = _mm_set_sd(classifier->alpha[1]);
					__m128d sum = _mm_set_sd(calc_sum(node->feature.rect[0],p_offset) * node->feature.rect[0].weight +
						calc_sum(node->feature.rect[1],p_offset) * node->feature.rect[1].weight);
					t = _mm_cmpgt_sd(t, sum);
					stage_sum = _mm_add_sd(stage_sum, _mm_blendv_pd(b, a, t));
#endif
				}
			}
			else
			{
				for( j = 0; j < cascade->stage_classifier[i].count; j++ )
				{
					CvHidHaarClassifier* classifier = cascade->stage_classifier[i].classifier + j;
					CvHidHaarTreeNode* node = classifier->node;
#ifndef __SSE2__
					double t = node->threshold*variance_norm_factor;
					double sum = calc_sum(node->feature.rect[0],p_offset) * node->feature.rect[0].weight;
					sum += calc_sum(node->feature.rect[1],p_offset) * node->feature.rect[1].weight;
					if( node->feature.rect[2].p0 )
						sum += calc_sum(node->feature.rect[2],p_offset) * node->feature.rect[2].weight;

					stage_sum += classifier->alpha[sum >= t];
#else
					// ayasin - NHM perf optim. Avoid use of costly flaky jcc
					__m128d t = _mm_set_sd(node->threshold*variance_norm_factor);
					__m128d a = _mm_set_sd(classifier->alpha[0]);
					__m128d b = _mm_set_sd(classifier->alpha[1]);
					double _sum = calc_sum(node->feature.rect[0],p_offset) * node->feature.rect[0].weight;
					_sum += calc_sum(node->feature.rect[1],p_offset) * node->feature.rect[1].weight;
					if( node->feature.rect[2].p0 )
						_sum += calc_sum(node->feature.rect[2],p_offset) * node->feature.rect[2].weight;
					__m128d sum = _mm_set_sd(_sum);

					t = _mm_cmpgt_sd(t, sum);
					stage_sum = _mm_add_sd(stage_sum, _mm_blendv_pd(b, a, t));
#endif
				}
			}

#ifndef __SSE2__
			if( stage_sum < cascade->stage_classifier[i].threshold )
#else
			__m128d i_threshold = _mm_set_sd(cascade->stage_classifier[i].threshold);
			if( _mm_comilt_sd(stage_sum, i_threshold) )
#endif
			{
				result = -i;
				EXIT;
			}
		}
	}
	else
	{
		for( i = start_stage; i < cascade->count; i++ )
		{
			double stage_sum = 0;

			for( j = 0; j < cascade->stage_classifier[i].count; j++ )
			{
				stage_sum += icvEvalHidHaarClassifier(
					cascade->stage_classifier[i].classifier + j,
					variance_norm_factor, p_offset );
			}

			if( stage_sum < cascade->stage_classifier[i].threshold )
			{
				result = -i;
				EXIT;
			}
		}
	}

	result = 1;

	__END__;

	return result;
}