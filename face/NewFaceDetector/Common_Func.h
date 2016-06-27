#if !defined  _COMMOND_FUNC_H
#define _COMMOND_FUNC_H

#include <stdio.h>
#include <opencv\cv.h>
#include <opencv\cxcore.h>
#include <opencv\highgui.h>
#include "HvTrain.h"


HvFeature hvHaarFeature( const char* desc,
				int x0, int y0, int w0, int h0, float wt0,
				int x1, int y1, int w1, int h1, float wt1,
				int x2 = 0, int y2 = 0,
				int w2 = 0, int h2 = 0,
				float wt2 = 0.0F);

void HV_SUM_OFFSETS_Func( int *p0, int *p1, int *p2, int *p3, CvRect rect, int step);

#define less_than( a, b ) ((a) < (b))

void Sort_32f( float* array, int length, int aux );

void SortIndexedValArray_16s( short* array, int length, HvValArray* aux);
void SortIndexedValArray_32s( int* array, int length, HvValArray* aux );


float EvalFastFeature( HvFastFeature* feature, int* sum, int* tilted );

void GetSortedIndices(CvMat* val,  CvMat* idx, int sortcols );
void GetSortedIndices(CvMat* val,  CvMat* idx, int sortrows, int sortcols);

#define CMP_VALUES( idx1, idx2 )                                 \
    ( *( (float*) (aux->data + ((int) (idx1)) * aux->step ) ) <  \
      *( (float*) (aux->data + ((int) (idx2)) * aux->step ) ) )


#define HV_MAT2VEC( mat, vdata, vstep, num )       \
    assert( (mat).rows == 1 || (mat).cols == 1 );  \
    (vdata) = ((mat).data.ptr);                    \
    if( (mat).rows == 1 )                          \
    {                                              \
        (vstep) = CV_ELEM_SIZE( (mat).type );      \
        (num) = (mat).cols;                        \
    }                                              \
    else                                           \
    {                                              \
        (vstep) = (mat).step;                      \
        (num) = (mat).rows;                        \
    }

/*
 * get tilted image offsets for <rect> corner points 
 * step - row step (measured in image pixels!) of tilted image
 */
#define HV_TILTED_OFFSETS( p0, p1, p2, p3, rect, step )                   \
    /* (x, y) */                                                          \
    (p0) = (rect).x + (step) * (rect).y;                                  \
    /* (x - h, y + h) */                                                  \
    (p1) = (rect).x - (rect).height + (step) * ((rect).y + (rect).height);\
    /* (x + w, y + w) */                                                  \
    (p2) = (rect).x + (rect).width + (step) * ((rect).y + (rect).width);  \
    /* (x + w - h, y + w + h) */                                          \
    (p3) = (rect).x + (rect).width - (rect).height                        \
           + (step) * ((rect).y + (rect).width + (rect).height);


#define HV_SUM_OFFSETS( p0, p1, p2, p3, rect, step )                      \
    /* (x, y) */                                                          \
    (p0) = (rect).x + (step) * (rect).y;                                  \
    /* (x + w, y) */                                                      \
    (p1) = (rect).x + (rect).width + (step) * (rect).y;                   \
    /* (x, y + h) */                                                      \
    (p2) = (rect).x + (step) * ((rect).y + (rect).height);                \
    /* (x + w, y + h) */                                                  \
    (p3) = (rect).x + (rect).width + (step) * ((rect).y + (rect).height);

HvStumpClassifier* CreateStumpClassifier();
void ReleaseStumpClassifier( HvStumpClassifier** Stumpclassifier );

HvRealStumpClassifier* CreateRealStumpClassifier();
void ReleaseRealStumpClassifier( HvRealStumpClassifier** Stumpclassifier );

int FindStumpThreshold_sq_16s(                                             
        uchar* data,		// valcache의 pointer (특정 feature에 대해서)
		size_t datastep,	// 4 (float)                                         
        uchar* wdata,		// weight의 pointer
		size_t wstep,       // 4 (float)                                   
        uchar* ydata,		// class의 pointer (1.0 or -1.0)
		size_t ystep,		// 4 (float)                                          
        uchar* idxdata,		// sorted index data의 pointer (특정 feature에 대해서)
		size_t idxstep,		// 2 (short)
		int num,			// 200 : m = image 수                           
        float* lerror,		// float* (output)                                                               
        float* rerror,		// float* (output)                                                             
        float* threshold,	// float* (output)
		float* left,		// float* (output)
		float* right,		// float* (output)                             
        float* sumw,		// float* (output)
		float* sumwy,		// float* (output)
		float* sumwyy );	// float* (output)                           


int FindStumpThreshold_sq_function_32int(                                            
        uchar* data, 
		size_t datastep, 
        uchar* wdata,
		size_t wstep,
        uchar* ydata,
		size_t ystep, 
        uchar* idxdata,
		size_t idxstep, 
		int num,                                         
        float* lerror,                                                                   
        float* rerror,                                                                   
        float* threshold, 
		float* left,
		float* right,                                     
        float* sumw,
		float* sumwy,
		float* sumwyy );

int General_FindStumpThreshold_sq_function_32int(                                            
        uchar* data, 
		size_t datastep, 
        uchar* wdata,
		size_t wstep,
        uchar* ydata,
		size_t ystep, 
        uchar* idxdata,
		size_t idxstep, 
		int num,                                         
        float* lerror,                                                                   
        float* rerror,                                                                   
        float* threshold, 
		float* left,
		float* right,                                     
        float* sumw,
		float* sumwy,
		float* sumwyy );
float CalRealAdaBoosting_Epsilon(int nBinNum);

bool FindRealStump_function_32int(                                            
        uchar* data, 
		size_t datastep, 
        uchar* wdata,
		size_t wstep,
        uchar* ydata,
		size_t ystep, 
        uchar* idxdata,
		size_t idxstep, 
		int num,                                         
        float* fError,
		int nBinNum,
        float* fMin,
		float* fBinWidth_Inv,
		float* fVal
		);

extern bool bDebug;

bool CLEAR_DEBUGLOG();
bool WRITE_DEBUGLOG(char*  sInfo);
bool WRITE_DEBUGLOG(char*  sInfo, int data);
bool WRITE_DEBUGLOG(char*  sInfo, float data);
bool WRITE_DEBUGLOG(char*  sInfo, double data);

void ConvertToFastFeature( HvFeature* feature,
                           HvFastFeature* fastFeature,
                           int size, int step,
						   double dScale=1.0);

void ConvertToFastFeature( HvFeature* feature,
                           HvFastFeature* fastFeature,
                           int size, int step,
						   int x, int y,
						   int dScale);
void GetAuxImages( CvMat* img, CvMat* sum, CvMat* tilted,
                   CvMat* sqsum, float *mean, float* normfactor,
				   int nOrignalWidth=24, int nOrignalHeight=24);

void biGammaCorrection(IplImage* IPl_Image);

#endif
