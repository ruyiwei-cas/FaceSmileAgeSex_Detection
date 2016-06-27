/*
 * HvTrain.h
 *
 * training of cascade of boosted classifiers based on haar features
 */

#ifndef __HVTRAIN_H_
#define __HVTRAIN_H_

#include "Const.h"
//////////////////// 
#include <opencv\highgui.h>

#include <math.h>
#include <time.h>
#include <sys/stat.h>
#ifdef WIN32
	#include <direct.h>
#else
	#include<dirent.h>
#endif

#include <float.h>
////////////////////////////////

#define STAGE_CART_FILE_NAME "AdaBoostCARTHaarClassifier.txt"

#define HV_FEATURE_MAX      3
#define HV_FEATURE_DESC_MAX 20



#ifndef PATH_MAX
#define PATH_MAX 512
#endif /* PATH_MAX */


#define FACE_NORM_MIN  (100/484)

//typedef int    sum_type;
//typedef double sqsum_type;
//typedef short  idx_type;

#define HV_SUM_MAT_TYPE CV_32SC1
#define HV_SQSUM_MAT_TYPE CV_64FC1
//#define HV_IDX_MAT_TYPE CV_16SC1
#define HV_IDX_MAT_TYPE CV_32SC1
#define HV_THRESHOLD_EPS (0.00001F)

typedef struct HvValArray 
{
    uchar* data;
    size_t step;
} HvValArray;

typedef struct HvFeature
{
    char desc[HV_FEATURE_DESC_MAX];
    int  tilted;
    struct
    {
        CvRect r;
        float weight;
    } rect[HV_FEATURE_MAX];
} HvFeature;

typedef struct HvFastFeature
{
    int tilted;
    struct
    {
        int p0, p1, p2, p3;
        float weight;
    } rect[HV_FEATURE_MAX];
} HvFastFeature;

typedef struct HvIntFeatures
{
    CvSize winsize;
    int count;
    HvFeature* feature;
    HvFastFeature* fastfeature;
} HvIntFeatures;

/* Prepared for training samples */
typedef struct HvTrainingData
{
    CvSize winsize;     /* training image size */
    int    maxnum;      /* maximum number of samples */
    CvMat  sum;         /* sum images (each row represents image) */
    CvMat  tilted;      /* tilted sum images (each row represents image) */
    CvMat  normfactor;  /* normalization factor */
	CvMat  mean;		/* mean value array */
    CvMat  cls;         /* classes. 1.0 - object, 0.0 - background */
    CvMat  weights;     /* weights */

    CvMat* valcache;    /* precalculated feature values (CV_32FC1) */
    CvMat* idxcache;    /* presorted indices (HV_IDX_MAT_TYPE) */
} HvTrainingData;

/* Passed to callback functions */
typedef struct HvUserData
{
    HvTrainingData* trainingData;
    HvIntFeatures* haarFeatures;
} HvUserData;

typedef struct HvCARTClassifier
{
    int flags;                                                                           
    int count;	// number of internal nodes

    // internal nodes (each array of <count> elements)
    int* compidx;
    float* threshold;
    int* left;
    int* right;
    
    float* val;	// leaves (array of <count>+1 elements) 
} HvCARTClassifier;

typedef struct HvBackgroundData
{
    int    count;
    char** filename;
    int    last;
    int    round;
    CvSize winsize;
} HvBackgroundData;

typedef struct HvBackgroundReader
{
    CvMat   src;
    CvMat   img;
    CvPoint offset;
    float   scale;
    float   scalefactor;
    float   stepfactor;
    CvPoint point;
} HvBackgroundReader;

typedef struct HvStumpTrainParams
{
    int portion; /* number of components calculated in each thread */
    int numcomp; /* total number of components */
    
    CvMat* sortedIdx; /* presorted samples indices */
    HvUserData* userdata; /* passed to callback */
} HvStumpTrainParams;

typedef struct HvStumpClassifier
{
    int flags;                                                                           
    int compidx;
    
    float lerror; /* impurity of the right node */
    float rerror; /* impurity of the left  node */
    
    float threshold;
    float left;
    float right;
} HvStumpClassifier;

typedef struct HvRealStumpClassifier
{
    int flags;                                                                           
    int compidx;
    
    float fError; 
    
	int nBinNum;
    float fMin;
	float fBinWidth_Inv;
	float fVal[MAX_REAL_CLASSIFIER_BIN];
} HvRealStumpClassifier;

// internal structure used in CART creation 
typedef struct HvCARTNode
{
    CvMat* sampleIdx;
    HvStumpClassifier* stump;
    int parent;
    int leftflag;
    float errdrop;
} HvCARTNode;

typedef struct HvTrainParams
{
// desired number of internal nodes 
	int count;
    HvStumpTrainParams* stumpTrainParams;
    HvUserData* userdata;
} HvTrainParams;

typedef struct HvVecFile
{
    FILE*  input;
    int    count;
    int    vecsize;
    int    last;
    short* vector;
} HvVecFile;


//	#define MAX(a,b)	(((a) > (b)) ? (a) : (b))

#define IMIN(a, b)  ((a) ^ (((a)^(b)) & (((a) < (b)) - 1)))
#define calc_sum(rect, offset) \
	((rect).p0[offset] - (rect).p1[offset] - (rect).p2[offset] + (rect).p3[offset])

#endif /* __HVTRAIN_H_ */
