
#if !defined(_Const_H_)
#define _Const_H_

#define SAMPLE_NUM    3000
#define POSE_ESTIMATER_NODE_MAX_STAGE_NUM  7
#define POSE_VERIFIER_NODE_MAX_STAGE_NUM  7

#define MAX_BACKGROUND_SAMPLE_NUM  (VIEW_NUM*SAMPLE_NUM)

#define  VIEW_NUM      7
#define  PRE_CACULATE_FEATURE_NUM   10000
#define  WIN_WIDTH  24
#define  WIN_HEIGHT  WIN_WIDTH
#define  MEM_SIZE  600

#define MAX_STAGE_NUM  25

#define  MAX_REAL_CLASSIFIER_BIN  64
#define  BIN_DELTA            (0.01/MAX_REAL_CLASSIFIER_BIN)  

/*
#define  dMinMean  (91.338844)
#define dMaxMean   (238.861572)
#define dMinNorm   (6.734526*2)
#define dMaxNorm   (91.079849)
*/
#define  dMinMean  (91.338844*0.3)
#define dMaxMean   (238.861572)
#define dMinNorm   (6.734526 * 2)
#define dMaxNorm   (91.079849)

#endif
