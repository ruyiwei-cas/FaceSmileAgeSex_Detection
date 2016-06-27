#ifndef  PHOTO_INDEXING_TYPE_H
#define PHOTO_INDEXING_TYPE_H

#include "MyType_Main.h"
//#include <afx.h>
#include <ctime>//Sirui
typedef struct ClothesHist
{
	//char sDate[11]; // XXXX:XX:XX
#ifdef __linux__	
	tm Pic_date;
#endif

#ifdef __win__	
	CTime Pic_date;
#endif	

	int FaceID;
	double HSV_Hist[HUMAN_CLOTHES_HIST_BIN_NUMBER*HUMAN_CLOTHES_HIST_BIN_NUMBER*HUMAN_CLOTHES_HIST_BIN_NUMBER];
} ClothesHist;

#endif