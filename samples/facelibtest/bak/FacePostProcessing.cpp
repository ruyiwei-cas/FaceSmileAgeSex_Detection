#include "../Platform.h"

#ifdef __win__
#include <afx.h>
#endif
#include "MyType.h"
#include "FaceRecognitionUtility.h"


////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the Face post processing
///  written by Ren Haibing 
/// Acknowledge		: None
///
/// Function name	: FaceID_PostProcessing
/// Description	    : reject the imposters
///
/// Argument		:	Result_FaceID -- recognition result
/// Argument		:	Result_Prob_FaceID -- recognition confidence measurement
/// Argument		:	Face_Valid_Flag -- valid face flag
/// Argument		:	nface_num -- detected face number
///
/// Return type		: 
///
/// Create Time		: 2014-10-31  10:01
///
///
/// Side Effect		: 
///                   
///////////////////////////////////////////////////////////////////////////////////////////////
void FaceID_PostProcessing(int *Result_FaceID, float *Result_Prob_FaceID, int *Face_Valid_Flag, int nface_num)
{
	int FaceID_Array[MAX_FACE_ID];
	float FaceID_MaxProb[MAX_FACE_ID];
	int i;

	for(i=0;i<MAX_FACE_ID;i++)
	{
		FaceID_Array[i] = 0;
		FaceID_MaxProb[i] = 0.0f;
	}
	int nFaceID;
	//1. remove the result when confidence < threshold
	for(i=0;i<nface_num;i++)
	{
		if(Face_Valid_Flag[i]!= 1)
			continue;
		if(Result_Prob_FaceID[i]<FACE_ID_THRESHOLD)
			Result_FaceID[i] = -1;  //face name is set as "N/A"
	}

	//2. check if 2 face images have the same ID
	for(i=0;i<nface_num;i++)
	{
		if(Face_Valid_Flag[i]!= 1)
			continue;
		nFaceID = Result_FaceID[i];
		FaceID_Array[nFaceID]++;
		if(FaceID_MaxProb[nFaceID] < Result_Prob_FaceID[i])
			FaceID_MaxProb[nFaceID] = Result_Prob_FaceID[i];
	}

	for(i=0;i<nface_num;i++)
	{
		if(Face_Valid_Flag[i]!= 1)
			continue;
		nFaceID = Result_FaceID[i];
		if(FaceID_Array[nFaceID]>1) // more than 1 images have the same ID
		{
			if(Result_Prob_FaceID[i] < FaceID_MaxProb[nFaceID]-0.001) // only remove the one with small probility
				Result_FaceID[i] = -1;  //face name is set as "N/A"		
		}		
	}
}
////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the Face post processing
///  written by Ren Haibing 
/// Acknowledge		: None
///
/// Function name	: FaceID_PostProcessing
/// Description	    : reject the imposters
///
/// Argument		:	FaceRecognitionResult -- recognition result
/// Argument		:	Face_Valid_Flag -- valid face flag
/// Argument		:	nface_num -- detected face number
///
/// Return type		: 
///
/// Create Time		: 2014-11-4  13:01
///
///
/// Side Effect		: 
///                   
///////////////////////////////////////////////////////////////////////////////////////////////
void FaceID_PostProcessing(Human_Attribute *FaceRecognitionResult, int *Face_Valid_Flag, int nface_num)
{
	int FaceID_Array[MAX_FACE_ID];
	float FaceID_MaxProb[MAX_FACE_ID];
	int i;

	for(i=0;i<MAX_FACE_ID;i++)
	{
		FaceID_Array[i] = 0;
		FaceID_MaxProb[i] = 0.0f;
	}
	int nFaceID;

	//2. check if 2 face images have the same ID
	for(i=0;i<nface_num;i++)
	{
		if(Face_Valid_Flag[i]!= 1)
			continue;
		nFaceID = FaceRecognitionResult[i].FaceID;
		FaceID_Array[nFaceID]++;
		if(FaceID_MaxProb[nFaceID] < FaceRecognitionResult[i].Prob_FaceID)
			FaceID_MaxProb[nFaceID] = FaceRecognitionResult[i].Prob_FaceID;
	}

	for(i=0;i<nface_num;i++)
	{
		if(Face_Valid_Flag[i]!= 1)
			continue;
		nFaceID = FaceRecognitionResult[i].FaceID;
		if(nFaceID<0) continue;
		if(FaceID_Array[nFaceID]>1) // more than 1 images have the same ID
		{
			if(FaceRecognitionResult[i].Prob_FaceID < FaceID_MaxProb[nFaceID]-0.001) // only remove the one with small probility
				FaceRecognitionResult[i].FaceID = -1;  //face name is set as "N/A"		
		}		
	}

	//1. remove the result when confidence < threshold
	for(i=0;i<nface_num;i++)
	{
		if(Face_Valid_Flag[i]!= 1)
			continue;
		if(FaceRecognitionResult[i].Prob_FaceID<FACE_ID_THRESHOLD)
			FaceRecognitionResult[i].FaceID = -1;  //face name is set as "N/A"
	}

}