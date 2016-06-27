#include <stdio.h>
#include <stdlib.h>

#include "opencv/cv.h"
#include "opencv/highgui.h"
#include <opencv2/opencv.hpp> 
#include <opencv\cxcore.h>
#include "libexif/exif-data.h"  //for libexif library

using namespace std;
using namespace cv;

#include "../Platform.h"

#ifdef __win__
#include <afx.h>
#endif
#include <io.h>
#include "MyType.h"

#include "HumanClothesHistUtility.h"
#include "FaceRecognitionUtility.h"
#include "ContextBasedFaceRecognitionDLL.h"

extern int Face_Valid_Flag[MAX_FACE_NUMBER];
extern Face_Attribute FaceRecognitionResult[MAX_FACE_NUMBER];
extern int nFaceSetSize;

////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the human clothes histogram processing
///  written by Ren Haibing 
/// Acknowledge		: None
///
/// Function name	: GetFileDate
/// Description	    : get file create date 
///
/// Argument		:	sFilename -- file name
/// Argument		:	sDate -- returned date
///
/// Return type		: 
///
/// Create Time		: 2014-11-19  12:46
///
///
/// Side Effect		: 
///                   
///////////////////////////////////////////////////////////////////////////////////////////////
int GetFileDate(char *sFilename, char *sDate)
{
	ExifData *imgExif;
	ExifByteOrder byteOrder;
	ExifEntry *imgExifEntry;

	imgExif = exif_data_new_from_file(sFilename);
	if (imgExif)
	{
		byteOrder = exif_data_get_byte_order(imgExif);
		imgExifEntry = exif_data_get_entry(imgExif, EXIF_TAG_DATE_TIME);
		if (imgExifEntry)
		{
			char *sCurrentDate = (char *)(imgExifEntry->data);
			memcpy(sDate, sCurrentDate, 11);
		}
		else
		{
			exif_data_free(imgExif);
			return -1;
		}

	}
	else return -1;

	if (imgExif)
		exif_data_free(imgExif);
	return 1;
}


int HumanClothesHist_Generation(char *sImgList)
{
	FILE *fpImgList = fopen(sImgList, "rt");
	if (fpImgList == nullptr)
	{
		cout << sImgList << "  doesn't exist" << endl;
		return -1;
	}

	//read image list
	char sPath[1024];
	int imgNo = 0;

	// Step 1: get all the clothes histogram (frontal view + family member)
	while (fgets(sPath, 1024, fpImgList))
	{
		cout << "---------------------------------------------------------" << endl;

		if (sPath[strlen(sPath) - 1] == '\n') //remove the end \n in file name, or it will cause error in reading file (e.g. libexif).
			sPath[strlen(sPath) - 1] = '\0';

		//--------------------------------------------------------------------------

		imgNo++;
		printf(" %d\n", imgNo);
		int nResult = ContextFeature_Collection(sPath);
		if (nResult < 0)
			printf("No Date information!\n");

	}//while(!feof(fpImgList))
	fclose(fpImgList);

	// step 2: generate the histogram for each day * each person

	ContextFeature_Generation(nFaceSetSize - 1);
	return 0;
}

