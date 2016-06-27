

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
#include "ProfileFaceRecognitionUtility.h"


extern int Face_Valid_Flag[MAX_FACE_NUMBER];	
extern Human_Attribute FaceRecognitionResult[MAX_FACE_NUMBER];
char *CLOTHE_HIST_PATH="ClothesHist";


void PersonHist_Generation(vector<ClothesHist> &AllClothesHist, int nPersonID, vector<ClothesHist> &AverageClothesHistArray);

vector<ClothesHist> AllPersonClothesHist[FACE_TEMPLATE_MAX_NUM];
////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the human clothes histogram processing
///  written by Ren Haibing 
/// Acknowledge		: None
///
/// Function name	: DateString_to_CTime
/// Description	    : calculate the body region with face region
///
/// Argument		:	sDate -- date string(YYYY:MM:DD, YYYY_MM_DD)
///
/// Return type		: 
///
/// Create Time		: 2014-11-12  10:44
///
///
/// Side Effect		: 
///                   
///////////////////////////////////////////////////////////////////////////////////////////////
#ifdef __win__
CTime DateString_to_CTime(char *sDate) // date format
{
	char* sDateString = sDate;//YYYY-MM-dd HH:mm:ss
	char year[5];
	memcpy(year, sDateString, 4);
	year[4] = '\0';
	int date_year = atoi(year);

	char month[3];
	memcpy(month, &(sDateString[5]), 2);
	month[2] = '\0';
	int date_month = atoi(month);

	char day[3];
	memcpy(day, &(sDateString[8]), 2);
	day[2] = '\0';
	int date_day = atoi(day);
	CTime t(date_year, date_month, date_day, 0, 0, 0);
	return t;
}
#endif

#ifdef __linux__
tm DateString_to_CTime(char *sDate) // date format
{
	char*	DateString = sDate;//YYYY-MM-dd HH:mm:ss
	char year[5];
	memcpy(year, &DateString, 4);
	year[4]='\0';
	int date_year=atoi(year);
	char month[3];
	memcpy(month, &DateString[5],2);
	month[2]='\0';
	int date_month=atoi(month);
	char day[3];
	memcpy(day, &DateString[8],2);
	day[2]='\0';
	int date_day=atoi(day);
	tm ret;
	ret.tm_year=date_year;
	ret.tm_mon=date_month;
	ret.tm_mday=date_day;

	//TODO: release temporary char[]

	return ret;
}
#endif

////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the human clothes histogram processing
///  written by Ren Haibing 
/// Acknowledge		: None
///
/// Function name	: CalculateBodyRegion
/// Description	    : calculate the body region with face region
///
/// Argument		:	ColorImage -- source image
/// Argument		:	FaceRegion -- face region
/// Argument		:	BodyRegion -- body region
///
/// Return type		: 
///
/// Create Time		: 2014-11-05  16:17
///
///
/// Side Effect		: 
///                   
///////////////////////////////////////////////////////////////////////////////////////////////
void CalculateBodyRegion(IplImage *ColorImage, CvRect FaceRegion, CvRect *BodyRegion)
{
	int nBody_Width = int(FaceRegion.width * 1.5);
	int nBody_Height = FaceRegion.width * 2;

	int nBody_x = int(FaceRegion.x - (nBody_Width- FaceRegion.width) *0.5);
	if(nBody_x < 0) 
		nBody_x= 0;

	int nBody_y = int(FaceRegion.y + FaceRegion.height * 1.5);
	if(nBody_y > ColorImage->height-1) 
		nBody_y = ColorImage->height-1;

	if(nBody_x+nBody_Width>ColorImage->width-1)
		nBody_Width = ColorImage->width-1 - nBody_x;

	if(nBody_y+nBody_Height>ColorImage->height-1)
		nBody_Height = ColorImage->height-1 - nBody_y;

	BodyRegion->x = nBody_x;
	BodyRegion->y = nBody_y;
	BodyRegion->width = nBody_Width;
	BodyRegion->height = nBody_Height;
}
////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the human clothes histogram processing
///  written by Ren Haibing 
/// Acknowledge		: None
///
/// Function name	: GetClothesHist
/// Description	    : reject the imposters
///
/// Argument		:	ColorImage -- source image
/// Argument		:	DetectionResult -- face region 
///
/// Return type		: 
///
/// Create Time		: 2014-11-5  16:58
///
///
/// Side Effect		: DetectionResult->ClothesHis will be the result
///                   
///////////////////////////////////////////////////////////////////////////////////////////////
void GetImageHist(IplImage *HSVImg, double *Hist)
{
	double s0 = 180.0/double(HUMAN_CLOTHES_HIST_BIN_NUMBER);
	double s1 = 256.0/double(HUMAN_CLOTHES_HIST_BIN_NUMBER);
	double s2 = 256.0/double(HUMAN_CLOTHES_HIST_BIN_NUMBER);

	int nHist_Dim = HUMAN_CLOTHES_HIST_DIM;
	int nHist_Step = HUMAN_CLOTHES_HIST_BIN_NUMBER*HUMAN_CLOTHES_HIST_BIN_NUMBER;
	for(int i=0;i<nHist_Dim;i++)
		Hist[i] = 0;

	uchar *p;
	int bin_h, bin_s, bin_v;
	for(int i=0;i<HSVImg->height;i++)
	{
		p = (uchar *)(HSVImg->imageData + HSVImg->widthStep * i);
		for(int j=0;j<HSVImg->width;j++)
		{// calcualte the bin no. for each pixel
			bin_h = int(p[0]/s0);
			if(bin_h>=HUMAN_CLOTHES_HIST_BIN_NUMBER)
				bin_h =HUMAN_CLOTHES_HIST_BIN_NUMBER-1;
			bin_s = int(p[1]/s1);
			bin_v = int(p[2]/s2);
			Hist[bin_h *nHist_Step + bin_s * HUMAN_CLOTHES_HIST_BIN_NUMBER + bin_v] ++;		

			p = p+3;
		}
	}
	double t = 0;
	for(int i=0;i<nHist_Dim;i++)
		t += Hist[i];
		
	t = 1.0/t;
	for(int i=0;i<nHist_Dim;i++)
		Hist[i] = Hist[i]*t;
}
void GetClothesHist(IplImage *ColorImage, Human_Attribute *DetectionResult, char *sBodyImageFilename) 
{
	CvRect BodyRegion;
	CalculateBodyRegion(ColorImage,DetectionResult->FaceRegion, &BodyRegion);
	cvSetImageROI(ColorImage, BodyRegion);

	IplImage * BodyImage = cvCreateImage(cvSize(BodyRegion.width, BodyRegion.height), IPL_DEPTH_8U, ColorImage->nChannels);
	cvCopyImage(ColorImage,BodyImage);

	if(sBodyImageFilename != NULL)
		cvSaveImage(sBodyImageFilename, BodyImage);

	IplImage * HSVImg = cvCreateImage(cvSize(BodyRegion.width, BodyRegion.height), IPL_DEPTH_8U, ColorImage->nChannels);

	cvCvtColor(BodyImage,HSVImg,CV_BGR2HSV);	

	GetImageHist(HSVImg, DetectionResult->ClothesHis);
	cvResetImageROI(ColorImage);

	cvReleaseImage(&BodyImage);
	cvReleaseImage(&HSVImg);
}


////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the human clothes histogram processing
///  written by Ren Haibing 
/// Acknowledge		: None
///
/// Function name	: SavePersonCothesHistogram
/// Description	    : Save person clothes histogram
///
/// Argument		:	PersonClothesHist -- person histogram per each day
///
/// Return type		: 
///
/// Create Time		: 2014-11-12  10:23
///
///
/// Side Effect		: 
///                   
///////////////////////////////////////////////////////////////////////////////////////////////
void SavePersonCothesHistogram(vector<ClothesHist> &PersonClothesHist)
{
	int nSize = PersonClothesHist.size();
	if(nSize<=0) return ;

	int nPersonID = PersonClothesHist[0].FaceID;
	char sFolderName[1024];
	char sFilename[1024];
	sprintf(sFolderName,"%s\\%d", CLOTHE_HIST_PATH, nPersonID);
	sprintf(sFilename,"%s\\Index.txt", sFolderName);

	if(_access(sFolderName,06)==-1 )
	{
		char sCommand[1024];
		sprintf(sCommand, "mkdir %s",sFolderName);
		system (sCommand);
	}

	FILE *fIndexFile = fopen(sFilename, "wt");
	fprintf(fIndexFile, "Size:%d\n", nSize);


	FILE *fFile;
	for(int i=0;i<nSize;i++)
	{
		// save index file
#ifdef __win__
		CString sDate = PersonClothesHist[i].Pic_date.Format("%Y_%m_%d");
		char sText[20];
		memset(sText, 0, 20);
		for(int j=0;j<sDate.GetLength();j++)
			sText[j] = sDate.GetBuffer(0)[j];
		fprintf(fIndexFile, "%s\n", sText);
		sprintf(sFilename,"%s\\%s.txt",sFolderName, sText);
#endif
#ifdef __linux__
		char sDate[11];
		strftime (sDate,10,"%Y %m %d",&PersonClothesHist[i].Pic_date);
		sDate[10]='\0';
		fprintf(fIndexFile, "%s\n", sDate);
		sprintf(sFilename,"%s\\%s.txt",sFolderName, sDate);
#endif
		// save histogram file
		fFile = fopen(sFilename, "wt");
		for(int j=0;j<HUMAN_CLOTHES_HIST_DIM;j++)
			fprintf(fFile,"%f\n", PersonClothesHist[i].HSV_Hist[j]);
		fclose(fFile);
	}	
	fclose(fIndexFile);
}
////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the human clothes histogram processing
///  written by Ren Haibing 
/// Acknowledge		: None
///
/// Function name	: LoadPersonCothesHistogram
/// Description	    : load person clothes histogram
///
/// Argument		:	nPersonID -- person ID
/// Argument		:	PersonClothesHist -- person histogram per each day
///
/// Return type		: histogram number
///
/// Create Time		: 2014-11-12  10:23
///
///
/// Side Effect		: 
///                   
///////////////////////////////////////////////////////////////////////////////////////////////
int LoadPersonCothesHistogram(int nPersonID, vector<ClothesHist> &PersonClothesHist)
{
	int nSize;
	char sFolderName[1024];
	char sFilename[1024];
	sprintf(sFolderName,"%s\\%d", CLOTHE_HIST_PATH, nPersonID);

	sprintf(sFilename,"%s\\Index.txt", sFolderName);
	FILE *fIndexFile = fopen(sFilename, "r");
	if(fIndexFile ==NULL)
		return 0;
	int nReadNum = fscanf(fIndexFile, "Size:%d\n", &nSize);
	if(nSize ==0)
	{
		fclose(fIndexFile);
		return nSize;
	}

	FILE *fFile;
	char sDate[1024];
	for(int i=0;i<nSize;i++)
	{
		nReadNum = fscanf(fIndexFile,"%s\n", sDate);
		
		ClothesHist AverageHist;
		AverageHist.FaceID = nPersonID;
		AverageHist.Pic_date = DateString_to_CTime(sDate);

		sprintf(sFilename,"%s\\%s.txt",sFolderName, sDate);

		fFile = fopen(sFilename, "r");
		for(int j=0;j<HUMAN_CLOTHES_HIST_DIM;j++)
			nReadNum = fscanf(fFile,"%lf\n", &(AverageHist.HSV_Hist[j]));
		fclose(fFile);

		PersonClothesHist.push_back(AverageHist);
	}	
	fclose(fIndexFile);
	return nSize;
}
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
			char *sCurrentDate = (char *)( imgExifEntry->data);
			memcpy(sDate, sCurrentDate, 11);
		} 
		else
		{
			exif_data_free(imgExif);
			return -1;
		}
		
    }
	else return -1;

	if(imgExif)
		exif_data_free(imgExif);
	return 1;
}
////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the human clothes histogram processing
///  written by Ren Haibing 
/// Acknowledge		: None
///
/// Function name	: HumanClothesHist_Generation
/// Description	    : generate the human clothes histogram for each person * each day
///
/// Argument		:	ColorImage -- source image
/// Argument		:	DetectionResult -- face region 
///
/// Return type		: 
///
/// Create Time		: 2014-11-6  16:58
///
///
/// Side Effect		: 
///                   
///////////////////////////////////////////////////////////////////////////////////////////////
#include "libexif/exif-data.h"  //for libexif library
#ifdef __win__
#include <afx.h>
#endif

int HumanClothesHist_Generation(char *sImgList  )
{
	FILE *fpImgList   = fopen(sImgList, "rt");
	if (fpImgList == nullptr)
	{
		cout << sImgList << "  doesn't exist" << endl;
		return -1;
	}
	
	//read image list
	char sPath[1024];
	
	int imgNo=0;
	int nFace_Num;

	vector<ClothesHist> AllClothesHist;
	char sDate[12];
	memset(sDate, 0, 12);
	// Step 1: get all the clothes histogram (frontal view + family member)
	int nMAX_PersonID = -1;
	while(fgets(sPath, 1024, fpImgList))
	{
		cout<<"---------------------------------------------------------"<<endl;
	
		if (sPath[strlen(sPath)-1]=='\n') //remove the end \n in file name, or it will cause error in reading file (e.g. libexif).
			sPath[strlen(sPath)-1]='\0';

		int nValidDate = GetFileDate(sPath, sDate);
		if (nValidDate != 1)
		{
			imgNo++;
			printf(" %d\n", imgNo);
			continue;
		}

		char sRecognitionResultFilename[1024]; 
		char sImgPath1[1024]; 
		char tmpStr1[1024]; 
		strcpy(tmpStr1, sPath); 
		char* firstdot1 = strchr(tmpStr1,'.');
		*firstdot1 = NULL;
		strcpy(sImgPath1, tmpStr1);                   
		//output face detection and recongition image
		sprintf(sRecognitionResultFilename, "%sFDFR14.txt", tmpStr1);  
		
		nFace_Num = LoadProfilingFile(sRecognitionResultFilename);
		for(int i=0;i<nFace_Num;i++)
		{
			if ((FaceRecognitionResult[i].FaceView == 0) && (FaceRecognitionResult[i].FaceID >= 0) && (FaceRecognitionResult[i].Prob_FaceID>= 0.25))
			{
				ClothesHist hist;
				hist.FaceID = FaceRecognitionResult[i].FaceID;
				if(FaceRecognitionResult[i].FaceID>nMAX_PersonID)
					nMAX_PersonID = FaceRecognitionResult[i].FaceID;
				
				memcpy(hist.HSV_Hist, FaceRecognitionResult[i].ClothesHis, HUMAN_CLOTHES_HIST_DIM*sizeof(double));
				
				hist.Pic_date = DateString_to_CTime(sDate);
				AllClothesHist.push_back(hist);
			}
		}
		//--------------------------------------------------------------------------

		imgNo++;
		printf(" %d\n", imgNo);

	}//while(!feof(fpImgList))

	// step 2: generate the histogram for each day * each person
	vector<ClothesHist> PersonClothesHist;
	for(int i=0;i<=nMAX_PersonID;i++)
	{
		PersonClothesHist.clear();
		PersonHist_Generation(AllClothesHist, i, PersonClothesHist);

		// step 2.2: save the result
		SavePersonCothesHistogram(PersonClothesHist);
	}

	fclose(fpImgList);

	return 0;
}
////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the human clothes histogram processing
///  written by Ren Haibing 
/// Acknowledge		: None
///
/// Function name	: Exist_CTime
/// Description	    : check whether the input ctime exists in the ctime set
///
/// Argument		:	TimeArray -- ctime set
/// Argument		:	nNumber -- set length 
/// Argument		:	SearchedTime --input ctime variable
///
/// Return type		:  >=0: exit; -1: not exist
///
/// Create Time		: 2014-11-11  15:54
///
///
/// Side Effect		: 
///                   
///////////////////////////////////////////////////////////////////////////////////////////////
#ifdef __win__
int Exist_CTime(CTime *TimeArray, int nNumber, CTime SearchedTime)
{
	if(nNumber==0)
		return -1;

	int nNo = 0;
	int nFound = 0;
	while((nFound==0)  &&(nNo<=nNumber))
	{
		if(TimeArray[nNo] == SearchedTime)
			nFound =1;
		else nNo++;
	}
	if(nFound ==1) return nNo;
	else return -1;
}
#endif
#ifdef __linux__
int Exist_CTime(tm *TimeArray, int nNumber, tm SearchedTime)
{
	if(nNumber==0)
		return -1;

	int nNo = 0;
	int nFound = 0;
	while((nFound==0)  &&(nNo<=nNumber))
	{
//		if(TimeArray[nNo] == SearchedTime)
//			nFound =1;
		tm element=TimeArray[nNo];
		if(element.tm_year==SearchedTime.tm_year && element.tm_mon==SearchedTime.tm_mon && element.tm_mday==SearchedTime.tm_mday){
			nFound = 1;
		}
		else nNo++;
	}
	if(nFound ==1) return nNo;
	else return -1;
}
#endif
////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the human clothes histogram processing
///  written by Ren Haibing 
/// Acknowledge		: None
///
/// Function name	: Sort_CTime
/// Description	    : sort the time array by order 
///
/// Argument		:	TimeArray -- time array
/// Argument		:	nNumber -- length of time array 
///
/// Return type		: 
///
/// Create Time		: 2014-11-12  9:21
///
///
/// Side Effect		: 
///                   
///////////////////////////////////////////////////////////////////////////////////////////////
#ifdef __win__
void Sort_CTime(CTime *TimeArray, int nNumber)
{
	CTime temp;
	for(int i=0;i<nNumber-1;i++)
		for(int j=i+1;j<nNumber;j++)
		{
			if(TimeArray[i]>TimeArray[j])
			{
				temp = TimeArray[i];
				TimeArray[i] = TimeArray[j];
				TimeArray[j] = temp;
			}
		}
}
#endif
#ifdef __linux__
void Sort_CTime(tm *TimeArray, int nNumber)
{
	tm temp;
	for(int i=0;i<nNumber-1;i++)
		for(int j=i+1;j<nNumber;j++)
		{
			int ival=TimeArray[i].tm_year*10000+TimeArray[i].tm_mon*100+TimeArray[i].tm_mday;
			int jval=TimeArray[j].tm_year*10000+TimeArray[j].tm_mon*100+TimeArray[j].tm_mday;//TODO: should use difftime
			//if(TimeArray[i]>TimeArray[j])
			if(ival>jval)
			{
				temp = TimeArray[i];
				TimeArray[i] = TimeArray[j];
				TimeArray[j] = temp;
			}
		}
}
#endif
////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the human clothes histogram processing
///  written by Ren Haibing 
/// Acknowledge		: None
///
/// Function name	: PersonHist_Generation
/// Description	    : generate the human clothes histogram for a person 
///
/// Argument		:	AllClothesHist -- histogram vector
/// Argument		:	nPersonID -- person ID 
/// Argument		:	AverageClothesHistArray -- returned clothes histogram of each day * each person
///
/// Return type		: 
///
/// Create Time		: 2014-11-11  15:54
///
///
/// Side Effect		: 
///                   
///////////////////////////////////////////////////////////////////////////////////////////////
#ifdef __win__
void PersonHist_Generation(vector<ClothesHist> &AllClothesHist, int nPersonID, vector<ClothesHist> &AverageClothesHistArray)
{
	CTime TimeArray[5000];
	int nNumber = 0;
	// 1. add all date to the array
	for(unsigned int i=0;i<AllClothesHist.size();i++)
	{
		if(AllClothesHist[i].FaceID != nPersonID)
			continue;
		int nExist = Exist_CTime(TimeArray,nNumber, AllClothesHist[i].Pic_date);
		if(nExist<0)
		{
			TimeArray[nNumber] = AllClothesHist[i].Pic_date;
			nNumber++;
		}//if(nExist<0)
	}
	// 2. Sort the date
	Sort_CTime(TimeArray, nNumber);

	//3. calculate a average histogram for each date
	int nHist_Number; 
	for(int i=0;i<nNumber;i++)
	{
		ClothesHist AverageHist;
		AverageHist.FaceID = nPersonID;
		AverageHist.Pic_date = TimeArray[i];

		nHist_Number = 0;
		for(int m=0;m<HUMAN_CLOTHES_HIST_DIM;m++)
			AverageHist.HSV_Hist[m] = 0.0f;

		for(unsigned int j=0;j<AllClothesHist.size();j++)
		{
			if(AllClothesHist[j].FaceID != nPersonID)
				continue;
			if(AllClothesHist[j].Pic_date != TimeArray[i])
				continue;
			nHist_Number++;

			for(int m=0;m<HUMAN_CLOTHES_HIST_DIM;m++)
				AverageHist.HSV_Hist[m] += AllClothesHist[j].HSV_Hist[m];
		}//for(int j=0;j<AllClothesHist.size();j++)

		for(int m=0;m<HUMAN_CLOTHES_HIST_DIM;m++)
			AverageHist.HSV_Hist[m] /= nHist_Number;

		AverageClothesHistArray.push_back(AverageHist);
	}
}
#endif
#ifdef __linux__
void PersonHist_Generation(vector<ClothesHist> &AllClothesHist, int nPersonID, vector<ClothesHist> &AverageClothesHistArray)
{

	tm TimeArray[5000];
	int nNumber = 0;
	// 1. add all date to the array
	for(unsigned int i=0;i<AllClothesHist.size();i++)
	{
		if(AllClothesHist[i].FaceID != nPersonID)
			continue;
		int nExist = Exist_CTime(TimeArray,nNumber, AllClothesHist[i].Pic_date);
		if(nExist<0)
		{
			TimeArray[nNumber] = AllClothesHist[i].Pic_date;
			nNumber++;
		}//if(nExist<0)
	}
	// 2. Sort the date
	Sort_CTime(TimeArray, nNumber);
	//3. calculate a average histogram for each date
	int nHist_Number;
	for(int i=0;i<nNumber;i++)
	{
		ClothesHist AverageHist;
		AverageHist.FaceID = nPersonID;
		AverageHist.Pic_date = TimeArray[i];

		nHist_Number = 0;
		for(int m=0;m<HUMAN_CLOTHES_HIST_DIM;m++)
			AverageHist.HSV_Hist[m] = 0.0f;

		for(unsigned int j=0;j<AllClothesHist.size();j++)
		{
			if(AllClothesHist[j].FaceID != nPersonID)
				continue;
			tm picdate=AllClothesHist[j].Pic_date;
			if(picdate.tm_year!=TimeArray[i].tm_year || picdate.tm_mon!=TimeArray[i].tm_mon || picdate.tm_mday!=TimeArray[i].tm_mday){
				continue;
			}
			nHist_Number++;

			for(int m=0;m<HUMAN_CLOTHES_HIST_DIM;m++)
				AverageHist.HSV_Hist[m] += AllClothesHist[j].HSV_Hist[m];
		}//for(int j=0;j<AllClothesHist.size();j++)

		for(int m=0;m<HUMAN_CLOTHES_HIST_DIM;m++)
			AverageHist.HSV_Hist[m] /= nHist_Number;

		AverageClothesHistArray.push_back(AverageHist);
	}
}
#endif

////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the human clothes histogram processing
///  written by Ren Haibing 
/// Acknowledge		: None
///
/// Function name	: Init_ClothesHist
/// Description	    : load all clothes histogram
///
/// Argument		:	
/// Return type		:   
///
/// Create Time		: 2014-11-13  14:38
///
///
/// Side Effect		: 
///                   
///////////////////////////////////////////////////////////////////////////////////////////////
void Init_ClothesHist()
{
	for(int i=0;i<FACE_TEMPLATE_MAX_NUM;i++)
	{
		AllPersonClothesHist[i].clear();
		LoadPersonCothesHistogram(i, AllPersonClothesHist[i]);
	}	
}
////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the human clothes histogram processing
///  written by Ren Haibing 
/// Acknowledge		: None
///
/// Function name	: Histogram_matching
/// Description	    : calcualte the L2 distance between 2 histograms
///
/// Argument		:	Hist1 -- histogram 1
/// Argument		:	Hist2 -- histogram 2
/// Argument		:	nLength -- histogram dimension
///
/// Return type		:   distance
///
/// Create Time		: 2014-11-12  15:50
///
///
/// Side Effect		: 
///                   
///////////////////////////////////////////////////////////////////////////////////////////////
double Histogram_matching(double* Hist1, double *Hist2, int nLength)
{
	int nMethod =1;
	double dError = 0.0;
	double dSimilarity, dDis;
	double dTemp0 = 0.0;
	double dTemp1 = 0.0;
	double dTemp2 = 0.0;

	switch(nMethod)
	{
		case 0:// L2 normal
			dError = 0.0;
			for(int i=0;i<nLength;i++)
			{
				dDis = Hist1[i] - Hist2[i];
				dError += dDis * dDis;
			}
			dError /= nLength;
			dSimilarity = 1-10*sqrt(dError);
			break;
		case 1: // cosine 

			for(int i=0;i<nLength;i++)
			{
				dTemp0 += Hist1[i] * Hist2[i];
				dTemp1 += Hist1[i]* Hist1[i];
				dTemp2 += Hist2[i]* Hist2[i];
			}
			dTemp1 = sqrt(dTemp1);
			dTemp2 = sqrt(dTemp2);
			dSimilarity = dTemp0 /(dTemp1 * dTemp2);
			break;
		default:
			break;
	}
	return dSimilarity;
}

////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the human clothes histogram processing
///  written by Ren Haibing 
/// Acknowledge		: None
///
/// Function name	: PredictID_With_Histogram_matching
/// Description	    : 
///
/// Argument		:	Hist -- input histogram
/// Argument		:	sDate -- input date
/// Argument		:	dError -- minimal matching error
/// Argument		:	bSameDay-- whether same date is required
///
/// Return type		:   person ID
///
/// Create Time		: 2014-11-13  16:40
///
///
/// Side Effect		: 
///                   
///////////////////////////////////////////////////////////////////////////////////////////////
#ifdef __win__
int PredictID_With_Histogram_matching(double *Hist, char *sDate, double* dScore, bool bSameDay)
{
	int nID = -1;
	double dMaxScore = -100000.0f;
	double dCurrentScore;
	int nSize;
	CTime Hist_Time = DateString_to_CTime(sDate);
	bool bFound = false;
	for(int i=0;i<FACE_TEMPLATE_MAX_NUM;i++)
	{
		nSize = AllPersonClothesHist[i].size();
		for(int j=0;j<nSize;j++)
		{
			if((bSameDay) && (AllPersonClothesHist[i][j].Pic_date == Hist_Time))
			{
				dCurrentScore = Histogram_matching(AllPersonClothesHist[i][j].HSV_Hist, Hist,HUMAN_CLOTHES_HIST_DIM);
				if(dCurrentScore>dMaxScore)
				{
					dMaxScore = dCurrentScore;
					nID = i;
					bFound = true;
				}
			}
		}//for(int j=0;j<nSize;j++)
	}//for(int i=0;i<FACE_TEMPLATE_MAX_NUM;i++)
	*dScore = dMaxScore;
	return nID;

}
#endif
#ifdef __linux__
int PredictID_With_Histogram_matching(double *Hist, char *sDate, double* dScore, bool bSameDay)
{
	int nID = -1;
	double dMaxScore = -100000.0f;
	double dCurrentScore;
	int nSize;
	tm Hist_Time = DateString_to_CTime(sDate);
	for(int i=0;i<FACE_TEMPLATE_MAX_NUM;i++)
	{
		nSize = AllPersonClothesHist[i].size();
		for(int j=0;j<nSize;j++)
		{
			tm picdate=AllPersonClothesHist[i][j].Pic_date;
			//if((bSameDay) && (AllPersonClothesHist[i][j].Pic_date == Hist_Time))
			if((bSameDay) && picdate.tm_year==Hist_Time.tm_year && picdate.tm_mon==Hist_Time.tm_mon && picdate.tm_mday==Hist_Time.tm_mday)
			{
				dCurrentScore = Histogram_matching(AllPersonClothesHist[i][j].HSV_Hist, Hist,HUMAN_CLOTHES_HIST_DIM);
				if(dCurrentScore>dMaxScore)
				{
					dMaxScore = dCurrentScore;
					nID = i;
				}
			}
		}//for(int j=0;j<nSize;j++)
	}//for(int i=0;i<FACE_TEMPLATE_MAX_NUM;i++)
	*dScore = dMaxScore;
	return nID;
}
#endif
