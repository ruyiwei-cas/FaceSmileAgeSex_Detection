#ifndef  HUMAN_CLOTHES_HIST_UTILITY_H
#define HUMAN_CLOTHES_HIST_UTILITY_H


int HumanClothesHist_Generation(char *sImgList);
void GetClothesHist(IplImage *ColorImage, Human_Attribute *DetectionResult, char *sBodyImageFilename=NULL) ;
void Init_ClothesHist();
int PredictID_With_Histogram_matching(double *Hist, char *sDate, double* dError, bool bSameDay);

double Histogram_matching(double* Hist1, double *Hist2, int nLength);
int GetFileDate(char *sFilename, char *sDate);
#endif