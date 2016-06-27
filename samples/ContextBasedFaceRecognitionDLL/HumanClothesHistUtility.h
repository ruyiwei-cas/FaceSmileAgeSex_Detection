#ifndef  HUMAN_CLOTHES_HIST_UTILITY_H
#define HUMAN_CLOTHES_HIST_UTILITY_H


int HumanClothesHist_Generation(char *sImgList);

extern "C" __declspec(dllexport)
void GetClothesFeature(IplImage *ColorImage, Face_Attribute *DetectionResult);

extern "C" __declspec(dllexport)
void Init_ClothesFeature();

extern "C" __declspec(dllexport)
int PredictID_With_Clothes_matching(double *Hist, char *sDate, double* dError, bool bSameDay);

double Histogram_matching(double* Hist1, double *Hist2, int nLength);
int GetFileDate(char *sFilename, char *sDate);

extern "C" __declspec(dllexport)
int ContextFeature_Collection(char *sFilename);

extern "C" __declspec(dllexport)
void ContextFeature_Generation(int nMAX_PersonID);
#endif