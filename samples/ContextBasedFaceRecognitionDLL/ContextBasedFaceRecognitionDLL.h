
extern "C" __declspec(dllexport)
void InitProfileFaceDetector();

extern "C" __declspec(dllexport)
int DetectProfileFace(IplImage* gray_image, Rect *FaceRegion, Rect *FrontalFaceRegion, int nFrontalFaceNum);

extern "C" __declspec(dllexport)
void GetClothesFeature(IplImage *ColorImage, Face_Attribute *DetectionResult);

extern "C" __declspec(dllexport)
int PredictID_With_Clothes_matching(double *Hist, char *sDate, double* dScore, bool bSameDay);

extern "C" __declspec(dllexport)
int ContextFeature_Collection(char *sFilename);

extern "C" __declspec(dllexport)
void ContextFeature_Generation(int nMAX_PersonID);