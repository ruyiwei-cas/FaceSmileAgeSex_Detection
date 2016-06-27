
extern "C" __declspec(dllexport)
void InitProfileFaceDetector();

extern "C" __declspec(dllexport)
void ReleaseProfileFaceDetector();


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



extern "C" __declspec(dllexport)
int WriteProfilingFile(char *lpFilename, int nFace_Num, Face_Attribute *FaceRecognitionResult, int *Face_Valid_Flag);

extern "C" __declspec(dllexport)
int LoadProfilingFile(char *lpFilename, Face_Attribute* FaceRecognitionResult, int *Face_Valid_Flag);

extern "C" __declspec(dllexport)
void Init_ClothesFeature();


extern "C" __declspec(dllexport)
void FaceID_PostProcessing(int *Result_FaceID, float *Result_Prob_FaceID, int *Face_Valid_Flag, int nface_num);

extern "C" __declspec(dllexport)
void FaceID_PostProcessing2(Face_Attribute *FaceRecognitionResult, int *Face_Valid_Flag, int nface_num);

