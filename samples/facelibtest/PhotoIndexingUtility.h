#ifndef  PHOTO_INDEXING_UTILITY_H
#define PHOTO_INDEXING_UTILITY_H


void PhotoIndexing_Init();
void PhotoIndexing_Release();
int PhotoIndexing_ImageEvaluation(char *sFilename, int *scene_label);
IplImage *ReadImage_Withexif(char *sFilename);

#endif