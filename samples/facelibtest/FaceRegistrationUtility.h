#ifndef  FACE_REGISTRATION_UTILITY_H
#define FACE_REGISTRATION_UTILITY_H

void FaceRegistration_Init();
void FaceRegistration_AddUser(char *sUserName, IplImage *m_FaceTemplate[FACE_TEMPLATE_MAX_NUM], int nTotalFaceNum);
void FaceRegistration_Release();

int FaceRegistration_DetectFace(char *sImageFilename, char *sDesFaceName, IplImage *m_FaceTemplate[FACE_TEMPLATE_MAX_NUM], int *nTotalNum);

#endif