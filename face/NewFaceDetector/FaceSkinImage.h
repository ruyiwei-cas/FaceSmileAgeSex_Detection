#pragma once

#include "FD_Data.h"

class CFaceSkinImage
{
public:
	double m_dScale;
public:
	unsigned char *m_lpSkinImage;
	int m_nWidth;  
	int m_nHeight;

	unsigned char *m_lpBinImage;

	FdSize orisize;
	//FdSize size;
	FdSize sumsize;

	int m_nWinSize;

	int *Sum_BinImage;
	int *m_Binp0, *m_Binp1, *m_Binp2, *m_Binp3;
	int m_nPixelNumberThreshold;

public:
	CFaceSkinImage(void);
	~CFaceSkinImage(void);
	void Skin2Binary(void);
	int Init(unsigned char *lpSkinImage, int nWidth, int nHeight);
	
	int Get_FaceSkin_Sum(int nStart_x, int nStart_y);
	void FaceSkin_SetWinSize(int nWinSize);
	bool FaceSkin_Verification(int nStart_x, int nStart_y);
private:
	void CalcualteIntegralImages();
	int MemoryAllocation(void);
	void MemoryRelease();
	int SkinImage_DownSampling(unsigned char *lpInputSkinImage,int nWidth, int nHeight);
};

