// FaceSkinImage.cpp: implementation of the CFaceSkinImage class.
//
//////////////////////////////////////////////////////////////////////

#include "FaceSkinImage.h"
#include <string.h>
#include <assert.h>
#include <stdlib.h>

#define INVALID_Skin_THRESHOLD 100


CFaceSkinImage::CFaceSkinImage(void)
{
	m_dScale = 0.5;

	m_lpSkinImage = nullptr;
	m_lpBinImage = nullptr;
	m_nWidth = 0;
	m_nHeight = 0;

	Sum_BinImage = nullptr;
}


CFaceSkinImage::~CFaceSkinImage(void)
{
	MemoryRelease();
}
////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the CFaceSkinImage Class
///  written by Ren Haibing 
/// Acknowledge		: None
///
/// Function name	: SkinImage_DownSampling
/// Description	    : down sampling the input Skin image with NN method
///
/// Argument		: lpSkinImage -- the input Skin data
/// Argument		: 
///
/// Return type		: 
///
/// Create Time		: 2015-2-10  10:11
///
///
/// Side Effect		: 
///
///////////////////////////////////////////////////////////////////////////////////////////////
int CFaceSkinImage::SkinImage_DownSampling(unsigned char *lpInputSkinImage, int nWidth, int nHeight)
{
	double dSclae_Inv =  1.0 / m_dScale;
	int nOffset = 0;
	int x,y;
	int nInput_x, nInput_y;
	for(y=0;y<m_nHeight;y++)
	{
		nInput_y = int(y * dSclae_Inv);
		for(x=0;x<m_nWidth;x++,nOffset++)
		{
			nInput_x = int(x * dSclae_Inv);
			m_lpSkinImage[nOffset] =lpInputSkinImage[nInput_y * nWidth + nInput_x];
		}
	}
	return 0;
}

////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the CFaceSkinImage Class
///  written by Ren Haibing 
/// Acknowledge		: None
///
/// Function name	: Init
/// Description	    : init the CFaceSkinImage Class with the input
///
/// Argument		: lpSkinImage -- the input Skin data
/// Argument		: nWidth, nHeight -- image size
///
/// Return type		: 
///
/// Create Time		: 2015-1-19  10:05
///
///
/// Side Effect		: 
///
///////////////////////////////////////////////////////////////////////////////////////////////
int CFaceSkinImage::Init(unsigned char *lpSkinImage, int nWidth, int nHeight)
{
	if(lpSkinImage == nullptr)
	{
		MemoryRelease();
		return 0;
	}
	if((m_nWidth != nWidth * m_dScale) | (m_nHeight != nHeight* m_dScale)) 
	{
		MemoryRelease();
		m_nWidth = int(nWidth * m_dScale);
		m_nHeight = int(nHeight * m_dScale);
		MemoryAllocation();
	}
		
	SkinImage_DownSampling(lpSkinImage, nWidth, nHeight);
	Skin2Binary();
	CalcualteIntegralImages();
	return 1;
}
int CFaceSkinImage::MemoryAllocation(void)
{
	m_lpBinImage = (unsigned char *)malloc(m_nHeight*m_nWidth);
	assert(m_lpBinImage);

	m_lpSkinImage = (unsigned char *)malloc(m_nHeight*m_nWidth*sizeof(unsigned char));
	assert(m_lpSkinImage);

	orisize = fdSize(m_nWidth, m_nHeight);
	sumsize = fdSize(orisize.width + 1, orisize.height + 1 );

	int size_space = sumsize.width * sumsize.height;
	int sumsize_space = sumsize.width * sumsize.height;

	Sum_BinImage = (int *)malloc(sizeof(int)* sumsize_space);
	memset(Sum_BinImage, 0, sizeof(int)*sumsize_space);

	return 0;
}
void CFaceSkinImage::MemoryRelease(void)
{
	m_nWidth = 0;
	m_nHeight = 0;
	if(m_lpBinImage != nullptr) 
	{
		free(m_lpBinImage);
		m_lpBinImage = nullptr;
	}

	if(m_lpSkinImage != nullptr) 
	{
		free(m_lpSkinImage);
		m_lpSkinImage = nullptr;
	}

	if(Sum_BinImage != nullptr) 
	{
		free(Sum_BinImage);
		Sum_BinImage = nullptr;
	}
}
////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the CFaceSkinImage Class
///  written by Ren Haibing 
/// Acknowledge		: None
///
/// Function name	: Skin2Binary
/// Description	    : get the binary image (valid Skin pixel 1; invlid 0);
///
/// Argument		: 
///
/// Return type		: 
///
/// Create Time		: 2015-1-19  15:39
///
///
/// Side Effect		: 
///
///////////////////////////////////////////////////////////////////////////////////////////////
void CFaceSkinImage::Skin2Binary(void)
{
	int i;
	for(i=0;i<m_nWidth*m_nHeight;i++)
		if(m_lpSkinImage[i] <INVALID_Skin_THRESHOLD) 
			m_lpBinImage[i] = 0;
		else m_lpBinImage[i] = 1;

	//	SaveBinImage("c:/temp/bin.bmp");
}
////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the CFaceSkinImage Class
///  written by Ren Haibing 
/// Acknowledge		: None
///
/// Function name	: CalcualteIntegralImages
/// Description	    : calculate the integral images for Skin and bin images
///
/// Argument		: 
///
/// Return type		: 
///
/// Create Time		: 2015-1-19  16:21
///
///
/// Side Effect		: 
///
///////////////////////////////////////////////////////////////////////////////////////////////
void CFaceSkinImage::CalcualteIntegralImages(void)
{
	int size_space = sumsize.width * sumsize.height;
	int sumsize_space = sumsize.width * sumsize.height;

	int *Bin_s = (int *)malloc(sizeof(int)* sumsize.width);
//	int Bin_s[1024];
	memset(Bin_s, 0, sizeof(int)*sumsize.width);
	Bin_s[0] = 0;

	int i,j;
	for(j=0 ; j<m_nHeight ; j++)
		for(i=0 ; i<m_nWidth ; i++)
		{
			Bin_s[i+1] = Bin_s[i] + m_lpBinImage[j*m_nWidth+i];
			Sum_BinImage[((j+1)*sumsize.width) + (i+1)] = Sum_BinImage[j*sumsize.width + (i+1)] + Bin_s[i+1];
		}

	free(Bin_s);
}
////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the CFaceSkinImage Class
///  written by Ren Haibing 
/// Acknowledge		: None
///
/// Function name	: FaceSkin_SetWinSize
/// Description	    : given window size, caculate the window pointer offsets
///
/// Argument		: 
///
/// Return type		: 
///
/// Create Time		: 2015-1-20  13:30
///
///
/// Side Effect		: 
///
///////////////////////////////////////////////////////////////////////////////////////////////
void CFaceSkinImage::FaceSkin_SetWinSize(int nWinSize)
{
	FdRect 	equ_rect;		

	m_nWinSize = int(nWinSize * m_dScale);

    equ_rect.x = equ_rect.y = int(m_nWinSize * 0.05);							
    equ_rect.width	= int(m_nWinSize * 0.9); 
	equ_rect.height = int(m_nWinSize * 0.9);

	m_nPixelNumberThreshold = int(equ_rect.width * equ_rect.height * 0.4);

 	m_Binp0 = Sum_BinImage + (equ_rect.y * sumsize.width + equ_rect.x);												
	m_Binp1 = Sum_BinImage + (equ_rect.y * sumsize.width + (equ_rect.x + equ_rect.width));							
	m_Binp2 = Sum_BinImage + ((equ_rect.y + equ_rect.height) * sumsize.width + equ_rect.x);							
	m_Binp3 = Sum_BinImage + ((equ_rect.y + equ_rect.height) * sumsize.width + (equ_rect.x + equ_rect.width));	
}
////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the CFaceSkinImage Class
///  written by Ren Haibing 
/// Acknowledge		: None
///
/// Function name	: Get_FaceSkin_ValidPixelNumber
/// Description	    : get the valid pixel number in the window
///
/// Argument		:  (nStart_x, int nStart_y) -- window left-up corner 
/// Argument		:  *nPixelNum -- valid pixel number 
///
/// Return type		:  0 -- reject the window
///
/// Create Time		: 2015-1-20  13:24
///
///
/// Side Effect		: 
///
///////////////////////////////////////////////////////////////////////////////////////////////
bool CFaceSkinImage::FaceSkin_Verification(int nSrcStart_x, int nSrcStart_y)
{
	int nStart_y = int(nSrcStart_y * m_dScale);
	int nStart_x = int(nSrcStart_x * m_dScale);
    int p_offset = nStart_y * sumsize.width + nStart_x;

	/// 1. calculate the mean and variance
	int nPixelNum = m_Binp0[p_offset] - m_Binp1[p_offset] - m_Binp2[p_offset] + m_Binp3[p_offset];

	if(nPixelNum<m_nPixelNumberThreshold)
		return false;
	return true;
}
