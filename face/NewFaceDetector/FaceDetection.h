//FaceDetection.h: interface for theFaceDetection class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_HVFACEDETECTION_H__D4090768_D15C_40CA_8A89_FF726EAF7833__INCLUDED_)
#define AFX_HVFACEDETECTION_H__D4090768_D15C_40CA_8A89_FF726EAF7833__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include "FD_Data.h"
#include "HvChainCascadeClassifier.h"
#include "FaceSkinImage.h"

#define MAX_FACE_CANDIDATE_NUMBER 10000

typedef struct _CComp 
{
		FdRect	rect;
		int		neighbors;
		int		eyeLocation; // 0 = left, 1 = right
		float	view;
		double  confidence;
		~_CComp(){
//			printf("destructor of CComp\n");
		}
} CComp;

class FaceDetection  
{
public:
	// image size
	int m_nImageWidth,m_nImageHeight;
	int m_nImageScale;
	double m_dDetectionScale;
	// image buffer
	unsigned char *m_lpImage; 


	int *s;
	int *sum_image;
	int *tilt_image;
	double *sqsum_image;
	int *sq;
	FdSize orisize;
	FdSize size;
	FdSize sumsize;

	int size_space;
	int sumsize_space;
	//CCompArray m_Seq;
	CComp *m_Seq[MAX_FACE_CANDIDATE_NUMBER];
	int m_nSeq_Candidate_Num;

	// window
	int m_nWinWidth, m_nWinHeight;
	double m_dWinScale;
	
	// frontal view detector
	HvChainCascadeClassifier *m_CascadeClassifier;

	// computation analysis
	int m_nTotalWinNum;
	int m_nAdaCompute_Num;

	CFaceSkinImage m_FaceSkinImage;
	bool m_bWithSkinColor;

	double m_dCentralRatio;
	bool m_bROI_CenteralRatio_Set;
public:
	// tracking related
	bool m_bROI_Set;
	CvRect m_ImageROI;
	void SetImageROI(CvRect ImageROI)
	{
		m_ImageROI = ImageROI;
		m_bROI_Set = true;
	}
	void SetImageROI(double dCenterRatio)
	{
		double dOffsetRatio = (1 - dCenterRatio) * 0.3;
		m_ImageROI.x = int(m_nImageWidth * dOffsetRatio);
		m_ImageROI.y = int(m_nImageHeight * dOffsetRatio);
		m_ImageROI.width = int(m_nImageWidth*dCenterRatio);
		m_ImageROI.height = int(m_nImageHeight*dCenterRatio);

		m_bROI_Set = true;
	}

	void ClearImageROI()
	{
		m_bROI_Set = false;
		m_bROI_CenteralRatio_Set = false;
	}
	
	// face size range
	bool m_bFaceSizeRange_Set;
	int m_nMinSize, m_nMaxSize; // width = height = size
	void SetFaceSizeRange(int nMinSize, int nMaxSize)
	{
		m_bFaceSizeRange_Set = true;
		m_nMinSize = nMinSize;
		m_nMaxSize = nMaxSize;
	}
	void ClearFaceSizeRange()
	{
		m_bFaceSizeRange_Set = false;
	}
public:
	int DetectFace(IplImage *lpImage, unsigned short* nDepthImage, FdAvgComp *faces);
	int DetectFace(IplImage *lpImage, unsigned char* lpSkinImage, FdAvgComp *faces, int parallel_flag);
	int DetectFacePyramid(IplImage *lpImage, FdAvgComp *faces);
	int DetectFace_Partial(IplImage *lpImage, unsigned char* lpSkinImage, FdAvgComp *faces,
		int nTotal_Part_Num, int nCurrentPartNo);

 	float VerifytFace(IplImage *lpImage, FdRect face);
	void Init(int nWinWidth, int nWinHeight, double dWinScale);
	void Cal_IntegralImage();
	/* 2015-5-18, add two sub task of calculate integral image */
	void Cal_IntegralImage_one();
	void Cal_IntegralImage_two();

	void Release();
	void SetParameter(int width, int height, int pic_scale);
	void MemoryAllocation();

	FaceDetection(){};
	FaceDetection(int nWinWidth, int nWinHeight);
	virtual ~FaceDetection();
	int  Ren_MergeCandidates(   FdAvgComp *faces, int min_neighbors);
	int DetectCandidates(FdSize winsize, int ystep, int xstep);
	int DetectCandidates(FdSize winsize, int ystep, int xstep, double dScale);
	void Initialize(int width, int height, int flag);

	bool LoadDefaultFrontalDetector(const unsigned short* nFaceDetector_Int, const double *nFaceDetector_double);

};

class CPoint_Ren
{
public:
	int x;
	int y;
};

#endif 
// !defined(AFX_HVFACEDETECTION_H__D4090768_D15C_40CA_8A89_FF726EAF7833__INCLUDED_)
