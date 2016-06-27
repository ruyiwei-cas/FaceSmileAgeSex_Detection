#include "ColorGMM.h"
#define EM_REFINE


CColorGMM::CColorGMM(void)
{
	m_nClusterNum = 8;
	m_nLookupTable_Interval = 4;
	m_nLookupTalbe_Channel_Bin = 256/m_nLookupTable_Interval;
	m_lpLookupTable_Data = (unsigned char *)malloc(m_nLookupTalbe_Channel_Bin*m_nLookupTalbe_Channel_Bin*m_nLookupTalbe_Channel_Bin);
}


CColorGMM::~CColorGMM(void)
{
	if (m_lpLookupTable_Data != nullptr) free(m_lpLookupTable_Data);
}
////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of CColorGMM
///  written by Ren Haibing 
/// Acknowledge		: 
///
/// Function name	: TrainLookupTalbe
/// Description	    : train lookup table 
///
/// Argument		: 
///
/// Return type		:  
///
/// Create Time		: 2015-4-27  14:20
///
///
/// Side Effect		: 
///                    
///////////////////////////////////////////////////////////////////////////////////////////////
int CColorGMM::TrainLookupTalbe(Mat lpTrainingImage)
{
	//Mat lpTrainingImage = imread(sFilename);
	GaussianBlur(lpTrainingImage, lpTrainingImage,  cvSize(5,5),5);
	
	Mat lpImageHSV;
	cvtColor(lpTrainingImage,lpImageHSV, CV_BGR2HSV);

	int nSampleNum = 0;
	int nImageWidth = lpTrainingImage.cols;
	int nImageHeight = lpTrainingImage.rows;
	unsigned char *lpImageBuffer;
	for(int i=0;i<nImageHeight;i++)
		for(int j=0;j<nImageWidth;j++)
		{
			lpImageBuffer = lpImageHSV.ptr<unsigned char >(i,j);
			if((lpImageBuffer[0] >0) || (lpImageBuffer[1] >0) || (lpImageBuffer[2] >0))
				nSampleNum++;
		}

	Mat TrainingSamples(nSampleNum, 3, CV_8UC1);
	int nSampleNo = 0;
	for(int i=0;i<nImageHeight;i++)
		for(int j=0;j<nImageWidth;j++)
		{
			lpImageBuffer = lpImageHSV.ptr<unsigned char >(i,j);
			if((lpImageBuffer[0] >0) || (lpImageBuffer[1] >0) || (lpImageBuffer[2] >0))
			{
				TrainingSamples.ptr<unsigned char >(nSampleNo, 0)[0] = lpImageBuffer[0];
				TrainingSamples.ptr<unsigned char >(nSampleNo, 1)[0] = lpImageBuffer[1];
				TrainingSamples.ptr<unsigned char >(nSampleNo, 2)[0] = lpImageBuffer[2];
				nSampleNo++;
			}
		}
	

	
    m_Em_params.means = NULL;
    m_Em_params.probs = NULL;
    m_Em_params.covs = NULL;
    m_Em_params.weights = NULL;
    m_Em_params.nclusters = m_nClusterNum;
    m_Em_params.start_step = CvEM::START_AUTO_STEP;
    m_Em_params.cov_mat_type = CvEM::COV_MAT_SPHERICAL;
    m_Em_params.term_crit.max_iter = 300;
    m_Em_params.term_crit.epsilon = 0.1;
		

	//cvReshape
	cout << "Begin to estimate the GMM parameter" << endl;
	Mat labels;
#ifdef EM_REFINE
    CvEM Em1;
    Em1.train( TrainingSamples, Mat(), m_Em_params, &labels );

    m_Em_params.cov_mat_type = CvEM::COV_MAT_DIAGONAL;
    m_Em_params.start_step = CvEM::START_E_STEP;
    m_Em_params.means = Em1.get_means();
    m_Em_params.covs = Em1.get_covs();
    m_Em_params.weights = Em1.get_weights();
#endif

    m_Em.train( TrainingSamples, Mat(), m_Em_params, &labels );

	cout << "Begin to calculate the lookup table" << endl;
	int Half_Inteval = 0.5* m_nLookupTable_Interval;
	Mat LookupTable_RGBImage(m_nLookupTalbe_Channel_Bin*m_nLookupTalbe_Channel_Bin, m_nLookupTalbe_Channel_Bin, CV_8UC3);
	for(int i=0;i<m_nLookupTalbe_Channel_Bin;i++)
	{
		cout <<"Lookup table, step 1 :" << i <<"/" <<m_nLookupTalbe_Channel_Bin <<endl;

		for(int j= 0;j<m_nLookupTalbe_Channel_Bin;j++)
			for(int k=0;k<m_nLookupTalbe_Channel_Bin;k++)
			{
				LookupTable_RGBImage.ptr<unsigned char >(i*m_nLookupTalbe_Channel_Bin+j, k)[0] = i *m_nLookupTable_Interval + Half_Inteval;
				LookupTable_RGBImage.ptr<unsigned char >(i*m_nLookupTalbe_Channel_Bin+j, k)[1] = j *m_nLookupTable_Interval + Half_Inteval;
				LookupTable_RGBImage.ptr<unsigned char >(i*m_nLookupTalbe_Channel_Bin+j, k)[2] = k *m_nLookupTable_Interval + Half_Inteval;
			}
	}
	Mat LookupTable_HSVImage;
	cvtColor(LookupTable_RGBImage,LookupTable_HSVImage, CV_BGR2HSV);


	double dResult;
	Mat Samples(1, 3, CV_8UC1);
	for(int i=0;i<m_nLookupTalbe_Channel_Bin;i++)
	{
		cout <<"Lookup table, step 2 :" << i <<"/" <<m_nLookupTalbe_Channel_Bin <<endl;
		for(int j= 0;j<m_nLookupTalbe_Channel_Bin;j++)
			for(int k=0;k<m_nLookupTalbe_Channel_Bin;k++)
			{
				Samples.at<unsigned char>(0,0) = LookupTable_HSVImage.ptr<unsigned char >(i*m_nLookupTalbe_Channel_Bin+j, k)[0];
				Samples.at<unsigned char>(0,1) = LookupTable_HSVImage.ptr<unsigned char >(i*m_nLookupTalbe_Channel_Bin+j, k)[1];
				Samples.at<unsigned char>(0,2) = LookupTable_HSVImage.ptr<unsigned char >(i*m_nLookupTalbe_Channel_Bin+j, k)[2];
				dResult = m_Em.calcLikelihood(Samples);
				
				int offset = 20;
				int temp = int(dResult + offset + 3);
				if(temp <0) temp = 0;
				if(temp > offset) temp = offset;
				
				temp = temp * 255.0/offset;
				if(temp>255) temp= 255;
				if(temp<0) temp= 0;
				m_lpLookupTable_Data[(i*m_nLookupTalbe_Channel_Bin+j)*m_nLookupTalbe_Channel_Bin+k] = (unsigned char)( temp );
			}
	}

	cout << "Begin to test training sample" << endl;
	//Mat Prob;
	Mat Result(lpTrainingImage.rows, lpTrainingImage.cols, CV_8UC1);
	for(int i=0;i<nImageHeight;i++)
		for(int j=0;j<nImageWidth;j++)
		{
			Samples.ptr<unsigned char >(0,0)[0] = lpImageHSV.ptr<unsigned char>(i,j)[0];
			Samples.ptr<unsigned char >(0,1)[0] = lpImageHSV.ptr<unsigned char >(i,j)[1];
			Samples.ptr<unsigned char >(0,2)[0] = lpImageHSV.ptr<unsigned char >(i,j)[2];


			double test = m_Em.calcLikelihood(Samples);
			int offset = 20;
			int temp = test + offset + 3;
			if(temp <0) temp = 0;
			if(temp > offset) temp = offset;

			temp = temp * 255.0/offset;
			if(temp>255) temp= 255;
			if(temp<0) temp= 0;
			Result.at<unsigned char>(i,j) = (unsigned char)( temp );
		}
	imshow("Training sample result image", Result);
	// create a lookup table
	cout << "Finish registration" << endl;

	return 0;
}

////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of CColorGMM
///  written by Ren Haibing 
/// Acknowledge		: 
///
/// Function name	: RGB_to_ColorMap
/// Description	    : transfer RGB image to color map 
///
/// Argument		: 
///
/// Return type		:  
///
/// Create Time		: 2015-4-28  10:23
///
///
/// Side Effect		: 
///                    
///////////////////////////////////////////////////////////////////////////////////////////////
Mat CColorGMM::RGB_to_ColorMap(Mat ImageMat)
{
	int nImageWidth = ImageMat.cols;
	int nImageHeight = ImageMat.rows;
	Mat ColorMap = cvCreateMat(nImageHeight, nImageWidth, CV_8UC1);
	int B,G,R;
	for(int i=0;i<nImageHeight;i++)
		for(int j=0;j<nImageWidth;j++)
		{
			B = ImageMat.ptr<unsigned char>(i,j)[0];
			G = ImageMat.ptr<unsigned char>(i,j)[1];
			R = ImageMat.ptr<unsigned char>(i,j)[2];
			if (B != 0)
			{
				int a = 0;
			}
			B = int(B*1.0/m_nLookupTable_Interval + 0.5);
			G = int(G*1.0/m_nLookupTable_Interval + 0.5);
			R = int(R*1.0/m_nLookupTable_Interval + 0.5);
			ColorMap.at<unsigned char>(i,j) = m_lpLookupTable_Data[((B*m_nLookupTalbe_Channel_Bin)+G)*m_nLookupTalbe_Channel_Bin+R];
			//ColorMap.at<unsigned char>(i, j) = m_lpLookupTable_Data[((R*m_nLookupTalbe_Channel_Bin) + G)*m_nLookupTalbe_Channel_Bin + B];
		}
	return ColorMap;
}
////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of CColorGMM
///  written by Ren Haibing 
/// Acknowledge		: 
///
/// Function name	: RGB_to_ColorMap
/// Description	    : transfer RGB image to color map 
///
/// Argument		: 
///
/// Return type		:  
///
/// Create Time		: 2015-4-28  10:23
///
///
/// Side Effect		: 
///                    
///////////////////////////////////////////////////////////////////////////////////////////////
void CColorGMM::RGB_to_ColorMap(IplImage*SrcImage, IplImage *ColorMap)
{
	int nImageWidth = SrcImage->width;
	int nImageHeight = SrcImage->height;
	int B, G, R;

	int nOffset;
	unsigned char *lpTemp;
	for (int i = 0; i < nImageHeight; i++)
	{
		lpTemp = (unsigned char *)(SrcImage->imageData + SrcImage->widthStep * i);
		for (int j = 0; j<nImageWidth; j++)
		{
			B = lpTemp[j*3 + 0];
			G = lpTemp[j*3 + 1];
			R = lpTemp[j*3 + 2];
			if (B != 0)
			{
				int a = 0;
			}
			B = int(B*1.0 / m_nLookupTable_Interval + 0.5);
			G = int(G*1.0 / m_nLookupTable_Interval + 0.5);
			R = int(R*1.0 / m_nLookupTable_Interval + 0.5);
			ColorMap->imageData[i*ColorMap->widthStep + j] = m_lpLookupTable_Data[((B*m_nLookupTalbe_Channel_Bin) + G)*m_nLookupTalbe_Channel_Bin + R];
		}
		//ColorMap.at<unsigned char>(i, j) = m_lpLookupTable_Data[((R*m_nLookupTalbe_Channel_Bin) + G)*m_nLookupTalbe_Channel_Bin + B];
	}
}

////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of CColorGMM
///  written by Ren Haibing 
/// Acknowledge		: 
///
/// Function name	: SaveLookupTable
/// Description	    : save lookup table
///
/// Argument		: 
///
/// Return type		:  
///
/// Create Time		: 2015-4-28  13:13
///
///
/// Side Effect		: 
///                    
///////////////////////////////////////////////////////////////////////////////////////////////
bool CColorGMM::SaveLookupTable(char *sFilename)
{
	FILE *file= fopen(sFilename, "wb");
	if(file == nullptr)
	{
		cout <<"Can't write file:" << sFilename  <<endl;
		return false;
	}
	int nResult = fwrite(m_lpLookupTable_Data, 1,m_nLookupTalbe_Channel_Bin*m_nLookupTalbe_Channel_Bin*m_nLookupTalbe_Channel_Bin, file);
	fclose(file);
	if(nResult == m_nLookupTalbe_Channel_Bin*m_nLookupTalbe_Channel_Bin*m_nLookupTalbe_Channel_Bin) return true;
	else return false;
}
////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of CColorGMM
///  written by Ren Haibing 
/// Acknowledge		: 
///
/// Function name	: LoadLookupTable
/// Description	    : load lookup table
///
/// Argument		: 
///
/// Return type		:  
///
/// Create Time		: 2015-4-28  13:23
///
///
/// Side Effect		: 
///                    
///////////////////////////////////////////////////////////////////////////////////////////////
bool CColorGMM::LoadLookupTable(char *sFilename)
{
	FILE *file= fopen(sFilename, "rb");
	if(file == nullptr)
	{
		cout <<"Can't find file:" << sFilename  <<endl;
		return false;
	}
	int nResult = fread(m_lpLookupTable_Data, 1,m_nLookupTalbe_Channel_Bin*m_nLookupTalbe_Channel_Bin*m_nLookupTalbe_Channel_Bin, file);
	fclose(file);

	if(nResult == m_nLookupTalbe_Channel_Bin*m_nLookupTalbe_Channel_Bin*m_nLookupTalbe_Channel_Bin) return true;
	else return false;
}