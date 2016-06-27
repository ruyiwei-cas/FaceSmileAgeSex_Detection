// HvChainCascadeClassifier.cpp: implementation of the HvChainCascadeClassifier class.
//
//////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include "HvChainCascadeClassifier.h"

//#ifdef _DEBUG
#undef THIS_FILE
static char THIS_FILE[]=__FILE__;
//#define new DEBUG_NEW
//#endif

//////////////////////////////////////////////////////////////////////
// Construction
//////////////////////////////////////////////////////////////////////
HvChainCascadeClassifier::HvChainCascadeClassifier(int nWinWidth, int nWinHeight)
{
	m_nStageNum = 0;
	m_nWinWidth = nWinWidth;
	m_nWinHeight = nWinHeight;
}
//////////////////////////////////////////////////////////////////////
//copy Construction
//////////////////////////////////////////////////////////////////////
HvChainCascadeClassifier::HvChainCascadeClassifier(HvChainCascadeClassifier &c)
{
	m_nStageNum = c.m_nStageNum;
	m_nWinWidth = c.m_nWinWidth;
	m_nWinHeight = c.m_nWinHeight;
	

	m_realWindowSize = c.m_realWindowSize;
	m_sumsize = c.m_sumsize;
	scale = c.scale;
	m_invWindowArea = c.m_invWindowArea;
	m_sum_image = c.m_sum_image;
	m_tilt_image = c.m_tilt_image;
        m_sqsum_image = c.m_sqsum_image;
        m_pq0 = c.m_pq0;
	m_pq1 = c.m_pq1;
	m_pq2 = c.m_pq2;
	m_pq3 = c.m_pq3;
        m_p0 = c.m_p0;
	m_p1 = c.m_p1;
	m_p2 = c.m_p2;
	m_p3 = c.m_p3;


	for(int i=0; i<MAX_STAGE_NUM; i++){
		m_StageClassifier[i] = c.m_StageClassifier[i];
	}
}

//////////////////////////////////////////////////////////////////////
// Destruction
//////////////////////////////////////////////////////////////////////
HvChainCascadeClassifier::~HvChainCascadeClassifier()
{

}
////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the HvChainCascadeClassifier Class
///  written by Ren Haibing 
/// Acknowledge		: None
///
/// Function name	: LoadClassifier
/// Description	    : given the classifier directory name, load it
///
/// Argument		: sDirectoryName -- directory  name
///
/// Return type		: 
///
/// Create Time		: 2005-11-11  09:21
///
///
/// Side Effect		: 
///
///////////////////////////////////////////////////////////////////////////////////////////////
bool HvChainCascadeClassifier::LoadClassifier(char * sDirectoryName)
{
    m_nStageNum =0;
	
	bool bFind = true;
    char StagePathName[PATH_MAX];	

	while((m_nStageNum<MAX_STAGE_NUM) && bFind)
	{
		sprintf( StagePathName, "%s/%d/%s", sDirectoryName, m_nStageNum, STAGE_CART_FILE_NAME );
        
		bFind = m_StageClassifier[m_nStageNum].LoadStageClassifier(StagePathName);
		if(bFind)
			m_nStageNum++;
	}

	return true;
}
////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the HvChainCascadeClassifier Class
///  written by Ren Haibing 
/// Acknowledge		: None
///
/// Function name	: LoadStageClassifier
/// Description	    : given the classifier directory name, load it
///
/// Argument		: sDirectoryName -- directory  name
/// Argument		: nStageNo -- stage no.
///
/// Return type		: 
///
/// Create Time		: 2005-11-14  16:23
///
///
/// Side Effect		: 
///
///////////////////////////////////////////////////////////////////////////////////////////////
bool HvChainCascadeClassifier::LoadStageClassifier(char * sDirectoryName, int nStageNo)
{
  	bool bFind = true;
	
    char StagePathName[PATH_MAX];	
	if(nStageNo<0) return false;
	if(nStageNo>= MAX_STAGE_NUM) return false;

	sprintf( StagePathName, "%s/%d/%s", sDirectoryName, nStageNo, STAGE_CART_FILE_NAME );
	bFind = m_StageClassifier[nStageNo].LoadStageClassifier(StagePathName);

	return bFind;
}
////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the HvChainCascadeClassifier Class
///  written by Ren Haibing 
/// Acknowledge		: None
///
/// Function name	: LoadClassifier
/// Description	    : given the classifier directory name, load it
///
/// Argument		: sDirectoryName -- directory  name
///
/// Return type		: 
///
/// Create Time		: 2005-11-11  09:21
///
///
/// Side Effect		: 
///
///////////////////////////////////////////////////////////////////////////////////////////////
void HvChainCascadeClassifier::SaveClassifier(char * sDirectoryName)
{
   int i;
   for(i=0;i<m_nStageNum;i++)
  	   SaveStageClassifier(sDirectoryName, i);
}

void HvChainCascadeClassifier::SaveStageClassifier(char * sDirectoryName, int nStageNo)
{
   char StagePathName[PATH_MAX];
   memset(StagePathName, 0, PATH_MAX);
   sprintf( StagePathName, "%s/%d/%s", sDirectoryName, nStageNo, STAGE_CART_FILE_NAME );

   if(MakeDir(StagePathName))	// stage를 넣을 foler 없으면 folder를 만든다.
    {
	   m_StageClassifier[nStageNo].SaveStageClassifier(StagePathName);
    }  
   else
	   printf( "UNABLE TO CREATE DIRECTORY: %s\r\n", StagePathName );
}

////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the HvChainCascadeClassifier Class
///  written by Ren Haibing 
/// Acknowledge		: None
///
/// Function name	: Evaluate
/// Description	    : evaluate the parameter to be face or not
///
/// Argument		: sDirectoryName -- directory  name
///
/// Return type		: 
///
/// Create Time		: 2005-11-11  11:31
///
///
/// Side Effect		: 
///
///////////////////////////////////////////////////////////////////////////////////////////////
float HvChainCascadeClassifier::Evaluate(int *sum, int *tilted, 
									float normfactor)
{
	int i=0;
	bool bFace = true;
	float fPreviousScore = 0;
	float fCurrentScore =0;
	
	while((i<m_nStageNum) && bFace)
	{
		bFace  = m_StageClassifier[i].Evaluate(sum, tilted,  
			             normfactor, fPreviousScore, &fCurrentScore);
		if(!bFace) // face
			fCurrentScore = -(m_nStageNum-i);

		fPreviousScore = fCurrentScore;
		i++;
	}
	return fCurrentScore;
}


////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the HvChainCascadeClassifier Class
///  written by Ren Haibing 
/// Acknowledge		: None
///
/// Function name	: Evaluate_bool
/// Description	    : evaluate the parameter to be face or not
///
/// Argument		: 
///
/// Return type		:  true - face, false : not face;
///
/// Create Time		: 2005-11-11  11:31
///
///
/// Side Effect		: 
///
///////////////////////////////////////////////////////////////////////////////////////////////
bool HvChainCascadeClassifier::Evaluate_bool(int *sum, int *tilted, 									
									    float mean, float normfactor)
{
	
	if( (normfactor < dMinNorm) || (normfactor>dMaxNorm) ||
		(mean < dMinMean) || (mean > dMaxMean))
	{
		return false;
	}

	int i=0;
	bool bFace = true;
	float fPreviousScore = 0;
	float fCurrentScore = 0;
	while((i<m_nStageNum) && bFace)
	{
		bFace  = m_StageClassifier[i].Evaluate(sum, tilted,  
			             normfactor,fPreviousScore, &fCurrentScore);

		fPreviousScore = fCurrentScore;
		i++;
	}
	return bFace;
}

////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the HvChainCascadeClassifier Class
///  written by Ren Haibing 
/// Acknowledge		: None
///
/// Function name	: Evaluate
/// Description	    : evaluate the parameter to be face or not
///
/// Argument		: dScale  = image size / 24(standard size)
///
/// Return type		: 
///
/// Create Time		: 2005-11-11  11:31
///
///
/// Side Effect		: 
///
///////////////////////////////////////////////////////////////////////////////////////////////
void HvChainCascadeClassifier::SetImageWinSize(int nWidth, int nHeight)
{
	int i,j;
	int nWeakClassiferNum;
	double dScale = nWidth*1.0/m_nWinWidth;


	for(i=0;i<m_nStageNum;i++)
	{
		nWeakClassiferNum = m_StageClassifier[i].m_nWeakNum;
		for(j=0;j<nWeakClassiferNum;j++)
		{
			m_StageClassifier[i].m_RealWeakClassifier[j]->GetFastFeature(dScale);
		}
	}
}


///-------------------------------------------------------------------------------------
///               The followings are for fast face detection in real time system

 ////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the HvChainCascadeClassifier Class
///  written by Ren Haibing 
/// Acknowledge		: None
///
/// Function name	: CalMeanNorm
/// Description	    : calculate the mean and norm of a window
///
/// Argument		: (nStart_x, nStart_y) --  windows start position
/// Argument		: fmean -- mean value for return 
/// Argument		: fnorm -- norm value for return
///
/// Return type		: 
///
/// Create Time		: 2006-01-04 11:35
///
///
/// Side Effect		: 
///
///////////////////////////////////////////////////////////////////////////////////////////////
bool HvChainCascadeClassifier::CalMeanNorm(int nStart_x, int nStart_y, float *fmean, float *Inv_fWinNorm)
 {
	int p_offset = nStart_y * m_sumsize.width + nStart_x;
	int pq_offset = p_offset;

//	p_offset = nStart_y * m_sumsize.width + nStart_x;
//	pq_offset = p_offset;

	/// 1. calculate the mean and variance
	double temp1 = m_p0[p_offset] - m_p1[p_offset] - m_p2[p_offset] + m_p3[p_offset];
	double mean, variance_norm_factor;
	mean = temp1 * m_invWindowArea;

	if((mean < dMinMean)||(mean > dMaxMean))
		return false;

	variance_norm_factor = m_pq0[pq_offset] - m_pq1[pq_offset] -
                           m_pq2[pq_offset] + m_pq3[pq_offset];	
	
	variance_norm_factor = sqrt(variance_norm_factor*m_invWindowArea - mean*mean);	
	if((variance_norm_factor<dMinNorm)||(variance_norm_factor>dMaxNorm))
		return false;

	*fmean = (float)mean;
//	*fnorm = (float)variance_norm_factor;
	*Inv_fWinNorm = float(1.0/variance_norm_factor);
	 return true;
 }
////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the HvChainCascadeClassifier Class
///  written by Ren Haibing 
/// Acknowledge		: None
///
/// Function name	: SetImagesForClassifier
/// Description	    : set the parameter for fast face detection
///
/// Argument		: (nArea_Width, nArea_Height) -- real windows size
/// Argument		: sum -- sum image 
/// Argument		: sqsum -- sqsum image
///
/// Return type		: 
///
/// Create Time		: 2005-12-28  11:09
///
///
/// Side Effect		: 
///
///////////////////////////////////////////////////////////////////////////////////////////////
void HvChainCascadeClassifier::SetImagesForClassifier( int nArea_Width,
						  int nArea_Height,
						  int nImageWidth,
						  int nImageHeight,
						  int *sum_image,
						  int *tilt_image,
						  double *sqsum_image  )
{
	int i,j,k;
	double dScaleFactor = nArea_Width * 1.0/m_nWinWidth;
	m_sum_image = tilt_image;
	m_tilt_image = tilt_image;
    	m_sqsum_image = sqsum_image;
    
	m_realWindowSize.width = nArea_Width;
	m_realWindowSize.height = nArea_Height;
	m_sumsize.width = nImageWidth + 1;
	m_sumsize.height = nImageHeight + 1;

	FdRect 	equ_rect;		
    	equ_rect.x = equ_rect.y = int(dScaleFactor+0.5);							
    	equ_rect.width	= int((m_nWinWidth-2)*dScaleFactor+0.5); 
	equ_rect.height = int((m_nWinWidth-2)*dScaleFactor+0.5);	
 
	double weight_scale = 1./(equ_rect.width*equ_rect.height);
    	m_invWindowArea = weight_scale;	

	m_p0 = sum_image + (equ_rect.y * m_sumsize.width + equ_rect.x);												
	m_p1 = sum_image + (equ_rect.y * m_sumsize.width + (equ_rect.x + equ_rect.width));							
	m_p2 = sum_image + ((equ_rect.y + equ_rect.height) * m_sumsize.width + equ_rect.x);							
	m_p3 = sum_image + ((equ_rect.y + equ_rect.height) * m_sumsize.width + (equ_rect.x + equ_rect.width));		

	m_pq0 = sqsum_image + (equ_rect.y * m_sumsize.width + equ_rect.x);											
	m_pq1 = sqsum_image + (equ_rect.y * m_sumsize.width + (equ_rect.x + equ_rect.width));						
	m_pq2 = sqsum_image + ((equ_rect.y + equ_rect.height) * m_sumsize.width + equ_rect.x);						
	m_pq3 = sqsum_image + ((equ_rect.y + equ_rect.height) * m_sumsize.width + (equ_rect.x + equ_rect.width));	

	/// modify the feature 
	int nWeakClassiferNum;
	HidFeature *Fast_HidFeature;
	HvFeature * Haarfeature;
	int nHaarRectNum;
    	FdRect tr;

	for(i=0;i<m_nStageNum;i++)
	{// for each stage
		nWeakClassiferNum = m_StageClassifier[i].m_nWeakNum;
		for(j=0;j<nWeakClassiferNum;j++)
		{// for each weak classifier
			Fast_HidFeature = &(m_StageClassifier[i].m_RealWeakClassifier[j]->m_HidFeature);
			Haarfeature = &(m_StageClassifier[i].m_RealWeakClassifier[j]->m_Feature);
			nHaarRectNum = m_StageClassifier[i].m_RealWeakClassifier[j]->m_nRectNum;

        		double sum0 = 0, area0 = 0;			
			for(k=0;k<nHaarRectNum;k++)
			{// for each rect
                		tr.x = int(Haarfeature->rect[k].r.x * dScaleFactor + 0.5);				
		                tr.y = int(Haarfeature->rect[k].r.y * dScaleFactor + 0.5);				
        		        tr.width = int(Haarfeature->rect[k].r.width * dScaleFactor + 0.5);				
                		tr.height = int(Haarfeature->rect[k].r.height * dScaleFactor + 0.5);				

				if(!(Haarfeature->tilted))
				{
					Fast_HidFeature->rect[k].p0 = sum_image + (tr.y * m_sumsize.width + tr.x);
					Fast_HidFeature->rect[k].p1 =  sum_image + (tr.y * m_sumsize.width + (tr.x + tr.width));
					Fast_HidFeature->rect[k].p2 = sum_image + ((tr.y + tr.height) * m_sumsize.width + tr.x);
					Fast_HidFeature->rect[k].p3 = sum_image + ((tr.y + tr.height) * m_sumsize.width + (tr.x + tr.width));
					Fast_HidFeature->rect[k].weight =  Haarfeature->rect[k].weight*weight_scale;
				}
				else
				{
					Fast_HidFeature->rect[k].p0 = tilt_image + (tr.y * m_sumsize.width + tr.x);
					Fast_HidFeature->rect[k].p1 = tilt_image + (tr.y+tr.height) * m_sumsize.width + (tr.x -tr.height);
					Fast_HidFeature->rect[k].p2 = tilt_image + (tr.y + tr.width) * m_sumsize.width + (tr.x+tr.width);
					Fast_HidFeature->rect[k].p3 = tilt_image + (tr.y + tr.height+tr.width) * m_sumsize.width + (tr.x + tr.width -tr.height);
					Fast_HidFeature->rect[k].weight = Haarfeature->rect[k].weight * weight_scale;
				}
	

				if( k == 0 ){
                   			 area0 = tr.width * tr.height;	
				}else{
                			sum0 += Fast_HidFeature->rect[k].weight * tr.width * tr.height;		
				}
			}//for(k=0
			Fast_HidFeature->rect[0].weight = (float)(-sum0/area0);
		}//for(j=0;
	}//for(i=0;

}
////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the HvChainCascadeClassifier Class
///  written by Ren Haibing 
/// Acknowledge		: None
///
/// Function name	: Evaluate
/// Description	    : evaluate the face area with the given rect
///
/// Argument		: sum -- sum image 
/// Argument		: sqsum -- sqsum image
///
/// Return type		: 
///
/// Create Time		: 2005-12-28  11:09
///
///
/// Side Effect		: 
///
///////////////////////////////////////////////////////////////////////////////////////////////
bool HvChainCascadeClassifier::Fast_Evaluate(int nStart_x, int nStart_y,
					float mean, float Inv_fNorm, float *fResult)
{
	int result = -1;
	int p_offset = nStart_y * m_sumsize.width + nStart_x;
	int pq_offset = p_offset;
	int i;

	//p_offset = nStart_y * m_sumsize.width + nStart_x;
    	//pq_offset = p_offset;

	/// 2. determine it is face or not
	bool bFace = true;
	float fConfidence =0;
	float fWeakClassifierValue;
	float fTotal_Confidence = 0;
	int nStageNo = 0;
	int nWeakClassifierNum;
	HidFeature* feature;
	HvRealHaarClassifier *RealWeakClassifier;
	int nBinNum;
	
	//fConfidence = 0;

	while((nStageNo < m_nStageNum) && bFace)
	{
		nWeakClassifierNum = m_StageClassifier[nStageNo].m_nWeakNum;
       		for (i = 0; i < nWeakClassifierNum; i++)
        	{
            		RealWeakClassifier = m_StageClassifier[nStageNo].m_RealWeakClassifier[i];
            		feature = &(RealWeakClassifier->m_HidFeature);
            		fWeakClassifierValue = calc_sum(feature->rect[0], p_offset)*feature->rect[0].weight;

            		fWeakClassifierValue += calc_sum(feature->rect[1], p_offset)*feature->rect[1].weight;

            		if (RealWeakClassifier->m_nRectNum == 3)
           		{
                		fWeakClassifierValue += calc_sum(feature->rect[2], p_offset)*feature->rect[2].weight;
            		}
            		//fWeakClassifierValue /= variance_norm_factor;
			fWeakClassifierValue *= Inv_fNorm;

            		nBinNum = (fWeakClassifierValue - RealWeakClassifier->m_fMin)*RealWeakClassifier->m_fBinWidth_Inv;

           		if (nBinNum < 0) nBinNum = 0;
            		//if (nBinNum >= MAX_REAL_CLASSIFIER_BIN) nBinNum = MAX_REAL_CLASSIFIER_BIN - 1;
            		if (nBinNum >= RealWeakClassifier->m_nBinNum) nBinNum = RealWeakClassifier->m_nBinNum - 1;

           		if (nBinNum < 0) nBinNum = 0;

            		fConfidence += RealWeakClassifier->m_fVal[nBinNum];
        	}
        
		if(fConfidence < m_StageClassifier[nStageNo].m_fThreshold)
		{
			bFace = false;
			break;			
		}
		nStageNo++;
	}

	if(!bFace)
	{ // not face
		*fResult =  -nStageNo;
		return false;
	}
	else
	{ // face
		*fResult = fConfidence ;
		return true;
	}
}

