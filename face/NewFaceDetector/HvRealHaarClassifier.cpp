// HvRealHaarClassifier.cpp: implementation of the HvRealHaarClassifier class.
//
//////////////////////////////////////////////////////////////////////

#include "HvRealHaarClassifier.h"
#include "Common_Func.h"

//#ifdef _DEBUG
#undef THIS_FILE
static char THIS_FILE[]=__FILE__;
//#define new DEBUG_NEW
//#endif

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////
HvRealHaarClassifier::HvRealHaarClassifier()
{
	m_bTilted = false;
	m_nWinWidth = WIN_WIDTH;
	m_nWinHeight = m_nWinWidth;
	
	m_nBinNum = MAX_REAL_CLASSIFIER_BIN;
}

HvRealHaarClassifier::~HvRealHaarClassifier()
{
}

////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the HvRealHaarClassifier Class
///  written by Ren Haibing 
/// Acknowledge		: None
///
/// Function name	: LoadClassifier
/// Description	    : load weak classifier
///
/// Argument		: 
///
/// Return type		: 
///
/// Create Time		: 2006-4-3  10:03
///
///
/// Side Effect		: 
///
///////////////////////////////////////////////////////////////////////////////////////////////
bool HvRealHaarClassifier::LoadHaarClassifier(FILE* file)
{
    int i;

    fscanf( file, "%d", &m_nFeatureNum ); // feature수 = 1
    if( m_nFeatureNum > 0 )
    {
 		fscanf( file, "%d", &m_nRectNum);
        for( i = 0; i < m_nRectNum; i++ ) // count = 1
        {
            LoadHaarFeature(file, i); 
		//	fscanf( file, "%f %f",	&(classifier->left_var), &(classifier->left_limit)); 
 		//	fscanf( file, "%f %f",	&(classifier->right_var), &(classifier->right_limit)); 
        }
        for( i = m_nRectNum; i < HV_FEATURE_MAX; i++ ) 
		{
			m_Feature.rect[i].r.x = 0;
			m_Feature.rect[i].r.y = 0;
			m_Feature.rect[i].r.width = 0;
			m_Feature.rect[i].r.height = 0;
			m_Feature.rect[i].weight = 0.0F;
		}

	    fscanf( file, "%s", &(m_Feature.desc) );		// feature description을 읽는다. 
		if(m_Feature.desc[0] == 't' )
		{
			m_Feature.tilted = true;
			m_bTilted = true;
		}
		else 
		{
			m_Feature.tilted = false;
			m_bTilted = false;
		}

        fscanf( file, "%f  %f   %d\n", &m_fMin, &m_fBinWidth_Inv, &m_nBinNum);
		if(m_nBinNum<0) 
			m_nBinNum = MAX_REAL_CLASSIFIER_BIN;
		if(m_nBinNum>MAX_REAL_CLASSIFIER_BIN)
			m_nBinNum = MAX_REAL_CLASSIFIER_BIN;

		for( i = 0; i <m_nBinNum; i++ )
        {
            fscanf( file, "%f",&(m_fVal[i]) ); 
        }
    }
    
	GetFastFeature();

	return true;
}
////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the HvRealHaarClassifier Class
///  written by Ren Haibing 
/// Acknowledge		: None
///
/// Function name	: SaveHaarClassifier
/// Description	    : save weak classifier
///
/// Argument		: 
///
/// Return type		: 
///
/// Create Time		: 2006-4-3  10:05
///
///
/// Side Effect		: 
///
///////////////////////////////////////////////////////////////////////////////////////////////
bool HvRealHaarClassifier::SaveHaarClassifier(FILE* file)
{
    fprintf( file, "%d\n", m_nFeatureNum);
	fprintf( file, "%d\n", m_nRectNum);

    for(int i = 0; i <m_nRectNum; i++ )
    {
        SaveHaarFeature( file, i );
    }
 
 	fprintf( file, "%s\n", m_Feature.desc);		// feature description을 읽는다. 
    fprintf( file, "%f  %f  %d\n",  m_fMin, m_fBinWidth_Inv, m_nBinNum);
	for(int i = 0; i < m_nBinNum; i++ )
    {
		fprintf( file, "%e  ", m_fVal[i]); 
	}
 	fprintf( file, "\n");		// feature description을 읽는다. 
	
	return true;
}

////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the HvRealHaarClassifier Class
///  written by Ren Haibing 
/// Acknowledge		: None
///
/// Function name	: LoadHaarFeature
/// Description	    : load haar classifier
///
/// Argument		: file -- file pointer
/// Argument		: nNo -- haar feature no
///
/// Return type		: 
///
/// Create Time		: 2006-4-3  10:10
///
///
/// Side Effect		: 
///
///////////////////////////////////////////////////////////////////////////////////////////////
bool HvRealHaarClassifier::LoadHaarFeature(FILE* file,  int nNo)
{
	int tmp;
	int weight;
    fscanf( file, "%d %d %d %d %d %d",			// x, y, width, height, 0, weight
        &(m_Feature.rect[nNo].r.x),
        &(m_Feature.rect[nNo].r.y),
        &(m_Feature.rect[nNo].r.width),
        &(m_Feature.rect[nNo].r.height),
        &tmp, &weight );
    m_Feature.rect[nNo].weight = (float) weight;	// int를 float으로 변환해서 저장.

 	return true;
}


////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the HvRealHaarClassifier Class
///  written by Ren Haibing 
/// Acknowledge		: None
///
/// Function name	: SaveHaarFeature
/// Description	    : save haar classifier
///
/// Argument		: file -- file pointer
/// Argument		: nNo -- haar feature no
///
/// Return type		: 
///
/// Create Time		: 2006-4-3  10:20
///
///
/// Side Effect		: 
///
///////////////////////////////////////////////////////////////////////////////////////////////
void HvRealHaarClassifier::SaveHaarFeature(FILE *file, int nNo)
{
//    fprintf( file, "%d\n", m_nFeatureNum);
	int nWeight = m_Feature.rect[nNo].weight;
    fprintf( file, "%d %d %d %d %d %d\n",
        m_Feature.rect[nNo].r.x,
        m_Feature.rect[nNo].r.y,
        m_Feature.rect[nNo].r.width,
        m_Feature.rect[nNo].r.height,
        0,
        nWeight );
}

////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the HvRealHaarClassifier Class
///  written by Ren Haibing 
/// Acknowledge		: None
///
/// Function name	: Evaluate
/// Description	    : evalualate the image feature 
///
/// Argument		: sum -- integral image
/// Argument		: tilted -- tilted integral image
/// Argument		: fNorm -- feature normal value
///
/// Return type		: 
///
/// Create Time		: 2005-11-22  19:19
///
///
/// Side Effect		: 
///
///////////////////////////////////////////////////////////////////////////////////////////////
float HvRealHaarClassifier::Evaluate(int* sum, int* tilted, 							 
								 float fNorm)
{
    int* img = NULL;
    int i = 0;
    float fValue = 0.0F;
    
    img = ( m_bTilted) ? tilted : sum;
    assert( img );

    for( i = 0; m_Fastfeature.rect[i].weight != 0.0F && i < HV_FEATURE_MAX; i++ )
    {
        fValue += m_Fastfeature.rect[i].weight *
            ( img[m_Fastfeature.rect[i].p0] - img[m_Fastfeature.rect[i].p1] -
              img[m_Fastfeature.rect[i].p2] + img[m_Fastfeature.rect[i].p3] );
    }

	fValue = fValue/fNorm;
    
	int nBinNum = int((fValue-m_fMin)*m_fBinWidth_Inv);

	if(nBinNum<0) 
		nBinNum = 0;
	if(nBinNum >= MAX_REAL_CLASSIFIER_BIN) 
		nBinNum = MAX_REAL_CLASSIFIER_BIN-1;

	return m_fVal[nBinNum];
}

void HvRealHaarClassifier::GetFastFeature(double dScale)
{
	ConvertToFastFeature(&m_Feature, &m_Fastfeature, 1, m_nWinWidth+1, dScale);
}
void HvRealHaarClassifier::GetFastFeature(int x, int y, int step,double dScale)
{
	ConvertToFastFeature(&m_Feature, &m_Fastfeature, 1, step, x, y, dScale);
}

////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the HvRealHaarClassifier Class
///  written by Ren Haibing 
/// Acknowledge		: None
///
/// Function name	: Flip_H
/// Description	    : flip the haar classifier
///
/// Argument		: SrcClassifier --source haar feature
///
/// Return type		: 
///
/// Create Time		: 2006-02-17  16:12
///
///
/// Side Effect		: 
///
///////////////////////////////////////////////////////////////////////////////////////////////
void HvRealHaarClassifier::Flip_H(HvRealHaarClassifier *SrcClassifier)
{
/*
	// 1: copy the values
	m_nFeatureNum = SrcClassifier->m_nFeatureNum;
	m_nRectNum = SrcClassifier->m_nRectNum;
	m_nWinWidth = SrcClassifier->m_nWinWidth;
	m_nWinHeight = SrcClassifier->m_nWinHeight;
	m_nLeft = SrcClassifier->m_nLeft;
	m_nRight = SrcClassifier->m_nRight;
	
	m_fVal[0] = SrcClassifier->m_fVal[0];
	m_fVal[1] = SrcClassifier->m_fVal[1];
	m_fThreshold = SrcClassifier->m_fThreshold;

	// 2: flip the fast feature
	int i;


	m_Feature.tilted = SrcClassifier->m_Feature.tilted;
	memcpy(m_Feature.desc, SrcClassifier->m_Feature.desc, HV_FEATURE_DESC_MAX);
	for(i=0;i<m_nRectNum;i++)
	{		
		m_Feature.rect[i].weight = SrcClassifier->m_Feature.rect[i].weight;

		m_Feature.rect[i].r = SrcClassifier->m_Feature.rect[i].r;
		m_Feature.rect[i].r.x = m_nWinWidth -1 - m_Feature.rect[i].r.x - m_Feature.rect[i].r.width;
	}
	for(i=m_nRectNum;i<HV_FEATURE_MAX;i++)
	{
		m_Feature.rect[i].weight = 0.0f;
		m_Feature.rect[i].r.x = 0;
		m_Feature.rect[i].r.y = 0;
		m_Feature.rect[i].r.width = 0;
		m_Feature.rect[i].r.height = 0;
	}

	GetFastFeature(1.0);

*/
}
