// HvChainStageClassifier.cpp: implementation of the HvChainStageClassifier class.
//
//////////////////////////////////////////////////////////////////////

#include "HvChainStageClassifier.h"

//#ifdef _DEBUG
#undef THIS_FILE
static char THIS_FILE[]=__FILE__;
//#define new DEBUG_NEW
//#endif


int MakeDir(char * filename )
{
//    char path[PATH_MAX];
//    char* p;
//    int pos;
//
//    struct _stat st;
//    strcpy( path, filename );
//
//    p = path;
//    for( ; ; )
//    {
//        pos = strcspn( p, "/\\" );
//
//        if( pos == (int) strlen( p ) ) break;
//        if( pos != 0 )
//        {
//            p[pos] = '\0';
//
//            if( p[pos-1] != ':' )
//            {
//                if( _stat( path, &st ) != 0 )
//                {
//                    if( _mkdir( path ) != 0 ) return 0;
//                }
//            }
//        }
//
//        p[pos] = '/';
//
//        p += pos + 1;
//    }

    return 1;
}

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

HvChainStageClassifier::HvChainStageClassifier()
{
	m_nWeakNum =0;
	m_RealWeakClassifier = NULL;
}

HvChainStageClassifier::~HvChainStageClassifier()
{
	ReleaseWeakClassifier();
}

////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the HvCascadeClassifier Class
///  written by Ren Haibing 
/// Acknowledge		: None
///
/// Function name	: CreateWeakClassifier
/// Description	    : given the stage classifier number, allocate the memory
///
/// Argument		: sFileName -- file  name
///
/// Return type		: 
///
/// Create Time		: 2005-11-11  09:21
///
///
/// Side Effect		: 
///
///////////////////////////////////////////////////////////////////////////////////////////////
bool HvChainStageClassifier::CreateWeakClassifier()
{
	int nSize = sizeof(HvRealHaarClassifier*);
	m_RealWeakClassifier = (HvRealHaarClassifier**)malloc(m_nWeakNum*nSize);
	assert(m_RealWeakClassifier);

	for(int i=0;i<m_nWeakNum;i++)
	{
		m_RealWeakClassifier[i] = new HvRealHaarClassifier();
		assert(m_RealWeakClassifier[i]);
	}
	return true;
}

void HvChainStageClassifier::ReleaseWeakClassifier()
{
	if(m_RealWeakClassifier == NULL)  return ;
	for(int i=0;i<m_nWeakNum;i++)
	{
		if(m_RealWeakClassifier[i] != NULL)
			delete m_RealWeakClassifier[i];
	}
	delete m_RealWeakClassifier;
	m_RealWeakClassifier = NULL;
}

////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the HvCascadeClassifier Class
///  written by Ren Haibing 
/// Acknowledge		: None
///
/// Function name	: LoadClassifier
/// Description	    : given the stage classifier 
///
/// Argument		: sFileName -- file  name
///
/// Return type		: 
///
/// Create Time		: 2005-11-11  09:21
///
///
/// Side Effect		: 
///
///////////////////////////////////////////////////////////////////////////////////////////////
bool HvChainStageClassifier::LoadStageClassifier(char * sFileName)
{
    FILE* file;
	int i;
    file = fopen(sFileName, "r");

	// file name error
	if(file == NULL) return false;

	/// read the weak classifier number
	m_nWeakNum = 0;
	fscanf( file, "%d", &m_nWeakNum );
	

	if( m_nWeakNum > 0 )
	{
		CreateWeakClassifier();
		for( i = 0; i < m_nWeakNum; i++ )
		{
		   m_RealWeakClassifier[i]->LoadHaarClassifier(file);
		}

		fscanf( file, "%e", &m_fThreshold );
	}

	/// close the file
    fclose( file );
	return true;
}

////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the HvCascadeClassifier Class
///  written by Ren Haibing 
/// Acknowledge		: None
///
/// Function name	: SaveStageClassifier
/// Description	    : given the stage classifier 
///
/// Argument		: sFileName -- file  name
///
/// Return type		: 
///
/// Create Time		: 2005-11-11  09:21
///
///
/// Side Effect		: 
///
///////////////////////////////////////////////////////////////////////////////////////////////
bool HvChainStageClassifier::SaveStageClassifier(char * sFileName)
{
	
	int i;
	if( !MakeDir( sFileName ) )	// make the stage directory
	{
		printf( "UNABLE TO CREATE DIRECTORY: %s\r\n", sFileName );
		return false;
	}
	FILE* file = fopen( sFileName, "w" );

	/// save the weak classifier number
    fprintf( file, "%d \n", m_nWeakNum );

	/// save each weak classifier
    for( i = 0; i < m_nWeakNum; i++ )
       m_RealWeakClassifier[i]->SaveHaarClassifier(file);

	/// save the stage threshold
    fprintf( file, "%f \n", m_fThreshold );

	fclose(file);
	return true;
}

////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the HvCascadeClassifier Class
///  written by Ren Haibing 
/// Acknowledge		: None
///
/// Function name	: Evaluate
/// Description	    : given the features, evalculate its score 
///
/// Argument		: sFileName -- file  name
///
/// Return type		: 
///
/// Create Time		: 2005-11-11  09:21
///
///
/// Side Effect		: 
///
///////////////////////////////////////////////////////////////////////////////////////////////
bool HvChainStageClassifier::Evaluate(int *sum, int *tilted, 
								      float normfactor, 
									  float fPreviousScore,
									  float *fReturnScore)
{
	int i;
	float fScore = fPreviousScore;
	for(i=0;i<m_nWeakNum;i++)
	{
		fScore += m_RealWeakClassifier[i]->Evaluate(sum, tilted, normfactor);                        
	}

	*fReturnScore = fScore;

	if(fScore>m_fThreshold - HV_THRESHOLD_EPS)	/// accepted as face
		return true;
	else /// rejected as non-face
		return false;
}
////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the HvCascadeClassifier Class
///  written by Ren Haibing 
/// Acknowledge		: None
///
/// Function name	: LoadStageFromSeq
/// Description	    : load the stage real weak classifier from the sequence
///
/// Argument		: seq -- the sequence
///
/// Return type		: 
///
/// Create Time		: 2006-4-5  14:57
///
///
/// Side Effect		: 
///
///////////////////////////////////////////////////////////////////////////////////////////////
bool HvChainStageClassifier::LoadStageFromRealSeq(CvSeq* seq)
{
	ReleaseWeakClassifier();
	/// 1. get the weak classifier number
	m_nWeakNum = seq->total;
	if(m_nWeakNum<0) return false;

	CreateWeakClassifier();

	HvRealHaarClassifier *RealHaarClassifier;
	
	for(int i=0;i<m_nWeakNum;i++)
	{
		RealHaarClassifier = *((HvRealHaarClassifier**) cvGetSeqElem(seq, i));

		m_RealWeakClassifier[i]->m_bTilted = RealHaarClassifier->m_bTilted;
		m_RealWeakClassifier[i]->m_Fastfeature = RealHaarClassifier->m_Fastfeature;
		m_RealWeakClassifier[i]->m_Feature = RealHaarClassifier->m_Feature;

		m_RealWeakClassifier[i]->m_nBinNum = RealHaarClassifier->m_nBinNum;
		m_RealWeakClassifier[i]->m_fMin = RealHaarClassifier->m_fMin;
		m_RealWeakClassifier[i]->m_fBinWidth_Inv = RealHaarClassifier->m_fBinWidth_Inv;
	
		memcpy(m_RealWeakClassifier[i]->m_fVal, RealHaarClassifier->m_fVal, sizeof(float)*MAX_REAL_CLASSIFIER_BIN);
			
		m_RealWeakClassifier[i]->m_nCompIdx = RealHaarClassifier->m_nCompIdx;
		m_RealWeakClassifier[i]->m_nFeatureNum = RealHaarClassifier->m_nFeatureNum;
		m_RealWeakClassifier[i]->m_nRectNum = RealHaarClassifier->m_nRectNum;
	}

	return true;
}

////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the HvCascadeClassifier Class
///  written by Ren Haibing 
/// Acknowledge		: None
///
/// Function name	: LoadStageClassifier_FromArray
/// Description	    : load the stage real weak classifier from the int and double array
///
/// Argument		: seq -- the sequence
///
/// Return type		: 
///
/// Create Time		: 2015-5-3  15:57
///
///
/// Side Effect		: 
///
///////////////////////////////////////////////////////////////////////////////////////////////
bool HvChainStageClassifier::LoadStageClassifier_FromArray(const unsigned short *Int_Array, const double * FloatArray, 
														  int*nIntRead, int *nFloatRead)
{
	int nIntNo =0;
	int nFloatNo = 0;
	int i, j, nFeatureNum, nRectNum, nHaar_type, nBinNum;
	/// read the weak classifier number
	m_nWeakNum = Int_Array[nIntNo];
	nIntNo++;
   
	if(m_nWeakNum <=0)
		return false;
	CreateWeakClassifier();
	for( i = 0; i < m_nWeakNum; i++ )
	{
		nFeatureNum = 1;
		nRectNum = Int_Array[nIntNo]; 
		nIntNo++;;
		nHaar_type = Int_Array[nIntNo];
		nIntNo++;
		nBinNum = Int_Array[nIntNo];
		nIntNo++;

		m_RealWeakClassifier[i]->m_nFeatureNum = nFeatureNum;
		m_RealWeakClassifier[i]->m_nRectNum = nRectNum;
		m_RealWeakClassifier[i]->m_nBinNum =  nBinNum;
		
		if(nHaar_type == 1)
		{
			m_RealWeakClassifier[i]->m_Feature.desc[0] = 't';
			m_RealWeakClassifier[i]->m_Feature.tilted = true;
			m_RealWeakClassifier[i]->m_bTilted = true;
		}
		else 
		{
			m_RealWeakClassifier[i]->m_Feature.desc[0] = 'h';
			m_RealWeakClassifier[i]->m_Feature.tilted = false;
			m_RealWeakClassifier[i]->m_bTilted = false;
		}
		
		// read rects
		for(j=0;j<nRectNum;j++)
		{
			m_RealWeakClassifier[i]->m_Feature.rect[j].r.x = Int_Array[nIntNo];
			nIntNo++;
			m_RealWeakClassifier[i]->m_Feature.rect[j].r.y = Int_Array[nIntNo];
			nIntNo++;
			m_RealWeakClassifier[i]->m_Feature.rect[j].r.width = Int_Array[nIntNo];
			nIntNo++;
			m_RealWeakClassifier[i]->m_Feature.rect[j].r.height = Int_Array[nIntNo];
			nIntNo++;

			m_RealWeakClassifier[i]->m_Feature.rect[j].weight = Int_Array[nIntNo] - 50;
			nIntNo++;
		}//for(j=0;j<nRectNum;j++)
		for( j = nRectNum; j < HV_FEATURE_MAX; j++ ) 
		{
			m_RealWeakClassifier[i]->m_Feature.rect[j].r.x = 0;
			m_RealWeakClassifier[i]->m_Feature.rect[j].r.y = 0;
			m_RealWeakClassifier[i]->m_Feature.rect[j].r.width = 0;
			m_RealWeakClassifier[i]->m_Feature.rect[j].r.height = 0;
			m_RealWeakClassifier[i]->m_Feature.rect[j].weight = 0.0F;
		}
		
		
		m_RealWeakClassifier[i]->m_fMin = FloatArray[nFloatNo];
		nFloatNo++;

		m_RealWeakClassifier[i]->m_fBinWidth_Inv = FloatArray[nFloatNo];
		nFloatNo++;

		m_RealWeakClassifier[i]->GetFastFeature();
		// read the value
		for(j=0;j<m_RealWeakClassifier[i]->m_nBinNum;j++)
		{
			m_RealWeakClassifier[i]->m_fVal[j] = FloatArray[nFloatNo];
			nFloatNo++;
		}
	}
	
	m_fThreshold = FloatArray[nFloatNo];
	nFloatNo++;

	*nIntRead = nIntNo;
	*nFloatRead = nFloatNo;
	return true;
}
