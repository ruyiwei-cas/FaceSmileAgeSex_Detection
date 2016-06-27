/**
*** Copyright (C) 1985-2011 Intel Corporation.  All rights reserved.
***
*** The information and source code contained herein is the exclusive
*** property of Intel Corporation and may not be disclosed, examined
*** or reproduced in whole or in part without explicit written authorization
*** from the company.
***
*** Embedded Application Lab, Intel Labs China.
**/

#include <algorithm>

#include "cxmcboost.hpp"
#include "cxfaceutil.hpp"

const int MCBOOST_GRPLOSS = 1; // two boost, GRPLOSS (better for imbalance) or BOOSTMA
const int USE_ALPHA = 1;

//////////////////////////////////////////////////////////////////////////
CxLogit::~CxLogit()
{
	clear();
}
void CxLogit::clear()
{
	if( m_beta != NULL )
		cvFree(&m_beta);
}

void CxLogit::read( CvFileStorage* fs, CvFileNode* root_node )
{
	bool ok = false;
	CV_FUNCNAME( "CxLogit::read" );

	__BEGIN__;
	clear();

	CV_CALL( m_dim = cvReadIntByName( fs, root_node, "nDim", -1 ) );
	CV_CALL( m_nC = cvReadIntByName( fs, root_node, "nC", -1 ) );
	if( m_dim <= 0 || m_nC <= 0 )
	{
		printf("Error models!\n" );
		exit(0);
	}

	CvMat* rawdat;	// CV_64F
	CV_CALL( rawdat = (CvMat*)cvReadByName(fs, root_node, "LogitBeta") );
	assert(rawdat->rows == m_nC && rawdat->cols == (m_dim+1) );

	m_beta = (float*)cvAlloc(sizeof(float)*(m_dim + 1)*m_nC);
	// m_vBeta = cvMat(m_nC, m_dim+1, CV_32F, m_beta);
	for(int i=0; i<m_nC; ++i)
	{
		for(int j=0; j<m_dim+1; ++j)
			m_beta[i*(m_dim+1) + j] = cvmGet(rawdat, i, j);
	}
	cvReleaseMat(&rawdat);

	ok = true;
	__END__;
	if( !ok )
		clear();
}

inline float computLogit(rarray& vx, float * m_beta, int m_dim, int kth)
{
	CvMat vbeta = cvMat(1, m_dim+1, CV_32F, m_beta + kth*(m_dim+1));
	CvMat vxmat = cvMat(1, m_dim+1, CV_32F, &vx[0]);
	float betax = cvDotProduct(&vbeta, &vxmat);

	float ret, exp_a;
	const float exp_max = 25.0f;
	if( fabs(betax) <= exp_max ) // when no overflow
	{
		exp_a = expf(betax);
		ret = exp_a/(1.0f + exp_a);

		return ret;
	}
	if( betax > exp_max )
		ret = 1.0;
	if( betax < -exp_max)
		ret = 0.0;

	return ret;
}

int CxLogit::predict(float* rvTest, float* ret /*=NULL*/)
{
	rarray vx(m_dim + 1);
	for(int i=0; i<m_dim; ++i)
	{
		vx[i] = rvTest[i];
	}
	vx[m_dim] = 1.0f;

	int id = 0;
	if( m_nC == 2 )
	{
		float prob = computLogit(vx, m_beta, m_dim, 0);
		if( prob > 0.5f )
			id = 0;
		else
			id = 1;
		if( ret != NULL )
		{
			ret[0] = prob;
			ret[1] = 1-prob;
		}
	}
	else
	{
		// multi-class version
		float psum = 0;
		std::vector<float> pscore(m_nC);
		for(int i=0; i<m_nC; i++)
		{
			pscore[i] = (float)computLogit(vx, m_beta, m_dim, i);
			psum += pscore[i];
		}	
		float smax = -1;
		for(int i=0; i<m_nC; ++i)
		{
			pscore[i] /= psum;
			if( pscore[i] > smax )
			{
				smax = pscore[i];
				id = i;
			}
		}
		if( ret != NULL )
			memcpy(ret, &pscore[0], sizeof(float)*m_nC);
	}

	return id;
}

//////////////////////////////////////////////////////////////////////////
CxMCBoostDetect::~CxMCBoostDetect()
{
	if( m_theLogit )
	{
		delete [] m_theLogit;
		m_theLogit = NULL;
	}

	if ( m_alphaList )
	{
		cvReleaseMat( &m_alphaList );
		m_alphaList = NULL;
	}

	if( m_exfea)
	{
		delete m_exfea;
		m_exfea = NULL;
	}

	if( m_pFea )
	{
		delete [] m_pFea;
		m_pFea = NULL;
	}
}

void CxMCBoostDetect::extFeature(IplImage* pCutFace, float* pFea)
{
	int round = m_defRound;
	int nDim = m_exfea->getFeaDim();

	if( m_pFea == NULL )
		m_pFea = new float[nDim];

	m_exfea->preproc(pCutFace);
	
	float* pFeaT = pFea;
	for(int i =0; i< round; i++)
	{
		m_exfea->extFeature(m_winList[i].rc, pFeaT);
		pFeaT = pFeaT + nDim;
	}
}

int CxMCBoostDetect::predict(IplImage* pCutFace, float *prob /* =NULL */)
{
	int round = m_defRound;
	int nDim = m_exfea->getFeaDim();

	if( m_pFea == NULL )
		m_pFea = new float[nDim];

	m_exfea->preproc(pCutFace);
	
	farray probA(m_nC);
	farray sum(m_nC);
	memset(&sum[0], 0, sizeof(float)*m_nC);
	for(int i=0; i<round; ++i)
	{
		m_exfea->extFeature(m_winList[i].rc, m_pFea);
		for(int k=0; k<nDim; ++k)
		{
			m_pFea[k] = (m_pFea[k] - m_winList[i].minv[k]) * m_winList[i].invdiff[k];
		}
		m_theLogit[i].predict(m_pFea, &probA[0]);
		if( USE_ALPHA == 0 || m_alphaList == NULL )
		{
			for(int j=0; j<m_nC; ++j)
				sum[j] += probA[j];
		}
		else
		{
			for(int j=0; j<m_nC; ++j)
				sum[j] += cvmGet(m_alphaList, 0, i) * probA[j];
		}
	}
	int id = 0;
	float maxv = 0;
	if( USE_ALPHA && m_alphaList != NULL )
	{
		float sumalpha = 0;
		for(int i=0; i<round; ++i)
		{
			sumalpha += cvmGet(m_alphaList, 0, i);
		}		
		for(int j=0; j<m_nC; ++j)
		{
			sum[j] /= sumalpha;
			if( sum[j] > maxv )
			{
				maxv = sum[j];
				id = j;
			}
		}
	}
	else
	{
		for(int j=0; j<m_nC; ++j)
		{
			sum[j] /= round;
			if( sum[j] > maxv )
			{
				maxv = sum[j];
				id = j;
			}
		}
	}
	if( prob != NULL )
	{
		for(int j=0; j<m_nC; ++j)
			prob[j] = (float)sum[j];		
	}
	return id;
}

int CxMCBoostDetect::predict(float* pFea, float *prob /* =NULL */)
{
	int round = m_defRound;
	int nDim = m_exfea->getFeaDim();

	if( m_pFea == NULL )
		m_pFea = new float[nDim];

	farray probA(m_nC);
	farray sum(m_nC);
	memset(&sum[0], 0, sizeof(float)*m_nC);
	float *pFeaT = pFea; 

	for(int i=0; i<round; ++i)
	{
		for(int k=0; k<nDim; ++k)
		{
			m_pFea[k] = (pFeaT[k] - m_winList[i].minv[k]) * m_winList[i].invdiff[k];
		}
		pFeaT += nDim;

		m_theLogit[i].predict(m_pFea, &probA[0]);

		if( USE_ALPHA == 0 || m_alphaList == NULL )
		{
			for(int j=0; j<m_nC; ++j)
				sum[j] += probA[j];
		}
		else
		{
			for(int j=0; j<m_nC; ++j)
				sum[j] += cvmGet(m_alphaList, 0, i) * probA[j];
		}
	}
	int id = 0;
	float maxv = 0;
	if( USE_ALPHA && m_alphaList != NULL )
	{
		float sumalpha = 0;
		for(int i=0; i<round; ++i)
		{
			sumalpha += cvmGet(m_alphaList, 0, i);
		}		
		for(int j=0; j<m_nC; ++j)
		{
			sum[j] /= sumalpha;
			if( sum[j] > maxv )
			{
				maxv = sum[j];
				id = j;
			}
		}
	}
	else
	{
		for(int j=0; j<m_nC; ++j)
		{
			sum[j] /= round;
			if( sum[j] > maxv )
			{
				maxv = sum[j];
				id = j;
			}
		}
	}
	if( prob != NULL )
	{
		for(int j=0; j<m_nC; ++j)
			prob[j] = (float)sum[j];		
	}

	return id;
}

int CxMCBoostDetect::predictDiff(float* pFea1, float* pFea2, float *prob /* =NULL */)
{
	int round = m_defRound;
	int nDim = m_exfea->getFeaDim();

	if( m_pFea == NULL )
		m_pFea = new float[nDim];

	farray probA(m_nC);
	farray sum(m_nC);
	memset(&sum[0], 0, sizeof(float)*m_nC);
	
	float *pFeaA = pFea1;
	float *pFeaB = pFea2;
	float normFeaA[256];
	float normFeaB[256];
	
	for(int i=0; i<round; ++i)
	{
		for(int k=0; k<nDim; ++k)
		{
			normFeaA[k] = (pFeaA[k] - m_winList[i].minv[k]) * m_winList[i].invdiff[k];
			normFeaB[k] = (pFeaB[k] - m_winList[i].minv[k]) * m_winList[i].invdiff[k];
		
			m_pFea[k] = fabsf(normFeaA[k]- normFeaB[k]);
		}
		pFeaA += nDim;
		pFeaB += nDim;

		m_theLogit[i].predict(m_pFea, &probA[0]);

		if( USE_ALPHA == 0 || m_alphaList == NULL )
		{
			for(int j=0; j<m_nC; ++j)
				sum[j] += probA[j];
		}
		else
		{
			for(int j=0; j<m_nC; ++j)
				sum[j] += cvmGet(m_alphaList, 0, i) * probA[j];
		}
	}
	int id = 0;
	float maxv = 0;
	if( USE_ALPHA && m_alphaList != NULL )
	{
		float sumalpha = 0;
		for(int i=0; i<round; ++i)
		{
			sumalpha += cvmGet(m_alphaList, 0, i);
		}		
		for(int j=0; j<m_nC; ++j)
		{
			sum[j] /= sumalpha;
			if( sum[j] > maxv )
			{
				maxv = sum[j];
				id = j;
			}
		}
	}
	else
	{
		for(int j=0; j<m_nC; ++j)
		{
			sum[j] /= round;
			if( sum[j] > maxv )
			{
				maxv = sum[j];
				id = j;
			}
		}
	}
	if( prob != NULL )
	{
		for(int j=0; j<m_nC; ++j)
			prob[j] = (float)sum[j];		
	}

	return id;
}

int CxMCBoostDetect::voteLabel(int face_trackid, int label)
{
	int label_out  = 0;
	int smooth_len = 65536;

	// assign result to item with same id or overwrite the last one
	bool bfind = false;
	int  idk = m_votebuff_idx;
	for(int k =0; k < NVOTEBUFF; k++)
	{	
		int cur_idx = (m_votebuff_idx-k +NVOTEBUFF) % NVOTEBUFF;
		if(face_trackid == m_votebuff_face_id[cur_idx])
		{
			bfind = true;
			idk   = cur_idx; 
			break;
		}
	}

	if(bfind == false) //overwrite case
	{
		m_votebuff_idx = idk = (idk+1) % NVOTEBUFF;
		m_votebuff_face_label[idk][0] = m_votebuff_face_label[idk][1]
		                              = m_votebuff_face_label[idk][2]
									  = m_votebuff_face_label[idk][3] = 0;
		m_votebuff_face_id[idk] = face_trackid;
		m_votebuff_face_label[idk][label] = 1;
	}
	else
	{
		m_votebuff_face_label[idk][label] = MIN(smooth_len, m_votebuff_face_label[idk][label] + 1);
	}

	std::vector<int>::iterator pVoteIter;
	pVoteIter = std::max_element(m_votebuff_face_label[idk].begin(), m_votebuff_face_label[idk].end()); // find the best match faces id with min dist value
	label_out = (int)(pVoteIter - m_votebuff_face_label[idk].begin());
	int ageMaxVote= m_votebuff_face_label[idk][label_out];

	if(ageMaxVote < 2)
		label_out = -1;  //not decided
	
	return label_out;
}

int CxMCBoostDetect::load( const char* path, const char* filename, int cutimg_size /*=128*/)
{
	m_defRound = 0;

	char fullname[256];
	sprintf(fullname,  "%s/%s", path, filename);
	FILE* fp = fopen(fullname, "rt");
	if( fp == NULL )
		return m_defRound;

	fscanf(fp, "%s\n", m_prefix);
	fscanf(fp, "%d %d %d %d %d %d %d\n", &m_nAlgo, &m_defRound, &m_fea_type, &m_fea_space, &m_fea_ng, &m_imgw, &m_nC);
	assert(m_defRound > 0 && m_nAlgo >= 0 && m_imgw > 1 && m_nC > 1 );
	
	m_imgh = m_imgw;
	m_alphaList = cvCreateMat(1, m_defRound, CV_32F);
	for(int i=0; i<m_defRound; ++i)
	{
		double v1;
		fscanf(fp, "%lf\n", &v1);
		cvmSet(m_alphaList, 0, i, v1);
	}

	m_winList.resize(m_defRound);
	for(int i=0; i<m_defRound; ++i)
	{
		CvRect rc;
		fscanf(fp, "%d %d %d %d %d %d\n", &m_winList[i].id, &m_winList[i].d, 
				&rc.x, &rc.y, &rc.width, &rc.height);
		assert( m_winList[i].d > 0 && m_winList[i].d <= 256 );

		//rescale m_imgw to cutimg_size by tao
		if(cutimg_size != m_imgw)
		{
			rc.x = rc.x * 1.0* cutimg_size / m_imgw;
			rc.y = rc.y * 1.0* cutimg_size / m_imgw; 
			rc.width = rc.width * 1.0* cutimg_size / m_imgw; 
			rc.height= rc.height* 1.0* cutimg_size / m_imgw; 
			if(rc.x+rc.width  > cutimg_size) rc.width--;
			if(rc.y+rc.height > cutimg_size) rc.height--;
		}
	
		m_winList[i].rc = rc;
		double v1, v2;
		for(int k=0; k<m_winList[i].d; ++k)
		{
			fscanf(fp, "%lf %lf\n", &v1, &v2);

			m_winList[i].minv[k] = v1;
			m_winList[i].maxv[k] = v2;
			if( (v2 - v1) > 0 )
				m_winList[i].invdiff[k] = 1.0/(v2-v1);
			else
				m_winList[i].invdiff[k] = 0.0;
		}
	}
	fclose(fp);

	assert( m_nAlgo == 2 );

	// determine algorithm
	printf("Cascade-Logit\n");
	m_theLogit = new CxLogit[m_defRound];

	char fname[256];
	for(int i=0; i<m_defRound; ++i)
	{
		sprintf(fname, "%s/%s_log_%02d.xml", path, m_prefix, i);
		m_theLogit[i].load(fname);
	}
	
	if( m_exfea == NULL )
	{
		m_exfea = new CxSlideWinFeature();
		m_exfea->setFeaParam(m_fea_type);
	}
	m_fea_dim = m_exfea->getFeaDim() * m_defRound;

	return m_defRound;
}
