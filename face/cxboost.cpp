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
#include <stdio.h>

#include "cxboost.hpp"
#include "cxfaceutil.hpp"

//////////////////////////////////////////////////////////////////////////
CxBoostDetect::~CxBoostDetect()
{
	if( m_ann )
	{
		for(int i=0; i<m_defRound; ++i)
			fann_destroy(m_ann[i]);
		delete [] m_ann;
		m_ann = NULL;
	}

	if( m_exfea)
	{
		delete m_exfea;
		m_exfea = NULL;
	}

	if( m_pFea )
	{
#ifdef	__SSE2__
		_mm_free(m_pFea);
		m_pFea = NULL;
#else		
		delete [] m_pFea;
		m_pFea = NULL;
#endif
	}
}

int CxBoostDetect::voteLabel(int face_trackid, int label, int vote_threshold /*=2*/, int smooth_len /*=8*/)
{
	int label_out  = 0;
	
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
		label_out = label;
		m_votebuff_face_id[idk] = face_trackid;
		m_votebuff_face_label[idk] = label;
	}
	else
	{
		label_out = m_votebuff_face_label[idk] + label;
		if(label_out > smooth_len)
			label_out = smooth_len;
		else if(label_out < -smooth_len) 
			label_out = -smooth_len;

		m_votebuff_face_label[idk] = label_out;
	}

	// output label
	if(label_out > vote_threshold)       label_out =  1;
	else if(label_out < -vote_threshold) label_out = -1;
	else label_out = 0;

	return label_out;
}

int CxBoostDetect::load(const char* path, const char* filename, int cutimg_size /*=64*/)
{
	m_defRound = 0;

	char fullname[256];
	sprintf(fullname,  "%s/%s", path, filename);
	//printf("Load: %s\n", fullname);

	FILE* fp = fopen(fullname, "rt");
	if( fp == NULL )
		return m_defRound;

	int max_round = 0;
	fscanf(fp, "%s\n", m_prefix);
	int sta = fscanf(fp, "%d %d %d %d %d %f\n", &m_nAlgo, &max_round, &m_fea_type, &m_imgw, &m_defRound, &m_defThreshold);

	//printf("m_defRound = %d\n", m_defRound);
	if(sta != 6 || m_defRound <= 0 || m_nAlgo < 4 )
	{
		fclose(fp);
		return m_defRound;
	}

	m_imgh = m_imgw;
	assert( m_defRound <= max_round );

	m_winList.resize(m_defRound);
	for(int i=0; i<m_defRound; ++i)
	{
		CvRect rc;
		fscanf(fp, "%d %d %d %d %d\n", &m_winList[i].id, &rc.x, &rc.y, &rc.width, &rc.height);
		
		//rescale reconizer model from default m_imgw to cutimg_size
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

		float v1, v2;
		int    win_num = (m_fea_type != FEA_GABOR320) ? 256 : 512;
		for(int k = 0; k < win_num; ++k)
		{
			fscanf(fp, "%f %f\n", &v1, &v2);

			m_winList[i].minv[k] = v1;
			m_winList[i].maxv[k] = v2;
			if ((v2 - v1) > 0.0f)
				m_winList[i].inv_diffv[k] = 1.0f/(v2 - v1);
			else
				m_winList[i].inv_diffv[k] = 0.0f;
		}
	}
	fclose(fp);

	assert( m_nAlgo >= 4 );
	m_ann = new struct fann* [m_defRound];

	char annname[256];
	for(int i = 0; i < m_defRound; ++i)
	{
		m_ann[i] = NULL;
		sprintf(annname, "%s/%s_ann_%02d.net", path, m_prefix, i);
		//printf("Load: %s\n",annname);modify by ruyiwei
		m_ann[i] = fann_create_from_file(annname);
		if( m_ann[i] == NULL )
		{
			m_defRound = i;
			break;
		}
	}

	if( m_exfea == NULL )
	{
		m_exfea = new CxSlideWinFeature();
		m_exfea->setFeaParam(m_fea_type);
	}

	m_fea_dim = m_exfea->getFeaDim() * m_defRound;

	//printf("init result: %d\n", m_defRound); modify by ruyiwei
	return m_defRound;
}

void CxBoostDetect::extFeature(IplImage* pCutFace, float* pFea)
{
	int   round = m_defRound;
	int   nDim  = m_exfea->getFeaDim();

	m_exfea->preproc(pCutFace);
	
	float* pFeaT = pFea;
	for(int i =0; i< round; i++)
	{
		m_exfea->extFeature(m_winList[i].rc, pFeaT);
		pFeaT = pFeaT + nDim;
	}
}

int CxBoostDetect::predict(IplImage* pCutFace, float *retprob /* =NULL */)
{
	int ret = 0;
	int round = m_defRound;
	int nDim = m_exfea->getFeaDim();

	//alloc fea buff m_pFea
#ifdef	__SSE2__
	if( m_pFea == NULL )
		m_pFea = (float *)_mm_malloc(sizeof(float)*nDim,16);
#else
	if( m_pFea == NULL )
		m_pFea = new float[nDim];
#endif

	//preproc
	m_exfea->preproc(pCutFace);
	
	float probB[2], sum = 0, prob = 0;
	for(int i=0; i<round; ++i)
	{
		m_exfea->extFeature(m_winList[i].rc, m_pFea);

		for(int k=0; k<nDim; ++k)
		{
			m_pFea[k] = (m_pFea[k] - m_winList[i].minv[k]) * m_winList[i].inv_diffv[k];
		}

		fann_type* out = fann_run(m_ann[i], m_pFea);
		probB[0] = out[0];
		probB[1] = out[1];
		icvMLP2Prob(probB);
		
		sum += probB[0];
	}
	
	prob = sum / round;
	if( retprob != NULL ) *retprob = prob;

	if( prob > m_defThreshold )   ret = 1;

	return ret;
}

int CxBoostDetect::predict(float *pFea,  float* retprob /*= NULL*/)
{
	int ret = 0;
	int round = m_defRound;
	int nDim = m_exfea->getFeaDim();

#ifdef	__SSE2__
	if( m_pFea == NULL )
		m_pFea = (float *)_mm_malloc(sizeof(float)*nDim,16);
#else
	if( m_pFea == NULL )
		m_pFea = new float[nDim];
#endif

	float probB[2], sum = 0, prob = 0;
	float *pFeaT = pFea; 

	for(int i=0; i<round; ++i)
	{
		for(int k=0; k<nDim; ++k)
		{
			m_pFea[k] = (pFeaT[k] - m_winList[i].minv[k]) * m_winList[i].inv_diffv[k];
		}
		pFeaT += nDim;

		fann_type* out = fann_run(m_ann[i], m_pFea);
		probB[0] = out[0];
		probB[1] = out[1];
		icvMLP2Prob(probB);

		sum += probB[0];
	}
	
	prob = sum / round;
	if( retprob != NULL ) *retprob = (float)prob;
	
	if( prob > m_defThreshold )   ret = 1;

	return ret;
}

int CxBoostDetect::predictDiff(float *pFea1, float *pFea2, float * retprob /*= NULL*/)
{
	int ret = 0;
	int round = m_defRound;
	int nDim = m_exfea->getFeaDim();

#ifdef	__SSE2__
	if( m_pFea == NULL )
		m_pFea = (float *)_mm_malloc(sizeof(float)*nDim,16);
#else
	if( m_pFea == NULL )
		m_pFea = new float[nDim];
#endif

	float probB[2], sum = 0, prob = 0;
	float *pFeaA = pFea1;
	float *pFeaB = pFea2;
	float normFeaA[256];
	float normFeaB[256];

	for(int i=0; i<round; ++i)
	{
		for(int k=0; k<nDim; ++k)
		{
			if( m_winList[i].maxv[k] - m_winList[i].minv[k] > 0 )
				normFeaA[k] = (pFeaA[k] - m_winList[i].minv[k]) * m_winList[i].inv_diffv[k];
			else
				normFeaA[k] = 0;

			if( m_winList[i].maxv[k] - m_winList[i].minv[k] > 0 )
				normFeaB[k] = (pFeaB[k] - m_winList[i].minv[k]) * m_winList[i].inv_diffv[k];
			else
				normFeaB[k] = 0;

			m_pFea[k] = fabsf(normFeaA[k]- normFeaB[k]);
/*
			m_pFea[k] = fabsf(pFeaA[k]- pFeaB[k]);
			m_pFea[k] = (m_pFea[k] - m_winList[i].minv[k]) * m_winList[i].inv_diffv[k];
*/
		}

		pFeaA += nDim;
		pFeaB += nDim;

		fann_type* out = fann_run(m_ann[i], m_pFea);
		probB[0] = out[0];
		probB[1] = out[1];
		icvMLP2Prob(probB);

		sum += probB[0];
	}

	prob = sum / round;
	if( retprob != NULL ) *retprob = (float)prob;

	if( prob > m_defThreshold )   ret = 1;

	return ret;
}


