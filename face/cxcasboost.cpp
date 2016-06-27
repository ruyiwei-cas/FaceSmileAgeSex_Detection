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

#include "cxcasboost.hpp"
#include "vecmath.hpp"
#include "cxfaceutil.hpp"
#include "cxdelighting.hpp"

#if defined (_OPENMP) && defined (__ICL)
#include <omp.h>
#endif

#ifndef WIN32
#define stricmp strcasecmp
#endif

using namespace std;
////////////////////////////////////////////////////////////////////////////////
const float g_localrc[] =
{
	8, 8, 48, 48,
	56, 0, 64, 64,
	60, 52, 32, 32,
	40, 24, 32, 32,
	0, 0, 32, 32,
	48, 52, 32, 32,
	72, 24, 32, 32,
	48, 8, 48, 48,
	12, 80, 48, 48,
	96, 20, 32, 32,
	0, 0, 48, 48,
	8, 20, 48, 48,
	36, 40, 80, 80,
	0, 32, 32, 32,
	20, 24, 32, 32,
	68, 72, 32, 32,
	36, 68, 48, 48,
	56, 20, 32, 32,
	96, 0, 32, 32,
	72, 64, 32, 32,
	68, 52, 32, 32,
	0, 44, 64, 64,
	52, 8, 64, 64,
	32, 32, 32, 32,
	24, 12, 80, 80,
	48, 12, 32, 32,
	72, 4, 32, 32,
	92, 32, 32, 32,
	0, 8, 48, 48,
	80, 68, 32, 32,
	0, 40, 48, 48,
	20, 28, 32, 32,
	28, 88, 32, 32,
	16, 4, 80, 80,
	44, 64, 32, 32,
	8, 12, 32, 32,
	80, 92, 32, 32,
	0, 4, 64, 64,
	48, 32, 32, 32,
	24, 60, 32, 32,
	44, 24, 48, 48,
	60, 72, 48, 48,
	72, 92, 32, 32,
	52, 20, 32, 32,
	4, 76, 32, 32,
	84, 68, 32, 32,
	0, 16, 48, 48,
	48, 40, 48, 48,
	44, 24, 32, 32,
	32, 24, 32, 32,
	32, 68, 32, 32,
	76, 60, 32, 32,
	0, 0, 128, 128,
	48, 40, 32, 32,
	48, 28, 80, 80,
	28, 0, 32, 32,
	68, 76, 32, 32,
	84, 36, 32, 32,
	68, 64, 48, 48,
	48, 72, 48, 48,
	16, 0, 48, 48,
	20, 20, 32, 32,
	8, 96, 32, 32,
	36, 20, 64, 64,
	24, 16, 32, 32,
	0, 28, 48, 48,
	80, 4, 48, 48,
	36, 40, 48, 48,
	20, 56, 48, 48,
	60, 68, 32, 32,
	68, 36, 32, 32,
	60, 60, 32, 32,
	12, 20, 48, 48,
	4, 40, 32, 32,
	20, 48, 48, 48,
	68, 0, 48, 48,
	48, 16, 32, 32,
	56, 36, 32, 32,
	24, 16, 48, 48,
	36, 92, 32, 32,
	44, 12, 48, 48,
	40, 24, 48, 48,
	28, 20, 80, 80,
	96, 36, 32, 32,
	12, 52, 32, 32,
	0, 16, 64, 64,
	96, 80, 32, 32,
	68, 56, 48, 48,
	24, 96, 32, 32,
	16, 24, 80, 80,
	68, 12, 32, 32,
	36, 32, 32, 32,
	48, 80, 32, 32,
	4, 68, 32, 32,
	92, 0, 32, 32,
	0, 88, 32, 32,
	40, 32, 80, 80,
	48, 8, 32, 32,
	96, 16, 32, 32,
	64, 44, 32, 32,
	24, 4, 32, 32,
	56, 44, 48, 48,
	28, 16, 80, 80,
	36, 0, 32, 32,
	48, 0, 32, 32,
	52, 20, 64, 64,
	64, 52, 64, 64,
	72, 68, 32, 32,
	4, 48, 80, 80,
	84, 24, 32, 32,
	56, 0, 48, 48,
	4, 36, 32, 32,
	56, 44, 32, 32,
	96, 44, 32, 32,
	20, 12, 32, 32,
	68, 88, 32, 32,
	76, 84, 32, 32,
	24, 88, 32, 32,
	16, 16, 80, 80,
	68, 56, 32, 32,
	8, 4, 32, 32,
	20, 16, 48, 48,
	28, 12, 80, 80,
	84, 72, 32, 32,
	44, 48, 80, 80,
	20, 68, 32, 32,
	52, 48, 64, 64,
	56, 72, 48, 48,
	72, 64, 48, 48,
	72, 80, 48, 48,
	48, 64, 48, 48,
	68, 68, 32, 32,
	76, 44, 32, 32,
	24, 48, 32, 32,
	44, 56, 64, 64,
	4, 44, 48, 48,
	16, 40, 64, 64,
	24, 32, 32, 32,
	72, 28, 32, 32,
	72, 44, 48, 48,
	64, 96, 32, 32,
	20, 20, 48, 48,
	52, 32, 32, 32,
	48, 64, 64, 64,
	8, 52, 32, 32,
	12, 52, 48, 48,
	48, 36, 80, 80,
	60, 92, 32, 32,
	72, 96, 32, 32,
	48, 92, 32, 32,
	8, 28, 48, 48,
	36, 76, 32, 32,
	52, 28, 32, 32,
	56, 52, 48, 48,
	72, 32, 48, 48,
	60, 68, 48, 48,
	48, 68, 32, 32,
	72, 52, 32, 32,
	40, 60, 48, 48,
	32, 4, 32, 32,
	32, 96, 32, 32,
	52, 36, 32, 32,
	72, 76, 48, 48,
	48, 60, 48, 48,
	56, 88, 32, 32,
	56, 20, 48, 48,
	44, 8, 32, 32,
	24, 64, 32, 32,
	20, 4, 32, 32,
	20, 32, 80, 80,
	72, 56, 48, 48,
	88, 72, 32, 32,
	48, 20, 80, 80,
	56, 84, 32, 32,
	40, 56, 32, 32,
	88, 40, 32, 32,
	72, 68, 48, 48,
	12, 32, 32, 32,
	32, 12, 32, 32,
	24, 36, 32, 32,
	20, 28, 48, 48,
	20, 92, 32, 32,
	64, 60, 32, 32,
	56, 96, 32, 32,
	20, 52, 32, 32,
	76, 52, 48, 48,
	0, 24, 48, 48,
	64, 32, 32, 32,
	80, 72, 32, 32,
	48, 76, 32, 32,
	56, 60, 32, 32,
	84, 32, 32, 32,
	44, 36, 80, 80,
	16, 24, 64, 64,
	68, 32, 32, 32,
	4, 4, 48, 48,
	28, 24, 32, 32,
	0, 48, 64, 64,
	0, 16, 32, 32,
	44, 72, 48, 48,
	92, 20, 32, 32,
	44, 8, 48, 48,
	72, 60, 32, 32,
	24, 28, 48, 48,
	32, 72, 32, 32,
	72, 72, 32, 32,
	44, 0, 32, 32,
	80, 64, 48, 48,
	12, 28, 32, 32,
	28, 32, 80, 80,
	4, 20, 32, 32,
	12, 16, 32, 32,
	68, 68, 48, 48,
	0, 32, 64, 64,
	40, 68, 48, 48,
	28, 24, 48, 48,
	76, 0, 48, 48,
	20, 4, 64, 64,
	36, 56, 32, 32,
	0, 8, 64, 64,
	16, 20, 80, 80,
	84, 52, 32, 32,
	68, 4, 32, 32,
	64, 88, 32, 32,
	76, 68, 32, 32,
	44, 28, 32, 32,
	40, 20, 64, 64,
	20, 40, 32, 32,
	60, 88, 32, 32,
	28, 8, 80, 80,
	52, 16, 32, 32,
	36, 0, 48, 48,
	32, 52, 32, 32,
	32, 80, 48, 48,
	64, 48, 32, 32,
	4, 64, 32, 32,
	8, 56, 32, 32,
	32, 24, 64, 64,
	80, 0, 32, 32,
	52, 72, 48, 48,
	8, 80, 32, 32,
	64, 20, 32, 32,
	80, 60, 48, 48,
	40, 52, 32, 32,
	64, 4, 64, 64,
	44, 4, 64, 64,
	4, 44, 80, 80,
	0, 76, 48, 48,
	12, 60, 48, 48,
	60, 28, 32, 32,
	72, 76, 32, 32,
	0, 44, 48, 48,
	36, 72, 48, 48,
	16, 56, 48, 48,
	36, 24, 48, 48,
	88, 52, 32, 32,
	8, 24, 32, 32,
	68, 60, 48, 48,
	36, 8, 32, 32,
	0, 56, 32, 32,
	0, 0, 96, 96,
	16, 0, 96, 96,
	32, 0, 96, 96,
	0, 16, 96, 96,
	16, 16, 96, 96,
	32, 16, 96, 96,
	0, 32, 96, 96,
	16, 32, 96, 96,
	32, 32, 96, 96,
	0, 0, 64, 64,
	16, 0, 64, 64,
	32, 0, 64, 64,
	48, 0, 64, 64,
	64, 0, 64, 64,
	16, 16, 64, 64,
	32, 16, 64, 64,
	48, 16, 64, 64,
	64, 16, 64, 64,
	16, 32, 64, 64,
	32, 32, 64, 64,
	48, 32, 64, 64,
	64, 32, 64, 64,
	16, 48, 64, 64,
	32, 48, 64, 64,
	48, 48, 64, 64,
	64, 48, 64, 64,
	0, 64, 64, 64,
	16, 64, 64, 64,
	32, 64, 64, 64,
	64, 64, 64, 64,
	32, 0, 48, 48,
	48, 0, 48, 48,
	64, 0, 48, 48,
	80, 0, 48, 48,
	16, 16, 48, 48,
	32, 16, 48, 48,
	48, 16, 48, 48,
	64, 16, 48, 48,
	80, 16, 48, 48,
	0, 32, 48, 48,
	16, 32, 48, 48,
	32, 32, 48, 48,
	48, 32, 48, 48,
	64, 32, 48, 48,
	80, 32, 48, 48,
	0, 48, 48, 48,
	16, 48, 48, 48,
	32, 48, 48, 48,
	48, 48, 48, 48,
	64, 48, 48, 48,
	80, 48, 48, 48,
	0, 64, 48, 48,
	16, 64, 48, 48,
	32, 64, 48, 48,
	64, 64, 48, 48,
	0, 80, 48, 48,
	16, 80, 48, 48,
	48, 80, 48, 48,
	64, 80, 48, 48,
	80, 80, 48, 48
};
//////////////////////////////////////////////////////////////////////////
CxBaseReg::~CxBaseReg()
{
	clear();
}

void CxBaseReg::clear()
{
	cvFree(&m_linearModel);
	cvFree(&m_rcweight);
}

int CxBaseReg::load(const char *filename)
{
	char name[256] = {0};
	FILE* fp = fopen(filename, "rt");
	if( fp == NULL )
		return 0;

	fscanf(fp, "%s %s\n", name, m_prefix);
	if( stricmp(name, "class") != 0 )
	{
		fclose(fp);
		printf("error in the model\n");
		exit(0);
	}
	memset(name, 0, sizeof(char)*256);
	fscanf(fp, "%s %d\n", name, &m_nAlgo);
	if( stricmp(name, "algo") != 0 || m_nAlgo < 0)
	{
		fclose(fp);
		printf("error in the model\n");
		exit(0);
	}
	memset(name, 0, sizeof(char)*256);
	fscanf(fp, "%s %d\n", name, &m_nRC);
	if( stricmp(name, "round") != 0 || m_nRC <= 0)
	{
		fclose(fp);
		printf("error in the model\n");
		exit(0);
	}

	memset(name, 0, sizeof(char)*256);
	fscanf(fp, "%s %d %d %d %d %d\n", name, &m_fea_type, &m_fea_space, &m_fea_ng, &m_imgw, &m_imgh);
	if( stricmp(name, "feature") != 0 || m_imgw <= 1 || m_imgh <= 1 )
	{
		fclose(fp);
		printf("error in the model\n");
		exit(0);
	}

	double val, v1, v2, v3;
	memset(name, 0, sizeof(char)*256);
	fscanf(fp, "%s %lf\n", name, &val);
	if( stricmp(name, "threshold") != 0 )
	{
		fclose(fp);
		printf("error in the model\n");
		exit(0);
	}
	m_threshold = val;

	// for section: auc_tpr_fpr
	memset(name, 0, sizeof(char)*256);	
	fscanf(fp, "%s %lf %lf %lf %lf\n", name, &val, &v1, &v2, &v3);
	m_auc = val;
	m_eer = v1;
	m_tpr = v2;
	m_fpr = v3;
	if( stricmp(name, "auc_eer_tpr_fpr") != 0 || m_auc <= 0 || m_auc > 1.0 )
	{
		fclose(fp);
		printf("error in the model\n");
		exit(0);
	}

	m_linearModel = (tagLinearModel*)cvAlloc(sizeof(tagLinearModel)*m_nRC);
	memset(m_linearModel, 0, sizeof(tagLinearModel) * m_nRC);
	memset(&m_winList, 0, sizeof(tagUsedWinGL));

	int d = 0;
	memset(name, 0, sizeof(char)*256);
	fscanf(fp, "%s %d\n", name, &d);
	if( stricmp(name, "rect") != 0 || d<=0 || d > MAX_FEA_DIM )
	{
		fclose(fp);
		printf("error in the model\n");
		exit(0);
	}
	m_winList.d = d;
	int rcd = d/m_nRC;

	m_d = d;
	for(int k=0; k<d; ++k)
	{
		fscanf(fp, "%lf %lf\n", &v1, &v2);
		m_winList.minv[k] = v1;
		m_winList.maxv[k] = v2;
		float diff = (m_winList.maxv[k]- m_winList.minv[k]);
		if( diff > FLT_EPSILON )
			m_scl[k] = 1.0f/diff;
		else
			m_scl[k] = 0;
	}
	memset(name, 0, sizeof(char)*256);
	fscanf(fp, "%s\n", name);
	if( stricmp(name, "linear_weight") != 0 )
	{
		fclose(fp);
		printf("error in the model\n");
		exit(0);
	}
	for(int i=0; i<m_nRC; ++i)
	{
		m_linearModel[i].nC = 2;
		m_linearModel[i].d = rcd;
		for(int k=0; k<rcd; ++k)
		{
			fscanf(fp, "%lf\n", &v1);
			m_linearModel[i].beta[k] = v1;
		}
		fscanf(fp, "%lf %lf %lf\n", &v1, &v2, &v3);		
		m_linearModel[i].bias = v1;
		m_linearModel[i].sigmoid[0] = v2;
		m_linearModel[i].sigmoid[1] = v3;
	}
	m_rcweight = (float*)cvAlloc(sizeof(float)*m_nRC);
	for(int i=0; i<m_nRC; ++i)
	{
		fscanf(fp, "%lf\n", &v1);
		m_rcweight[i] = v1;
	}
	fclose(fp);

	return 1;
}

int CxBaseReg::predict(float* pFeaDiff, float* retprob)
{
	int d = m_d;
	float prob = 0;
	int k, ret = 0;
	float CX_DECL_ALIGNED(16) pDecVal[128] = {0};
	float scale = 2.0f;
	for(k=0; k<d; k++)
	{
		// limited the range is useful
		float scl = m_scl[k];
		pFeaDiff[k] = scale*((pFeaDiff[k] - m_winList.minv[k]) * scl - 0.5f);
		pFeaDiff[k] = min(pFeaDiff[k], 0.5f*scale);
		pFeaDiff[k] = max(pFeaDiff[k], -0.5f*scale);
	}

	// weighting
	int rcd = d/m_nRC;
	float sumprob = 0;
	for(int i=0; i<m_nRC; ++i)
	{
		pDecVal[i] = icxLinearWX(&m_linearModel[i], pFeaDiff + i*rcd);
		if( pDecVal[i] > 25 )
			pDecVal[i] = 25;
		if( pDecVal[i] < -25 )
			pDecVal[i] = -25;

		float val = expf(-pDecVal[i]);
		sumprob += m_rcweight[i] * 1.0f/(1.0f +val);
	}
	prob = sumprob;

	if( prob > m_threshold ) ret = 1;
	if( retprob != NULL ) *retprob = prob;

	return ret;
}



// cascade detection
CxCasDetect::~CxCasDetect()
{
	if( m_baseReg != NULL ) 
	{
		delete [] m_baseReg;
		m_baseReg = NULL;
	}

	if(m_exfea != NULL)
	{
		delete m_exfea;
		m_exfea = NULL;
	}
	cvFree(&m_pFea);
	cvFree(&m_pFeaTmp);
	cvFree(&m_mu);
	cvFree(&m_prjmat);

	if(m_pImgNorm)
	{
		cvReleaseImage(&m_pImgNorm);
		m_pImgNorm = NULL;
	}
}

int CxCasDetect::load(const char* path, const char* filename, int cutimg_size /*=128*/)
{
	if( m_baseReg != NULL )
		delete [] m_baseReg;

	char modname[256] = {0};
	sprintf(modname, "%s/%s", path, filename);
	FILE *fp = fopen(modname, "rt");
	if(fp == NULL) return -1;

	int   sta = 0;
	int   nUsedRound= 1;
	int   diff_type = 6;
	float threshold = 0.978236f;
	sta = fscanf(fp, "stage %d\n",      &nUsedRound);
	sta = fscanf(fp, "diff_type  %d\n", &diff_type);
	sta = fscanf(fp, "threshold  %f\n",  &threshold);

	fclose(fp);

	m_defRound     = max(nUsedRound, 1);
	m_fea_type     = 0, m_fea_space = 0, m_fea_ng = 0;
	m_d			   = 0;
	m_diff_type    = diff_type;
	m_defThreshold = threshold;

	m_baseReg = new CxBaseReg[m_defRound];
	for(int i=0; i<m_defRound; ++i)
	{
		sprintf(modname, "%s/faceglb_cas%02d.model", path, i);
		if( m_baseReg[i].load(modname) == 0 )
		{
			m_defRound = i;
			if( m_defRound < 1 )
			{
				printf("Error in loading model\n");
				exit(0);
			}
			break;
		}

		if( i == 0 )
		{
			m_fea_type  = m_baseReg[0].m_fea_type;
			m_fea_space = m_baseReg[0].m_fea_space;
			m_fea_ng    = m_baseReg[0].m_fea_ng;

			m_imgw      = m_baseReg[0].m_imgw;
			m_imgh      = m_baseReg[0].m_imgh;
			m_d		    = m_baseReg[0].getFeatureDim();
		}
		m_baseReg[i].m_diff_type = diff_type;
	}
	if( m_exfea == NULL )
	{
		m_exfea = new CxSlideWinFeature();
		m_exfea->setFeaParam(m_fea_type);
	}
	m_pFea  = (float*)cvAlloc(sizeof(float)*m_d*2); //m_pFea = (float*)cvAlloc(sizeof(float)*MAX_FEA_DIM*2);
	m_pFeaTmp = (float*)cvAlloc(sizeof(float)*m_d*2);
	
	char prjname[256] = {0};
	sprintf(prjname, "%s/faceglb_blockprj.bmd", path);
	load8uPrjMat(prjname);

	return m_defRound;
}

// projection matrix in uchar format
void CxCasDetect::load8uPrjMat(const char* modname)
{
	int head[8] = {0};
	FILE* fp = fopen(modname, "rb");
	if( fp == NULL )
	{
		printf("error\n");
		exit(0);
	}
	fread(head, sizeof(int), 8, fp);
	m_odim = head[0];
	m_prjdim = head[1];
	m_nRC = head[2];
	assert( m_odim >0 && m_prjdim > 0 && m_nRC > 0 );

	if( m_mu != NULL ) cvFree(&m_mu);
	m_mu = (float*)cvAlloc(sizeof(float)*m_odim*m_nRC);
	fread(m_mu, sizeof(float), m_odim*m_nRC, fp);

	float minmaxv[1024*2] = {0};
	fread(minmaxv, sizeof(float), 2*m_nRC, fp);

	uchar* uprjmat = (uchar*)malloc(sizeof(uchar)*m_odim*m_prjdim * m_nRC);
	fread(uprjmat, sizeof(uchar), m_odim*m_prjdim*m_nRC, fp);
	fclose(fp);

	//////////////////////////////////////////////////////////////////////////
	if( m_prjmat != NULL ) cvFree(&m_prjmat);
	m_prjmat = (float*)cvAlloc(sizeof(float)*m_odim*m_prjdim * m_nRC);

	// decode
	for(int k=0; k<m_nRC; ++k)
	{
		CvMat uldaPrj = cvMat(m_prjdim, m_odim, CV_8U, uprjmat + k*m_prjdim*m_odim );
		CvMat ldaPrj = cvMat(m_prjdim, m_odim, CV_32F, m_prjmat + k*m_prjdim*m_odim );		

		float minv = minmaxv[2*k +0];
		float maxv = minmaxv[2*k +1];
		for(int i=0; i<m_prjdim; ++i) // row
		{
			for(int j=0; j<m_odim; ++j)
			{
				float v = CV_MAT_ELEM(uldaPrj, uchar, i, j);
				if( maxv - minv > 0 )
				{
					v = v/255.0 *(maxv - minv) + minv;
				}
				else
					v = 0;
				CV_MAT_ELEM(ldaPrj, float, i, j) = v;
			}
		}
	}
	free(uprjmat);
	return ;
}

// y = A*x, where A is (rxd), i.e., r-row, and d-column
inline void cblasAx(int r, int d, float* A, float *x, float* y)
{
	int i, j;
	int dsse = (d/4)*4;
	for(i=0; i<r; ++i) // rows
	{
		j=0;
		y[i] = 0;
		float* z = A + i*d;	// z is the i-th row of matrix A

#ifdef __SSE2__
		F128 xmm_sum;
		xmm_sum.pack = _mm_set1_ps(0);
		for(; j<dsse; j+=4)
		{
			__m128 xmm_a = _mm_mul_ps(_mm_loadu_ps(z+j), _mm_loadu_ps(x+j));
			xmm_sum.pack = _mm_add_ps(xmm_sum.pack, xmm_a);
		}
		y[i] = xmm_sum.f[0]+ xmm_sum.f[1]+ xmm_sum.f[2]+ xmm_sum.f[3];
#endif
		for(; j<d; j++)
		{
			y[i] += x[j]*z[j];
		}
	}
}

int CxCasDetect::predict(float *pDiffFea,  float* retprob /*= NULL*/)
{
	int   cnt = 0, ret = 0;
	float sum = 0;
	for(int i=0; i<m_defRound; ++i)
	{
		float prob = 0;
		int ret = m_baseReg[i].predict(pDiffFea, &prob);
		sum += prob;
		cnt += ret;
	}
	sum /= m_defRound;
	if( retprob != NULL ) *retprob = sum;
	if(cnt > m_defRound/2) ret = 1;

	return ret;
}

int CxCasDetect::predictDiff(float* aFea, float* bFea, float* retprob /*= NULL*/)
{
	float* pFeaDiff = m_pFea;
	float* aFeaF = m_pFeaTmp;
	float* bFeaF = m_pFeaTmp+m_d;
	uchar* aFeaU = (uchar*)aFea;
	uchar* bFeaU = (uchar*)bFea;

	for(int i = 0; i < m_d; i++)
	{
		aFeaF[i] = aFeaU[i]; 
		bFeaF[i] = bFeaU[i];
	}

	pairwise_feature_diff(aFeaF, bFeaF, pFeaDiff, m_d, m_diff_type);

	int cnt = 0, ret = 0;
	float sum = 0;
	for(int i=0; i<m_defRound; ++i)
	{
		float prob = 0;
		int ret = m_baseReg[i].predict(pFeaDiff, &prob);
		sum += prob;
		cnt += ret;
	}
	sum /= m_defRound;
	if( retprob != NULL ) *retprob = sum;
	if(cnt > m_defRound/2) ret = 1;

	return ret;
}

void CxCasDetect::extFeature(IplImage* pCutFace, float* pFea)
{
	if(m_pImgNorm == NULL || m_pImgNorm->width != pCutFace->width || m_pImgNorm->height != pCutFace->height )
		m_pImgNorm = cvCloneImage(pCutFace);

//	cvSaveImage("c:/temp/beforenorm.jpg", pCutFace);
	cvLightNormalize( pCutFace, m_pImgNorm, LIGHT_NORM_DCT);	
//	cvSaveImage("c:/temp/afternorm.jpg", m_pImgNorm);
	//////////////////////////////////////////////////////////////////////////
	// feature extraction
	int d0 = m_exfea->getFeaDim();
	int imgw = m_imgw;
	int imgh = m_imgh;
	int i, k;

	std::vector<CvRect> globalRC;
	globalRC.reserve(240);
	for(i=0; i<240; ++i)
	{
		CvRect rc = cvRect(g_localrc[4*i], g_localrc[4*i+1], g_localrc[4*i+2], g_localrc[4*i+3]);
		globalRC.push_back(rc);
	}
	int nrc = globalRC.size();
	assert( d0*nrc == m_d );

	float* pFeaA = m_pFea;
	m_exfea->preproc(m_pImgNorm);
	for(k=0; k<nrc; ++k)
	{
		m_exfea->extFeature(globalRC[k], pFeaA + (k)*d0);
	}
	//////////////////////////////////////////////////////////////////////////
	// projection to discriminant space
	i = 0;
#ifdef __SSE2__
	for(; i<(m_d/4)*4; i+=4)
	{
		_mm_storeu_ps(pFeaA+i,_mm_sub_ps(_mm_loadu_ps(pFeaA+i), _mm_loadu_ps(m_mu+i)) );		
	}
#endif
	for(; i<m_d; ++i)
	{
		pFeaA[i] -= m_mu[i];
	}

	float* pDstFea = m_pFea + m_d;
	for(k=0; k<m_nRC; ++k)
	{
		float* A = m_prjmat + k*m_prjdim*m_odim;
		float* x = pFeaA + k * m_odim;
		float* y = pDstFea + k * m_prjdim;
		cblasAx(m_prjdim, m_odim, A, x, y);
	}

	int p = m_prjdim*m_nRC;
	// the peak scale is around 1.6 (best)
	float peak_threshold = 1.6f/sqrtf(m_prjdim);
	// normalize each segment separately
	for(k=0; k<m_nRC; ++k)
	{
		float* pF = pDstFea + k*m_prjdim;
		normalizeVector(pF, m_prjdim, NORM_L2HYS, peak_threshold);
		normalizeVector(pF, m_prjdim, NORM_L2HYS, peak_threshold);
		quantizeHist(pF, m_prjdim, 8, peak_threshold, false);
	}

	uchar *pFeaU = (uchar *)pFea;
	for(i=0; i<p; ++i)
	{
		//pFea[i] = (uchar)(pDstFea[i] + 128);
		pFeaU[i] = (uchar)(pDstFea[i] + 128);
	}
}
