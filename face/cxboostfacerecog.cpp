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

#include "cxboostfacerecog.hpp"
#include "cxfaceutil.hpp"
using namespace std;


#ifdef _WIN32
#include <direct.h>
#endif

#define USE_CVHIGH_GUI

bool UDgreater ( float elem1, float elem2 )
{
   return elem1 > elem2;
}

void CxBoostFaceRecog::clear()
{
	m_maxfaceset_id = 0;

	if(m_casDetect) 
	{
		delete m_casDetect;
		m_casDetect =  NULL;
	}

	m_vTrainID.clear();

#ifdef	__SSE2__
	if( m_pFea )
		_mm_free(m_pFea);
#else
	if( m_pFea )
		delete [] m_pFea;
#endif
	
	//release vFaceset
	for(int i =0; i < m_vecFaceSet.size(); i++)
	{
		for(int j =0; j < m_vecFaceSet[i].vKeyFaceFeas.size(); j++)
		{
			delete []m_vecFaceSet[i].vKeyFaceFeas[j];
			m_vecFaceSet[i].vKeyFaceFeas[j] = NULL;
		}

		m_vecFaceSet[i].avgFaceFea.clear();
		
		for(int j =0; j < (int)m_vecFaceSet[i].vKeyFaceImgs.size(); j++)
		{
			cvReleaseImage(&m_vecFaceSet[i].vKeyFaceImgs[j]);
			m_vecFaceSet[i].vKeyFaceImgs[j] = NULL;
		}
	}

	m_vecFaceSet.clear();
	m_nFaces = 0;
}

CxBoostFaceRecog::~CxBoostFaceRecog(void)
{
	//free buff
	clear();
}

int CxBoostFaceRecog::load(int recognizerType, const char* path, const char* modprefix, int cutimg_size /*=128*/)
{
	m_recognizerType = recognizerType;
	m_defRound  =   0;

	if(m_recognizerType == RECOGNIZER_BOOST_GB240 || m_recognizerType == RECOGNIZER_BOOST_LBP59)
	{
		m_casDetect = new CxBoostDetect;
	}
	else if(m_recognizerType == RECOGNIZER_CAS_GLOH)
	{
		m_casDetect = new CxCasDetect;
	}

	m_defRound     = m_casDetect->load(path, modprefix, cutimg_size);
	m_defThreshold = m_casDetect->getDefThreshold();
	m_fea_dim      = m_casDetect->getFeatureDim();
	m_fea_type     = m_casDetect->getFeatureType();
	
	m_cutimg_size  = cutimg_size;
	
	return m_defRound;
}

void CxBoostFaceRecog::saveFaceModelXML(const char *sPathXML, vFaceSet *pvecFaceSet /*=NULL*/)
{
	if(pvecFaceSet == NULL)
		pvecFaceSet = &m_vecFaceSet;

	if(FACESET_DYNAMIC_STORE == 0)
	{
		char pImgPath[256];
		char pfeaPath[256];
		for(int i=0; i < (int)(*pvecFaceSet).size(); i++)
		{

			//creat a subfolder
			char sFullPath[256];
#ifdef _WIN32
			sprintf(sFullPath, "%s/%s", m_szImgDBPath, (*pvecFaceSet)[i].szFaceSetName.c_str());
			_mkdir(sFullPath);
#else
			sprintf(sFullPath, "mkdir %s/%s", m_szImgDBPath, (*pvecFaceSet)[i].szFaceSetName.c_str());
			system (sFullPath);
#endif

			//save feature and image files in the subfolder
			char pImgPath[255];
			for(int j=0; j < (int)(*pvecFaceSet)[i].vKeyFaceFeas.size(); j++)
			{
				//rename vKeyFaceNames
				sprintf(pImgPath, "%s/%d_%d.jpg", (*pvecFaceSet)[i].szFaceSetName.c_str(), (*pvecFaceSet)[i].nFaceSetID, j); //relative path
				(*pvecFaceSet)[i].vKeyFaceNames[j] = pImgPath;
				sprintf(pImgPath, "%s/%s", m_szImgDBPath, (*pvecFaceSet)[i].vKeyFaceNames[j].c_str()); // image full path to save
				sprintf(pfeaPath, "%s_%d.fea", pImgPath, m_fea_type);    // feature full path to save

				cvSaveImage(pImgPath,  (*pvecFaceSet)[i].vKeyFaceImgs[j]);

				FILE  *fp   = fopen(pfeaPath, "wb");
				float *pFea = (*pvecFaceSet)[i].vKeyFaceFeas[j];
				fwrite(pFea, sizeof(float), m_fea_dim, fp);
				fclose(fp);
			}
		}
	}

	//save faceset xml
	CFaceListXML		faceXML;
	faceXML.setImgDBPath(m_szImgDBPath);
	faceXML.save(sPathXML, (*pvecFaceSet), 0);
}

void CxBoostFaceRecog::getMergedFaceSet(vFaceSet& vvClusters, int weigthThreshold /*= 0*/)
{
	//1. clear vvClusters buffer
	for(int i =0; i < vvClusters.size(); i++)
	{
		for(int j =0; j < vvClusters[i].vKeyFaceFeas.size(); j++)
		{
			delete []vvClusters[i].vKeyFaceFeas[j];
			vvClusters[i].vKeyFaceFeas[j] = NULL;
		}

		vvClusters[i].avgFaceFea.clear();
		
		for(int i =0; i < (int)vvClusters[i].vKeyFaceImgs.size(); i++)
		{
			cvReleaseImage(&vvClusters[i].vKeyFaceImgs[i]);
			vvClusters[i].vKeyFaceImgs[i] = NULL;
		}
	}

	vvClusters.clear();


	//2. combine and merger m_vecFaceSet into
	for(int i =0; i < m_vecFaceSet.size(); i++)
	{
		int FaceSetIDA = m_vecFaceSet[i].nFaceSetID;

		int  idxJ  = -1;
		bool bFind = false;
		for(int j =0; j < vvClusters.size(); j++)
		{
			int FaceSetIDB = vvClusters[j].nFaceSetID;
			if( FaceSetIDA == FaceSetIDB )
			{
				bFind = true;
				idxJ  = j;
				break;
			}
		}

		if(bFind == false)
			vvClusters.push_back(m_vecFaceSet[i]);
		else
		{
			 vvClusters[idxJ].nFaceSetWeight += m_vecFaceSet[i].nFaceSetWeight;

			 //copy faces and features from cluster cB to cluster cA
			int nFacecA = m_vecFaceSet[i].vKeyFaceIDs.size();
			int nFacecB = vvClusters[idxJ].vKeyFaceIDs.size();
			int nFaceCount = nFacecA + nFacecB;
			for(int k =0; k < m_vecFaceSet[i].vKeyFaceIDs.size(); k++)
			{
				vvClusters[idxJ].vKeyFaceIDs.push_back(m_vecFaceSet[i].vKeyFaceIDs[k]);
				vvClusters[idxJ].vKeyFaceNames.push_back(m_vecFaceSet[i].vKeyFaceNames[k]);
				vvClusters[idxJ].vKeyFaceFeas.push_back(m_vecFaceSet[i].vKeyFaceFeas[k]);
				vvClusters[idxJ].vKeyFaceImgs.push_back(m_vecFaceSet[i].vKeyFaceImgs[k]);
			}
			for(int k=0; k < m_fea_dim; k++)
				vvClusters[idxJ].avgFaceFea[k] = (m_vecFaceSet[i].avgFaceFea[k] * nFacecA + vvClusters[idxJ].avgFaceFea[k] * nFacecB ) / nFaceCount;
		}
	}

	//remove clusters with small nFaceSetWeight < weigthThreshold
	for(int i = vvClusters.size()-1; i >=0 ; i--)
	{
		if(vvClusters[i].nFaceSetWeight < weigthThreshold)
		{
			vvClusters.erase(vvClusters.begin()+i);
		}
	}
}

int CxBoostFaceRecog::loadFaceModelXML(const char *sPathXML /*=NULL*/, bool bLoadExtFea /*=true*/)
{
	m_vTrainID.clear();
	m_nFaces = 0;

	if(sPathXML == NULL)
		return 0; //only Init vFaceSet and buff of m_matTrainFea and m_matTrainID
	
	//load face xml
	CFaceListXML		faceXML;
	faceXML.load(sPathXML, m_vecFaceSet);
	strcpy(m_szImgDBPath, faceXML.getImgDBPath().c_str());

	//total faces
	m_nFaces = 0;
	// modified by Ren, Haibing, 2014-10-20
	if(m_vecFaceSet.size() ==0)
		return 0;

	for(int i=0; i < (int)m_vecFaceSet.size(); i++)
		m_nFaces += (int)m_vecFaceSet[i].vKeyFaceIDs.size();

	m_vTrainID.resize(m_nFaces, -1);

 	if(bLoadExtFea == false)
		return m_nFaces;

	//load/extract feature of face exemplars
	int  nIdx = 0;
	int  nFaces;
	int  nFaceSetID;
	char pfeaPath[256];
	char pImgPath[256];

	for(int i=0; i<m_vecFaceSet.size(); i++)
	{
		nFaceSetID  = m_vecFaceSet[i].nFaceSetID;
		nFaces      = (int)m_vecFaceSet[i].vKeyFaceNames.size();
		m_vecFaceSet[i].vKeyFaceFeas.resize(nFaces, NULL);
		m_vecFaceSet[i].avgFaceFea.resize(m_fea_dim, 0);
		
		for(int j=0; j<nFaces; j++)
		{
			if(strlen(m_szImgDBPath) <=0)
				sprintf(pImgPath, "%s", m_vecFaceSet[i].vKeyFaceNames[j].c_str()); //full img path
			else
				sprintf(pImgPath, "%s/%s", m_szImgDBPath, m_vecFaceSet[i].vKeyFaceNames[j].c_str()); //full img path

			sprintf(pfeaPath, "%s_%d.fea", pImgPath, m_fea_type); // full fea path

			// load cut face image
			IplImage   *pCutFace = cvLoadImage(pImgPath, CV_LOAD_IMAGE_GRAYSCALE);
			// if the image is deleted, added by Ren, Haibing 2014-10-20
			if(pCutFace == NULL)
			{
				m_vecFaceSet[i].vKeyFaceFeas[j] = NULL;
				continue;
			}

			m_vTrainID[nIdx] = nFaceSetID; //faceSetID of each training face
			float *pFeaT = new float[m_fea_dim]; 
			m_vecFaceSet[i].vKeyFaceFeas[j] = pFeaT;
			//resize to aligned face size
			if(pCutFace->width != m_cutimg_size || pCutFace->height != m_cutimg_size)
			{
				IplImage   *pTmp = cvCreateImage(cvSize(m_cutimg_size, m_cutimg_size), IPL_DEPTH_8U, 1);
				cvResize(pCutFace, pTmp);
				cvReleaseImage(&pCutFace);
				pCutFace = pTmp;
			}
			m_vecFaceSet[i].vKeyFaceImgs.push_back(pCutFace);

			FILE *fp = fopen(pfeaPath, "rb");
			if(fp != NULL)
			{
				// read feature
				fread(pFeaT, sizeof(float), m_fea_dim, fp);
				fclose(fp);
			}
#ifdef USE_CVHIGH_GUI
			else
			{
				m_casDetect->extFeature(pCutFace, pFeaT);
				fp = fopen(pfeaPath, "wb");
				fwrite(pFeaT, sizeof(float), m_fea_dim, fp);
				fclose(fp);
			}
#endif			
			//accumulate feas;
			for(int k =0; k < m_fea_dim; k++)
				m_vecFaceSet[i].avgFaceFea[k] += pFeaT[k];

			nIdx ++;
			if(nIdx % 50 == 0)
				printf("loading %.2f%%\r", 100*(nIdx+0.0)/m_nFaces);

			//delete pFeaT; //Un-Modified by Nianzu, 2015-06-05
		}

		//average features
		for(int k =0; k < m_fea_dim; k++)
				m_vecFaceSet[i].avgFaceFea[k] /= nFaces;
	}

	m_maxfaceset_id = m_vecFaceSet[m_vecFaceSet.size()-1].nFaceSetID+1;

	printf("loading 100.00%%\n");
	return m_nFaces;
}

const char* CxBoostFaceRecog::getFaceName(int nFaceSetID)
{
	if(nFaceSetID >= 0)
	{
		int nFaceSetIdx = getFaceSetIdx(nFaceSetID);
		
		if(nFaceSetIdx >= 0)
			return m_vecFaceSet[nFaceSetIdx].szFaceSetName.c_str();
	}

	return "N/A";
}

float CxBoostFaceRecog::clusterSim(int idx1, int idx2)
{
	int nFaces1 = (int)m_vecFaceSet[idx1].vKeyFaceNames.size();
	int nFaces2 = (int)m_vecFaceSet[idx2].vKeyFaceNames.size();
	int nFaces  = nFaces1*nFaces2;

	float *pFea1, *pFea2;
	float  retProb;
	std::vector<float> vSim;
	vSim.reserve(nFaces);

	for(int i=0; i < nFaces1; i++)
	{
		pFea1 = m_vecFaceSet[idx1].vKeyFaceFeas[i];
		for(int j=0; j < nFaces2; j++)
		{
			pFea2 = m_vecFaceSet[idx2].vKeyFaceFeas[j];
			m_casDetect->predictDiff(pFea1, pFea2, &retProb);
			vSim.push_back(retProb);
		}
	}

	int KNeighbor = MIN(2, nFaces);
	partial_sort(vSim.begin(), vSim.begin()+KNeighbor, vSim.end(), UDgreater);
	
	float fClusterSim = 0.0f;
	for(int k =0; k < KNeighbor; k++)
		fClusterSim += vSim[k];
	
	fClusterSim /= KNeighbor;
	
	return fClusterSim;
}

float CxBoostFaceRecog::clusterAvgSim(int idx1, int idx2)
{
	
	float *pFea1 = &(m_vecFaceSet[idx1].avgFaceFea[0]);
	float *pFea2 = &(m_vecFaceSet[idx1].avgFaceFea[0]);
	float  retProb;
	m_casDetect->predictDiff(pFea1, pFea2, &retProb);

	return retProb;
}

float CxBoostFaceRecog::clusterSim(float *pFea, int idx)
{
	int nFaces = (int)m_vecFaceSet[idx].vKeyFaceNames.size();
	
	float *pFea2;
	float  retProb;
	std::vector<float> vSim;
	vSim.reserve(nFaces);
	
	for(int j=0; j < nFaces; j++)
	{
		pFea2 = m_vecFaceSet[idx].vKeyFaceFeas[j];
		m_casDetect->predictDiff(pFea, pFea2, &retProb);
		vSim.push_back(retProb);
	}

	int KNeighbor = MIN(2, sqrt(1.0f*nFaces));
	partial_sort(vSim.begin(), vSim.begin()+KNeighbor, vSim.end(), UDgreater);
	
	float fClusterSim = 0.0f;
	for(int k =0; k < KNeighbor; k++)
		fClusterSim += vSim[k];
	
	fClusterSim /= KNeighbor;
	
	return fClusterSim;
}

int CxBoostFaceRecog::predict(IplImage* pCutFace, float *prob /* =NULL */, bool bAutoCluster /*= false*/, int face_trackid /*= -1*/, int frameid /*= -1*/)
{
	if(m_nFaces < 1)
		return -1;

#ifdef	__SSE2__
	if( m_pFea == NULL )
		m_pFea = (float *)_mm_malloc(sizeof(float)*m_fea_dim,16);
#else
	if( m_pFea == NULL )
		m_pFea = new float[m_fea_dim];
#endif

	m_casDetect->extFeature(pCutFace, m_pFea);
	float *pFea = m_pFea;

	int   dense_threshold = 3;
	float threshold       = m_defThreshold; // 0.95f;//m_defThreshold;
	float threshold_high  = threshold+0.10f;//threshold+ (1.0f-threshold)*0.7f; //0.98f;//threshold+ (1.0f-threshold)*0.7f;
	float threshold_low   = threshold-0.10f;//threshold- threshold*0.1f; //0.6f;//threshold- threshold*0.1f;
	
	int nClusters =  m_vecFaceSet.size();
	std::vector<float> vClusterSim(nClusters, 0);
	std::vector<float> vSim;
	std::vector<int>   vMergeCluster;

	m_nFaces = 0;
	for(int i=0; i < (int)m_vecFaceSet.size(); i++)
		m_nFaces += (int)m_vecFaceSet[i].vKeyFaceIDs.size(); // modified by Ren, Haibing 2014-10-20
		//	m_nFaces += (int)m_vecFaceSet[i].vKeyFaceImgs.size();
	m_vTrainID.resize(m_nFaces, -1);

	int    idx = 0;
	int    nFaceSetID_MaxSim = -1, nFaceSetIdx_MaxSim = -1, nFaceSetIdx_MaxClusterSim = -1;
	float  retProb, fClusterMaxSim = -1;
	std::vector<float>	vProb;
	vProb.resize(m_nFaces, 0.0f);
	m_vTrainID.resize(m_nFaces, -1);
	
	bool bSameTrack = false;
	for(int i=0, idx=0; i < nClusters; i++)
	{
		int  nFaceSetWeight  = m_vecFaceSet[i].nFaceSetWeight;
		int  nFaceSetID      = m_vecFaceSet[i].nFaceSetID;
		int  nFaces          = (int)m_vecFaceSet[i].vKeyFaceNames.size(); // modified by Ren, Haibing 2014-10-20
		//int  nFaces          = (int)m_vecFaceSet[i].vKeyFaceImgs.size();

		bool bNonPredictFlag = (frameid >=0 && m_vecFaceSet[i].nCurFrameID == frameid); //has been recgonized face set of frameid
		bSameTrack = (m_vecFaceSet[i].nCurTrackID == face_trackid);

		vSim.resize(nFaces);
		for(int j=0; j<nFaces; j++)
		{
			if(bNonPredictFlag)
				retProb = 0;
			else
			{
				float *pFeaT = m_vecFaceSet[i].vKeyFaceFeas[j];//&(m_vecFaceSet[i].avgFaceFea[0]);
				if(pFeaT == NULL)  // modified by Ren, Haibing  2014-10-20
				{ // invalude face template
					retProb = 0.0;
				}
				else
				{
					int ret = m_casDetect->predictDiff(pFea, pFeaT, &retProb);
				}
			}
	
			vProb[idx] = (float)retProb;
			m_vTrainID[idx] = nFaceSetID;
			idx++;

			if(bAutoCluster)
				vSim[j] = retProb;
		}

		if(bAutoCluster ) 
		{	//dynamic clustering
			//KNN distance
			int KNeighbor = MIN(3, nFaces);//sqrt(1.0f*nFaces));
			partial_sort(vSim.begin(), vSim.begin()+KNeighbor, vSim.end(), UDgreater);
			
			float clusterSim = 0.0f;
			for(int k =0; k < KNeighbor; k++)
				clusterSim += vSim[k];
			
			clusterSim /= KNeighbor;
			vClusterSim[i] = clusterSim;

			if(clusterSim > fClusterMaxSim)
			{	//find max cluster similarity
				fClusterMaxSim = clusterSim;
				nFaceSetIdx_MaxClusterSim = i;
			}

			if(bSameTrack && clusterSim > threshold_low)
			{	//same face trackerID to merge if clusterSim > threshold_low, a lower threshold
				vMergeCluster.push_back(i);
			}
			else if(clusterSim > threshold && nFaceSetWeight > dense_threshold)
			{	//very similar to merge if nFaceSetWeight > dense_threshold
				vMergeCluster.push_back(i);
			}
		}
	}
	bSameTrack =false;
	std::vector<float>::iterator pProbIter;
	pProbIter = std::max_element(vProb.begin(), vProb.end()); // find the best match faces id with min dist value
	idx = (int)(pProbIter - vProb.begin());
	retProb = vProb[idx]; // max Prob value
	nFaceSetID_MaxSim = m_vTrainID[idx];
	
	if( prob != NULL ) *prob = retProb;
	
	if(bAutoCluster == false) 
		return nFaceSetID_MaxSim;

	/////////////////////////////////////////////////////////////////
	// dynamic auto clustering
	if(nFaceSetIdx_MaxClusterSim <0) //in case of exception case
		return nFaceSetID_MaxSim;

	nFaceSetID_MaxSim = m_vecFaceSet[nFaceSetIdx_MaxClusterSim].nFaceSetID;
	*prob = fClusterMaxSim;

	//update nFaceSetID_MaxSim by clsuter sim
	nFaceSetID_MaxSim = m_vecFaceSet[nFaceSetIdx_MaxClusterSim].nFaceSetID;

	if(frameid != -1 && m_oldframe_id != frameid)
	{ //update conflict facesetIDs with same m_oldframe_id at frameid changed point

		std::vector<CvPoint> vConflictCluster;
		for(int i=0; i < nClusters; i++)
		{
			if(m_vecFaceSet[i].nCurFrameID == m_oldframe_id)
				vConflictCluster.push_back(cvPoint(i, m_vecFaceSet[i].nFaceSetID));
		}

		if(vConflictCluster.size() > 1)
		{    //existing conflict clusters
			for(int i=0; i < vConflictCluster.size(); i++)
			{
				idx = vConflictCluster[i].x;
				for(int j =0; j < vConflictCluster.size(); j++)
				{
					std::vector<int>::iterator iFindConflict = find(m_vecFaceSet[idx].vConflictFaceSetIDs.begin(), m_vecFaceSet[idx].vConflictFaceSetIDs.end(), vConflictCluster[j].y);
					if(iFindConflict == m_vecFaceSet[idx].vConflictFaceSetIDs.end())
					{
						m_vecFaceSet[idx].vConflictFaceSetIDs.push_back(vConflictCluster[j].y);

						//limit the m_vecFaceSet[idx].vConflictFaceSetIDs size
						if(m_vecFaceSet[idx].vConflictFaceSetIDs.size() == FACESETCONFLICT_MAXSIZE)
							m_vecFaceSet[idx].vConflictFaceSetIDs.erase(m_vecFaceSet[idx].vConflictFaceSetIDs.begin(), m_vecFaceSet[idx].vConflictFaceSetIDs.begin()+FACESETCONFLICT_MAXSIZE/2);
					}
				}
			}
		}
	}

	m_oldframe_id = frameid;

	int nMergeClusters = (int)vMergeCluster.size();
	
	if(nMergeClusters == 0)
	{
		if(fClusterMaxSim < threshold ) 
		{
			if( fClusterMaxSim < threshold_low )
			{
				//0: insert a new faceset and add it as a new example if not similar to any cluster
				char sFaceName[32];
				sprintf(sFaceName, "unknown%d", m_maxfaceset_id);
				
				int nFaceSetIdx = insertEmptyFaceSet(sFaceName);
				tryInsertFace(pCutFace, nFaceSetIdx);
				m_vecFaceSet[nFaceSetIdx].nCurFrameID = frameid;
				m_vecFaceSet[nFaceSetIdx].nCurTrackID = face_trackid;
				nFaceSetID_MaxSim = m_vecFaceSet[nFaceSetIdx].nFaceSetID;

#ifdef _DEBUG
				printf("###Create a new cluster %s with prob sim %f, cluster sim %f, clusterFaceSetID %d, trackID %d end\n",
					sFaceName, retProb, fClusterMaxSim,  nFaceSetID_MaxSim,  face_trackid);
#endif
			}
		}
		else if(fClusterMaxSim < threshold_high ) 
		{	//insert to the max similar cluster if threshold < retProb < threshold_high
			if(frameid == -1 || m_vecFaceSet[nFaceSetIdx_MaxClusterSim].nCurFrameID != frameid)
			{
				tryInsertFace(pCutFace, nFaceSetIdx_MaxClusterSim, true);
				m_vecFaceSet[nFaceSetIdx_MaxClusterSim].nCurFrameID = frameid;
				m_vecFaceSet[nFaceSetIdx_MaxClusterSim].nCurTrackID = face_trackid;
#ifdef _DEBUG
				printf("###insert into existing cluster %d with prob sim %f, cluster sim %f trackID %d\n", 
					m_vecFaceSet[nFaceSetIdx_MaxClusterSim].nFaceSetID, retProb, fClusterMaxSim, face_trackid);
#endif
			}
		}
		else 
		{	//insert to the max similar cluster without image insertion if retProb >= threshold_high
			if(frameid == -1 || m_vecFaceSet[nFaceSetIdx_MaxClusterSim].nCurFrameID != frameid)
			{
				if(m_vecFaceSet[nFaceSetIdx_MaxClusterSim].nFaceSetWeight >= INT_MAX)
					m_vecFaceSet[nFaceSetIdx_MaxClusterSim].nFaceSetWeight /= 2;

				m_vecFaceSet[nFaceSetIdx_MaxClusterSim].nFaceSetWeight++;
				m_vecFaceSet[nFaceSetIdx_MaxClusterSim].nCurFrameID = frameid;
				m_vecFaceSet[nFaceSetIdx_MaxClusterSim].nCurTrackID = face_trackid;

#ifdef _DEBUG
				//printf("###insert into existing cluster %d with prob sim %f, cluster sim %f trackID %d %d\n", 
				//	m_vecFaceSet[nFaceSetIdx_MaxClusterSim].nFaceSetID, retProb, fClusterMaxSim, face_trackid, m_vecFaceSet[nFaceSetIdx_MaxClusterSim].nFaceSetWeight);
#endif
			}
		}
	}
	else if(fClusterMaxSim > threshold)
	{
		if(fClusterMaxSim < threshold_high ) 
		{
			if(frameid == -1 || m_vecFaceSet[nFaceSetIdx_MaxClusterSim].nCurFrameID != frameid)
			{
				tryInsertFace(pCutFace, nFaceSetIdx_MaxClusterSim, true);
				m_vecFaceSet[nFaceSetIdx_MaxClusterSim].nCurFrameID = frameid;
				m_vecFaceSet[nFaceSetIdx_MaxClusterSim].nCurTrackID = face_trackid;
#ifdef _DEBUG
				printf("###Merge insert into existing cluster %d with prob sim %f, cluster sim %f trackID %d %d\n", 
					m_vecFaceSet[nFaceSetIdx_MaxClusterSim].nFaceSetID, retProb, fClusterMaxSim, face_trackid, m_vecFaceSet[nFaceSetIdx_MaxClusterSim].nFaceSetWeight);
#endif
			}
		}
		else 
		{
			if(frameid == -1 || m_vecFaceSet[nFaceSetIdx_MaxClusterSim].nCurFrameID != frameid)
			{
				if(m_vecFaceSet[nFaceSetIdx_MaxClusterSim].nFaceSetWeight >= INT_MAX)
					m_vecFaceSet[nFaceSetIdx_MaxClusterSim].nFaceSetWeight /= 2;
				m_vecFaceSet[nFaceSetIdx_MaxClusterSim].nFaceSetWeight++;
				m_vecFaceSet[nFaceSetIdx_MaxClusterSim].nCurFrameID = frameid;
				m_vecFaceSet[nFaceSetIdx_MaxClusterSim].nCurTrackID = face_trackid;
			}

			//printf("###Merge insert into existing cluster %d with prob sim %f, cluster sim %f trackID %d\n",
			//	m_vecFaceSet[nFaceSetIdx_MaxClusterSim].nFaceSetID,  retProb, fClusterMaxSim, face_trackid);
		}

		//merge clusters
		//1: merge clusters, insert to the max simiar cluster and add it as a new example
		int   nCluster0      = vMergeCluster[0];
		int   nCluster0Weight = m_vecFaceSet[nCluster0].nFaceSetWeight;
	
		int   pos0   = m_vecFaceSet[nCluster0].szFaceSetName.find("unknown");
		nFaceSetID_MaxSim   = m_vecFaceSet[nCluster0].nFaceSetID;
		
		 int delK = 0;
		 for(int j =1; j < nMergeClusters; j++)
		 {
			int nClusterJ = vMergeCluster[j]-delK;
			int nClusterJWeight = m_vecFaceSet[nClusterJ].nFaceSetWeight;
		
			bSameTrack = m_vecFaceSet[nCluster0].nCurTrackID == m_vecFaceSet[nClusterJ].nCurTrackID;
			if(!bSameTrack && (nCluster0Weight < 3 || nClusterJWeight < 3))
				continue;

			if(!bSameTrack && (m_vecFaceSet[nCluster0].vKeyFaceNames.size() < 2 || m_vecFaceSet[nClusterJ].vKeyFaceNames.size() < 2) )
				continue;
			
			//continue if the two faces are in same frame
			if(frameid != -1 && m_vecFaceSet[nCluster0].nCurFrameID == m_vecFaceSet[nClusterJ].nCurFrameID) 
				continue;

			//continue if the two clusters are conflicted
			std::vector<int>::iterator iFindConflict = find(m_vecFaceSet[nCluster0].vConflictFaceSetIDs.begin(), m_vecFaceSet[nCluster0].vConflictFaceSetIDs.end(), m_vecFaceSet[nClusterJ].nFaceSetID);
			if(iFindConflict != m_vecFaceSet[nCluster0].vConflictFaceSetIDs.end())
				continue;

			float fClusterSim  = 0;
			//if(!bSameTrack)
			//{
			//	fClusterSim = clusterSim(nCluster0, nClusterJ);
			//	if(fClusterSim < threshold_low) continue;
			//}
			//else
			{
				if(vClusterSim[nCluster0] < threshold_high) continue;
				if(vClusterSim[nClusterJ] < threshold_high) continue;
			}
			
#ifdef _DEBUG			
			printf("###merge two cluster_i %d %s %f %d to cluster_j %d %s %f %d with prob sim %f, cluster sim %f %f ##%f\n", 
				m_vecFaceSet[nCluster0].nFaceSetID, m_vecFaceSet[nCluster0].szFaceSetName.c_str(), vClusterSim[nCluster0], nCluster0Weight,
				m_vecFaceSet[nClusterJ].nFaceSetID, m_vecFaceSet[nClusterJ].szFaceSetName.c_str(), vClusterSim[nClusterJ], nClusterJWeight, retProb, 
				vClusterSim[nCluster0], vClusterSim[nClusterJ], fClusterSim);
#endif

			m_vecFaceSet[nCluster0].nCurFrameID = frameid;
			m_vecFaceSet[nCluster0].nCurTrackID = face_trackid;

			delK++; 
			
			int nWeight0 = m_vecFaceSet[nCluster0].nFaceSetWeight;
			int nWeightJ = m_vecFaceSet[nClusterJ].nFaceSetWeight;

			//swap nFaceSetID and szFaceSetName by nFaceSetWeight and szFaceSetName
			if(nWeight0 > nWeightJ)
			{
				m_vecFaceSet[nClusterJ].nFaceSetID    = m_vecFaceSet[nCluster0].nFaceSetID;
			}
			else
				m_vecFaceSet[nCluster0].nFaceSetID    = m_vecFaceSet[nClusterJ].nFaceSetID;

			int pos0 =  m_vecFaceSet[nCluster0].szFaceSetName.find("unknown");
			int pos1 =  m_vecFaceSet[nClusterJ].szFaceSetName.find("unknown");
			if(pos0 >= 0 && pos1 < 0)
			{
				m_vecFaceSet[nCluster0].szFaceSetName = m_vecFaceSet[nClusterJ].szFaceSetName;
			}
			else if(nWeight0 > nWeightJ)
				m_vecFaceSet[nClusterJ].szFaceSetName = m_vecFaceSet[nCluster0].szFaceSetName;
			else
				m_vecFaceSet[nCluster0].szFaceSetName = m_vecFaceSet[nClusterJ].szFaceSetName;

			//package old codes into a fucntion to simplify
			//mergeClusters(m_vecFaceSet, nCluster0, nClusterJ);

/*			//old codes
			int pos1 =  m_vecFaceSet[nClusterJ].szFaceSetName.find("unknown");
			if(pos0 >= 0 && pos1 < 0)
			{	
				// switch and merge to cluster with valid face name if clusterJ has valid cluster name
				//printf("invert cluster %d %d\n", nCluster0, nClusterJ);
				pos0=pos1;
				m_vecFaceSet[nCluster0].szFaceSetName = m_vecFaceSet[nClusterJ].szFaceSetName;
				
				//copy images from cluster0 to clusterJ
				for(int k =0; k < m_vecFaceSet[nCluster0].vKeyFaceNames.size(); k++)
				{
					char pImgPath[512];
					if(strlen(m_szImgDBPath) <=0)
						sprintf(pImgPath, "%s", m_vecFaceSet[nCluster0].vKeyFaceNames[k].c_str()); //full img path
					else
						sprintf(pImgPath, "%s/%s", m_szImgDBPath, m_vecFaceSet[nCluster0].vKeyFaceNames[k].c_str()); //full img path
				
					IplImage* pFace = cvLoadImage(pImgPath, CV_LOAD_IMAGE_GRAYSCALE);
					if(pFace == NULL) continue;

					tryInsertFace(pFace, nClusterJ, true);

					cvReleaseImage(&pFace);

					delete []m_vecFaceSet[nCluster0].vKeyFaceFeas[k];
					m_vecFaceSet[nCluster0].vKeyFaceFeas[k] = NULL;
				}

				//empty cluster0
				m_vecFaceSet[nCluster0].nFaceSetWeight -= m_vecFaceSet[nCluster0].vKeyFaceNames.size();
				m_vecFaceSet[nCluster0].vKeyFaceIDs.clear();
				m_vecFaceSet[nCluster0].vKeyFaceNames.clear();
				m_vecFaceSet[nCluster0].vKeyFaceFeas.clear();
												
				for(int k =0; k < m_vecFaceSet[nClusterJ].vKeyFaceNames.size(); k++)
				{
					delete [] m_vecFaceSet[nClusterJ].vKeyFaceFeas[k];
					m_vecFaceSet[nClusterJ].vKeyFaceFeas[k] = NULL;
				}
			}
		
			//copy clusterJ to cluster0
			m_vecFaceSet[nCluster0].nFaceSetWeight += m_vecFaceSet[nClusterJ].nFaceSetWeight;

			for(int k =0; k < m_vecFaceSet[nClusterJ].vKeyFaceNames.size(); k++)
			{
				char pImgPath[512];
				if(strlen(m_szImgDBPath) <=0)
					sprintf(pImgPath, "%s", m_vecFaceSet[nClusterJ].vKeyFaceNames[k].c_str()); //full img path
				else
					sprintf(pImgPath, "%s/%s", m_szImgDBPath, m_vecFaceSet[nClusterJ].vKeyFaceNames[k].c_str()); //full img path
		
				IplImage* pFace = cvLoadImage(pImgPath, CV_LOAD_IMAGE_GRAYSCALE);
				if(pFace == NULL) continue;

				tryInsertFace(pFace, nCluster0, true);

				cvReleaseImage(&pFace);

				delete []m_vecFaceSet[nClusterJ].vKeyFaceFeas[k];
				m_vecFaceSet[nClusterJ].vKeyFaceFeas[k] = NULL;

				DeleteFile(pImgPath);

				char pFeaPath[512];
				sprintf(pFeaPath, "%s_%d.fea", pImgPath, m_fea_type); 
				DeleteFile(pFeaPath);
				//printf("###delete file %s in cluster", pImgPath);
			}

			char pFolderPath[512];
			if(strlen(m_szImgDBPath) <=0)
				sprintf(pFolderPath, "%s", m_vecFaceSet[nClusterJ].szFaceSetName.c_str()); //full folder path
			else
				sprintf(pFolderPath, "%s/%s", m_szImgDBPath, m_vecFaceSet[nClusterJ].szFaceSetName.c_str()); //full folder path

			int sta = _rmdir(pFolderPath);
			//printf("###delete folder %s \n", pFolderPath);

			m_vecFaceSet.erase(m_vecFaceSet.begin()+nClusterJ);

			//revise vote buff
			bool bfind = false;
			int  idk = m_votebuff_idx;
			for(int k =0; k < NVOTEBUFF; k++)
			{	
				int cur_idx = (m_votebuff_idx-k +NVOTEBUFF) % NVOTEBUFF;
				if(face_trackid == m_votebuff_face_id[cur_idx])
				{
					bfind = true;
					idk   = cur_idx; 

					for(int a =0; a < NVOTEPERSON; a++)
					{
						m_votebuff_face_label[idk][a] = -1; // face_label = invalid
						m_votebuff_face_label_vote[idk][a] = 0;  // vote_num   = 0
					}
					//set item
					m_votebuff_face_id[idk] = face_trackid;

					m_votebuff_face_label[idk][0] = nFaceSetID_MaxSim;  // face_label = label
					m_votebuff_face_label_vote[idk][0] = 1; // vote_num   = 1
					break;
				}
			}
*/
		 }
	}
	
	delete[] m_pFea; //Modified by Nianzu, 2015-06-05
	return nFaceSetID_MaxSim;
}	

int CxBoostFaceRecog::predict(float* pFea, float *prob /* =NULL */)
{
	if(m_nFaces < 1)
		return -1;
	
	int    idx = 0;
	float  retProb;
	std::vector<float>	vProb;
	vProb.resize(m_nFaces, 0.0f);
	m_vTrainID.resize(m_nFaces, -1);

	for(int i=0, idx=0; i < m_vecFaceSet.size(); i++)
	{
		int nFaceSetID  = m_vecFaceSet[i].nFaceSetID;
		int nFaces      = (int)m_vecFaceSet[i].vKeyFaceNames.size();
		
		float maxSim = 0;
		for(int j=0; j<nFaces; j++)
		{
			float *pFeaT = m_vecFaceSet[i].vKeyFaceFeas[j];
			int ret = m_casDetect->predictDiff(pFea, pFeaT, &retProb);
	
			vProb[idx] = (float)retProb;
			m_vTrainID[idx] = nFaceSetID;
			idx++;
		}
	}

	std::vector<float>::iterator pProbIter;
	pProbIter = std::max_element(vProb.begin(), vProb.end()); // find the best match faces id with min dist value
	idx = (int)(pProbIter - vProb.begin());
	retProb = vProb[idx]; // max Prob value

	if( prob != NULL ) *prob = retProb;

	int nFaceSetID = m_vTrainID[idx];
	
	return nFaceSetID;
}

int CxBoostFaceRecog::voteLabel(int face_trackid, int label)
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
			 
		//reset item array
		for(int a =0; a < NVOTEPERSON; a++)
		{
			m_votebuff_face_label[idk][a] = -1; // face_label = invalid
			m_votebuff_face_label_vote[idk][a] = 0;  // vote_num   = 0
		}
		
		//set item
		m_votebuff_face_id[idk] = face_trackid;
		m_votebuff_face_label[idk][0] = label;  // face_label = label
		m_votebuff_face_label_vote[idk][0] = 1; // vote_num   = 1
	}
	else if(label >= 0) //vote face only if label >= 0. If label < 0, it return old result
	{
		int  a = 0;
		bool bfind_label = false;
		for( a = 0; a < NVOTEPERSON; a++)
		{
			if(m_votebuff_face_label[idk][a] == label)
			{
				bfind_label = true;
				break;
			}
			else if(m_votebuff_face_label[idk][a] < 0)
				break;
		}

		if(bfind_label == true) //finded
			m_votebuff_face_label_vote[idk][a] = MIN(smooth_len, m_votebuff_face_label_vote[idk][a]+ 1);
		else
		{
			if(a >= NVOTEPERSON) //not find it in the array and find the min_vote to overwrite
			{
				std::vector<int>::iterator pVoteIter;
				pVoteIter = std::max_element(m_votebuff_face_label_vote[idk].begin(), m_votebuff_face_label_vote[idk].end()); // find the best match faces id with min dist value
				a = (int)(pVoteIter - m_votebuff_face_label_vote[idk].begin());
			}

			m_votebuff_face_label[idk][a] = label;
			m_votebuff_face_label_vote[idk][a] = MIN(smooth_len, m_votebuff_face_label_vote[idk][a] + 1);
		}
	}
	else if(label <= -1 && m_fea_type == FEA_CSLBP_8U) //it is not person in the database to punish the voting
	{
		for(int a = 0; a < NVOTEPERSON; a++)
			m_votebuff_face_label_vote[idk][a] = MAX(0, m_votebuff_face_label_vote[idk][a] -1);
	}

	std::vector<int>::iterator pVoteIter;
	pVoteIter = std::max_element(m_votebuff_face_label_vote[idk].begin(), m_votebuff_face_label_vote[idk].end()); // find the best match faces id with min dist value
	int vote_out = (int)(pVoteIter - m_votebuff_face_label_vote[idk].begin());
	label_out = m_votebuff_face_label[idk][vote_out];
	int MaxVote = m_votebuff_face_label_vote[idk][vote_out];

	if(MaxVote < 3)
		label_out = -1;  //not decided

	return label_out;
}

///////////////////////////////////////////////////////////////////////////////
// online interactive collecting face exemplars
bool CxBoostFaceRecog::isSimilarToFaceSet(IplImage* pCutFace, int nFaceSetIdx)
{
	bool bSimilar = false;

	// compare by facial SIFT/gabor features
	float face_threshold = (float)getDefThreshold(); //0.52 for gabor120, 0.56 for sift128
	float prob = 0;

	int   nFaceSetID = m_vecFaceSet[nFaceSetIdx].nFaceSetID;
	int   retFaceSetID    = predict(pCutFace, &prob);

	if( prob < face_threshold)
	{
		bSimilar = false;
		return bSimilar;
	}
	else if(retFaceSetID != nFaceSetID)
	{
		bSimilar = false;
		return bSimilar;
	}
	else
	{
		bSimilar = true;
		return bSimilar;
	}

	return bSimilar; // ret == 0, means different
}

int CxBoostFaceRecog::getFaceSetIdx(int nFaceSetID)
{
	int nFaceSetIdx = -1;
	bool bFind = false;
	for(int i=0; i< m_vecFaceSet.size(); i++ )
	{
		if( m_vecFaceSet[i].nFaceSetID == nFaceSetID)
		{
			bFind = true;
			nFaceSetIdx = i;
			return nFaceSetIdx;
		}
	}

	return nFaceSetIdx;
}

int CxBoostFaceRecog::getFaceSetID(int nFaceSetIdx)
{
	int nFaceSetID = -1;
	int nFaceSet = (int)m_vecFaceSet.size();
	if(nFaceSetIdx < 0 || nFaceSetIdx >= nFaceSet) return nFaceSetID;

	nFaceSetID = m_vecFaceSet[nFaceSetIdx].nFaceSetID;
	return nFaceSetID;
}

std::vector<std::string>* CxBoostFaceRecog::getKeyFacePaths(int faceSetIdx)
{
	int nFaceSet = (int)m_vecFaceSet.size();
	if(faceSetIdx < 0 || faceSetIdx >= nFaceSet) return NULL;

	return &(m_vecFaceSet[faceSetIdx].vKeyFaceNames);
}

int CxBoostFaceRecog::insertEmptyFaceSet(char *faceName, bool createFolder /*=true*/, int nFaceSetID /*=-1*/)
{
	// find to see whether it is an existed name in m_vecFaceSet
	int  nFaceSetIdx = -1;
	bool bFind = false;
	for(int i=0; i< m_vecFaceSet.size(); i++ )
	{
		if( strcmp(m_vecFaceSet[i].szFaceSetName.c_str(), faceName) == 0)
		{
			bFind = true;
			nFaceSetIdx = i;
			return nFaceSetIdx;
		}
	}

	//insert the new faceset in m_vecFaceSet
	nFaceSetIdx = (int)m_vecFaceSet.size();

	//remove no_frequently used faceset if cluster number = FACESET_MAXSIZE
	if(nFaceSetIdx == FACESET_MAXSIZE)
	{
		int idx = -1;
		int curMinWeight = INT_MAX;
	
		for(int i = 0; i < nFaceSetIdx; i++)
		{
			if(m_vecFaceSet[i].nFaceSetWeight < curMinWeight)
			{ 
				idx = i;
				curMinWeight = m_vecFaceSet[i].nFaceSetWeight;
			}
		}
		if(idx != -1) removeFaceSet(idx);

		nFaceSetIdx = (int)m_vecFaceSet.size();
	}

	if(m_maxfaceset_id == INT_MAX) m_maxfaceset_id = 0;         //esize if overflow
	if(nFaceSetID < 0) nFaceSetID = (int)m_maxfaceset_id++; //m_vecFaceSet.size();

	FaceSet curFaceSet;
	curFaceSet.nFaceSetID     = nFaceSetID;// ID should be the input order
	curFaceSet.szFaceSetName  = faceName;
	curFaceSet.avgFaceFea.resize(m_fea_dim, 0);
	m_vecFaceSet.push_back(curFaceSet);

	if(createFolder && FACESET_DYNAMIC_STORE)
	{
		char sFullPath[256];

#ifdef _WIN32
		sprintf(sFullPath, "%s/%s", m_szImgDBPath, faceName);
		_mkdir(sFullPath);
#else
		sprintf(sFullPath, "mkdir %s/%s", m_szImgDBPath, faceName);
		system (sFullPath);
#endif

#ifdef _DEBUG
		printf("create new folder %s\n", sFullPath);
#endif
	}

	return nFaceSetIdx;
}

bool CxBoostFaceRecog::tryInsertFace(IplImage* pCutFace, int nFaceSetIdx, bool bForceInsert /*= false*/)
{  
	bool bSimilar = false;

	if(bForceInsert == false)
		bSimilar = isSimilarToFaceSet(pCutFace, nFaceSetIdx);

	//insert face into nFaceSetIdx
	if(bSimilar == false)
	{
		//remove no_representative face if face set size = FACESETITEM_MAXSIZE
		int nFaceCount = MAX(m_vecFaceSet[nFaceSetIdx].vKeyFaceNames.size(), m_vecFaceSet[nFaceSetIdx].vKeyFaceIDs.size());
		int faceIdx    = -1;
		float minProb  = FLT_MAX, prob;

		if(nFaceCount == FACESETITEM_MAXSIZE)
		{
			for(int i =0; i < nFaceCount; i++)
			{
				isSimilarFaces(m_vecFaceSet[nFaceSetIdx].vKeyFaceFeas[i], &(m_vecFaceSet[nFaceSetIdx].avgFaceFea[0]), &prob);
				if(prob < minProb)
				{
					minProb = prob;
					faceIdx = i;
				}
			}
			if(faceIdx != -1) removeFace(nFaceSetIdx, faceIdx);
		}


		//save face exemplar
		int  nFaceSetID;
		int  keyFaceID;
		char faceName[32]  = "";
		char pfeaPath[256] = "";
		char pImgPath[256] = "";

		strcpy(faceName, m_vecFaceSet[nFaceSetIdx].szFaceSetName.c_str());

		//insert into existed faceset
		nFaceSetID = m_vecFaceSet[nFaceSetIdx].nFaceSetID;
		m_vecFaceSet[nFaceSetIdx].nFaceSetWeight++;
		keyFaceID  = (int)m_vecFaceSet[nFaceSetIdx].vKeyFaceNames.size();

		//expand matrix
		float *pFea = new float[m_fea_dim];
		m_casDetect->extFeature(pCutFace, pFea);
		m_vecFaceSet[nFaceSetIdx].vKeyFaceFeas.push_back(pFea);
		m_vecFaceSet[nFaceSetIdx].vKeyFaceIDs.push_back(-1);

		sprintf(pImgPath, "%s/%d_%d.jpg", faceName, nFaceSetID, keyFaceID); //relative path
		m_vecFaceSet[nFaceSetIdx].vKeyFaceNames.push_back(pImgPath);      //add key face img

		IplImage *pImg = cvCloneImage(pCutFace);
		m_vecFaceSet[nFaceSetIdx].vKeyFaceImgs.push_back(pImg);

		sprintf(pImgPath, "%s/%s/%d_%d.jpg", m_szImgDBPath, faceName, nFaceSetID, keyFaceID); // image full path to save
		sprintf(pfeaPath, "%s_%d.fea", pImgPath, m_fea_type);    // feature full path to save

		if(FACESET_DYNAMIC_STORE)
		{
			cvSaveImage(pImgPath, pCutFace);

			FILE *fp = fopen(pfeaPath, "wb");
			fwrite(pFea, sizeof(float), m_fea_dim, fp);
			fclose(fp);
		}

		//update average features
		nFaceCount = m_vecFaceSet[nFaceSetIdx].vKeyFaceIDs.size();
		for(int i=0; i < m_fea_dim; i++)
			m_vecFaceSet[nFaceSetIdx].avgFaceFea[i] = (m_vecFaceSet[nFaceSetIdx].avgFaceFea[i] * (nFaceCount-1) + pFea[i]) / nFaceCount;

#ifdef _DEBUG
		printf("insert face %s\n", pImgPath);
#endif

		//delete[] pFea;	//Modified by Nianzu, 2015-06-05
	}

	return bSimilar;
}

int  CxBoostFaceRecog::removeFaceSet(int nFaceSetIdx)
{
	int nFaceSet = (int)m_vecFaceSet.size()-1;
	if(nFaceSetIdx < 0 || nFaceSetIdx > nFaceSet) return -1;

	//remove fea buffer
	int nFaces = (int)m_vecFaceSet[nFaceSetIdx].vKeyFaceFeas.size();
	for(int i =0; i< nFaces; i++)
	{
		float *pFea =m_vecFaceSet[nFaceSetIdx].vKeyFaceFeas[i];
		delete []pFea;
		pFea = NULL;
	}

	//remove img buffer
	int nImgs = (int)m_vecFaceSet[nFaceSetIdx].vKeyFaceImgs.size();
	for(int i =0; i< nImgs; i++)
	{
		IplImage *pImg = m_vecFaceSet[nFaceSetIdx].vKeyFaceImgs[i]; 
		cvReleaseImage(&pImg);
		pImg = NULL;
	}

	m_vecFaceSet.erase(m_vecFaceSet.begin()+nFaceSetIdx);

	return 1;
}

int  CxBoostFaceRecog::removeFace(int nFaceSetIdx, int faceIdx)
{
	int nFaces = (int)m_vecFaceSet[nFaceSetIdx].vKeyFaceNames.size()-1;
	if(faceIdx < 0 || faceIdx > nFaces) return -1;

	//remove fea buffer
	float *pFea =m_vecFaceSet[nFaceSetIdx].vKeyFaceFeas[faceIdx];
	delete []pFea;
	pFea = NULL;

	//remove img buffer
	IplImage *pImg = m_vecFaceSet[nFaceSetIdx].vKeyFaceImgs[faceIdx]; 
	cvReleaseImage(&pImg);
	pImg = NULL;

	m_vecFaceSet[nFaceSetIdx].vKeyFaceIDs.erase(m_vecFaceSet[nFaceSetIdx].vKeyFaceIDs.begin()+faceIdx);
	m_vecFaceSet[nFaceSetIdx].vKeyFaceNames.erase(m_vecFaceSet[nFaceSetIdx].vKeyFaceNames.begin()+faceIdx);
	m_vecFaceSet[nFaceSetIdx].vKeyFaceFeas.erase(m_vecFaceSet[nFaceSetIdx].vKeyFaceFeas.begin()+faceIdx);
	m_vecFaceSet[nFaceSetIdx].vKeyFaceImgs.erase(m_vecFaceSet[nFaceSetIdx].vKeyFaceImgs.begin()+faceIdx);

	return 1;
}

void CxBoostFaceRecog::extFeature(IplImage* pCutFace, float* pFea)
{
	m_casDetect->extFeature(pCutFace, pFea);
}

bool CxBoostFaceRecog::isSimilarFaces(float* pFea1, float* pFea2, float *pProb /*=NULL*/)
{
	bool   ret = false;
	float retProb;

	m_casDetect->predictDiff(pFea1, pFea2, &retProb);

	if(pProb) *pProb = (float)retProb;

	if (retProb > m_defThreshold)  ret = true;

	return ret;
}

/////////////////////////////////////////////////////////////////////
//face clustering

// added by Patricia for Photo Man exchange training feature through data base not through xml file.
// main differences from the function "predict" are: 
// return flag: 
//   -1: error;
//    0: insert a new faceset and add it as a new example
//    1: merge clusters, insert to an existed faceSet and add it as a new example
//    2: insert to an existed faceSet without a new example
// return faceSetID and pFea pointer

int CxBoostFaceRecog::forwardCluster(float* pFea, int faceID, char *sCutFaceImg, vFaceSet& vvClusters, vFaceSet& vvRepClusters, float fThreshold /*=-1*/)
{
	if(vvClusters.size() < vvRepClusters.size())
	{
		FaceSet curFaceSet;
		vvClusters.resize(vvRepClusters.size(), curFaceSet);
	}

	int    flag = -1;
	
	float  *pFea1;
	float  *pFea0 = new float[m_fea_dim];

	if(fThreshold < 0)
		  fThreshold = getDefThreshold();

	//get pFea0 from sCutFaceImg or pFea
	if(pFea == NULL)
	{
		// extract face feature pFea of sCutFaceImg
		if( sCutFaceImg != NULL)
		{
			char   pImgPath[256];
			char   pfeaPath[256];
			sprintf(pImgPath, "%s", sCutFaceImg); //full img path
			sprintf(pfeaPath, "%s_%d.fea", pImgPath, m_fea_type); // full fea path

			FILE *fp = fopen(pfeaPath, "rb");
			if(fp == NULL)
			{
				// load cut face image
				IplImage   *pCutFace = cvLoadImage(pImgPath, CV_LOAD_IMAGE_GRAYSCALE);

				//resize to aligned face size
				if(pCutFace->width != m_cutimg_size || pCutFace->height != m_cutimg_size)
				{
					IplImage   *pTmp = cvCreateImage(cvSize(m_cutimg_size, m_cutimg_size), IPL_DEPTH_8U, 1);
					cvResize(pCutFace, pTmp);
					cvReleaseImage(&pCutFace);
					pCutFace = pTmp;
				}

				m_casDetect->extFeature(pCutFace, pFea0);
				cvReleaseImage(&pCutFace);

				fp = fopen(pfeaPath, "wb");
				fwrite(pFea0, sizeof(float), m_fea_dim, fp);
				fclose(fp);
			}
			else
			{
				// read feature
				fread(pFea0, sizeof(float), m_fea_dim, fp);
				fclose(fp);
			}
		}
		else
		{
			//Modified by Nianzu, 2015-06-05
			delete[] pFea0;
			return -1;
		}
	}
	else memcpy(pFea0, pFea, m_fea_dim*sizeof(float));

	//insert the face into the similar clusters
	std::vector <int> vMergeCluster;
	bool  bSim = false;
	float fProb;
	int nRepFaceSet = (int)vvRepClusters.size();
	for(int i =0; i < nRepFaceSet; i++)
	{
		bSim = false;
		int nRepFaces = (int)vvRepClusters[i].vKeyFaceNames.size();
		for(int j =0; j < nRepFaces; j++)
		{
			//get feature of vvRepClusters[i].vKeyFaceNames[j]
			pFea1 = vvRepClusters[i].vKeyFaceFeas[j];
			
			//similar or not
			bSim = isSimilarFaces(pFea0, pFea1, &fProb);

			if(fProb > fThreshold) bSim = true;
			else bSim = false;

			if(bSim == true) break;
		}

		if(bSim == true)
			vMergeCluster.push_back(i);
	}

	int nMergeClusters = (int)vMergeCluster.size();
	if(nMergeClusters == 0)
	{
		flag = 0; //0: insert a new faceset and add it as a new example
		char sFaceSetName[256];
		FaceSet curFaceSet;
		sprintf(sFaceSetName, "C%d", vvClusters.size());
		curFaceSet.nFaceSetID = (int)vvClusters.size();
		curFaceSet.szFaceSetName = sFaceSetName;

		curFaceSet.vKeyFaceIDs.push_back(faceID);
		curFaceSet.vKeyFaceNames.push_back(sCutFaceImg);
		
		vvClusters.push_back(curFaceSet);

		curFaceSet.vKeyFaceFeas.push_back(pFea0);
		vvRepClusters.push_back(curFaceSet);
	}
	else 
	{
		int nCluster0 = vMergeCluster[0];
		if(nMergeClusters == 1)
		{
			flag = 2; //2: insert to an existed faceSet without a new example
			vvClusters[nCluster0].vKeyFaceIDs.push_back(faceID);
			vvClusters[nCluster0].vKeyFaceNames.push_back(sCutFaceImg);

			//delete[] pFea0;	//Modified by Nianzu, 2015-06-05
		}
		else
		{
			//merge clusters
			flag = 1;//1: merge clusters, insert to an existed faceSet and add it as a new example
			//insert into nCluster0
			vvClusters[nCluster0].vKeyFaceIDs.push_back(faceID);
			vvClusters[nCluster0].vKeyFaceNames.push_back(sCutFaceImg);

			vvRepClusters[nCluster0].vKeyFaceIDs.push_back(faceID);
			vvRepClusters[nCluster0].vKeyFaceNames.push_back(sCutFaceImg);
			vvRepClusters[nCluster0].vKeyFaceFeas.push_back(pFea0);

			for(int j =1; j < nMergeClusters; j++)
			 {
				 int nClusterJ = vMergeCluster[j]-j+1;
				 
				 for(int k =0; k < vvClusters[nClusterJ].vKeyFaceIDs.size(); k++)
				 {
					vvClusters[nCluster0].vKeyFaceIDs.push_back(vvClusters[nClusterJ].vKeyFaceIDs[k]);
					vvClusters[nCluster0].vKeyFaceNames.push_back(vvClusters[nClusterJ].vKeyFaceNames[k]);
				}

				 for(int k =0; k < vvRepClusters[nClusterJ].vKeyFaceIDs.size(); k++)
				 {
					vvRepClusters[nCluster0].vKeyFaceIDs.push_back(vvRepClusters[nClusterJ].vKeyFaceIDs[k]);
					vvRepClusters[nCluster0].vKeyFaceNames.push_back(vvRepClusters[nClusterJ].vKeyFaceNames[k]);
					vvRepClusters[nCluster0].vKeyFaceFeas.push_back(vvRepClusters[nClusterJ].vKeyFaceFeas[k]);
					vvRepClusters[nClusterJ].vKeyFaceFeas[k] = NULL;
				}

				 vvClusters.erase(vvClusters.begin()+nClusterJ);
				 vvRepClusters.erase(vvRepClusters.begin()+nClusterJ);
			 }

			//rename face set id and vvRepClusters
			for(int i = 0; i < vvClusters.size(); i++)
			{
				char sFaceSetName[256];
				sprintf(sFaceSetName, "C%d", i);
				vvClusters[i].nFaceSetID = i;
				vvClusters[i].szFaceSetName = sFaceSetName;

				vvRepClusters[i].nFaceSetID = i;
				vvRepClusters[i].szFaceSetName = sFaceSetName;
			}
		}
	}

	delete[] pFea0;	//Modified by Nianzu, 2015-06-05
	
	return flag;       
}


////////////////////////////////////////////////////////////////////
bool cxCompareEdge(const CxEdge &e1, const CxEdge &e2)
{
	if(e1.fWeight < e2.fWeight)
		return true;

	return false;
}

bool myCompareEdgeBigger(const CxEdge &e1, const CxEdge &e2)
{
	if(e1.fWeight > e2.fWeight)
		return true;

	return false;
}

int mySeekNodeParent(std::vector <int> &vNodeParent, int nNode)
{
	//root flag vNodeParent[nNode]== -1
	while (vNodeParent[nNode] >= 0)
	{
		nNode = vNodeParent[nNode];
	}

	return nNode;
}

CvMat* CxBoostFaceRecog::clacSimMat(std::vector <std::string> vFaceImgList, CvMat* &pmSim)
{
	int   nFaces  = (int)vFaceImgList.size();
	int   dim = m_casDetect->getFeatureDim();

	// extract face features
	char   pImgPath[256];
	char   pfeaPath[256];
	float  *pFea  = new float[dim];
	float  *pFea1 = new float[dim];

	printf("extract facial features...\n");
	for(int i =0; i < nFaces; i++)
	{
		printf("%d%%\r", i*100/nFaces);
		sprintf(pImgPath, "%s", vFaceImgList[i].c_str()); //full img path
		sprintf(pfeaPath, "%s_%d.fea", pImgPath, m_fea_type); // full fea path

		FILE *fp = fopen(pfeaPath, "rb");
		if(fp == NULL)
		{
			// load cut face image
			IplImage   *pCutFace = cvLoadImage(pImgPath, CV_LOAD_IMAGE_GRAYSCALE);

			//resize to aligned face size
			if(pCutFace->width != m_cutimg_size || pCutFace->height != m_cutimg_size)
			{
				IplImage   *pTmp = cvCreateImage(cvSize(m_cutimg_size, m_cutimg_size), IPL_DEPTH_8U, 1);
				cvResize(pCutFace, pTmp);
				cvReleaseImage(&pCutFace);
				pCutFace = pTmp;
			}

			m_casDetect->extFeature(pCutFace, pFea);
			cvReleaseImage(&pCutFace);

			fp = fopen(pfeaPath, "wb");
			fwrite(pFea, sizeof(float), dim, fp);
			fclose(fp);
		}
	}

	//calculate dist matrix
	printf("\ncalculate sim matrix...\n");
	pmSim = cvCreateMat(nFaces, nFaces, CV_32FC1);
	float  prob;
	for(int i =0; i < nFaces; i++)
	{
		printf("%d%%\r", i*100/nFaces);

		sprintf(pImgPath, "%s", vFaceImgList[i].c_str()); //full img path
		sprintf(pfeaPath, "%s_%d.fea", pImgPath, m_fea_type); // full fea path
		FILE *fp = fopen(pfeaPath, "rb");
		if(fp != NULL)
		{
			// read feature
			fread(pFea, sizeof(float), dim, fp);
			fclose(fp);
		}

		cvmSet(pmSim, i, i, 1);
		for(int j = i+1; j < nFaces; j++)
		{
			sprintf(pImgPath, "%s", vFaceImgList[j].c_str()); //full img path
			sprintf(pfeaPath, "%s_%d.fea", pImgPath, m_fea_type); // full fea path
			FILE *fp1 = fopen(pfeaPath, "rb");
			if(fp1 != NULL)
			{
				// read feature
				fread(pFea1, sizeof(float), dim, fp1);
				fclose(fp1);
			}

			isSimilarFaces(pFea, pFea1, &prob);
			cvmSet(pmSim, i, j, (double)prob);
			cvmSet(pmSim, j, i, (double)prob);
		}
	}

	delete []pFea;	//Modified by Nianzu, 2015-06-05
	delete []pFea1;	//Modified by Nianzu, 2015-06-05
	printf("sim matrix calculation!\n");

	return pmSim;
}


CvMat* CxBoostFaceRecog::clacSimMat(std::vector <CvMat*> matFea, CvMat* &pmSim)
{
	int   nFaces  = (int)matFea.size();
	int   dim = m_casDetect->getFeatureDim();

	//calculate dist matrix
	printf("\ncalculate sim matrix...\n");
	pmSim = cvCreateMat(nFaces, nFaces, CV_32FC1);
	float  prob;
	float *pFea, *pFea1;
	for(int i =0; i < nFaces; i++)
	{
		printf("%d%%\r", i*100/nFaces);

		pFea = matFea[i]->data.fl;
		cvmSet(pmSim, i, i, 1);
		for(int j = i+1; j < nFaces; j++)
		{
			pFea1 = matFea[j]->data.fl;
			isSimilarFaces(pFea, pFea1, &prob);
			cvmSet(pmSim, i, j, (double)prob);
			cvmSet(pmSim, j, i, (double)prob);
		}
	}
	printf("sim matrix calculation!\n");
	return pmSim;
}

//HAC (create minspan tree) by kruskal algorithm 
int CxBoostFaceRecog::clusterHAC(CvMat* pmSim,  vFaceSet& vvFaceSet, float fThreshold /*=-1*/,
								 int nMinClusterNum /*=-1*/, std::vector<int> *pvExemplars /*=NULL*/)
{
	if(fThreshold < 0 ) fThreshold = getDefThreshold()*1.1;


	std::vector< std::vector<int> > vvClusters;
	//get edges
	int nNodes = pmSim->cols;
	int nEdges=0;
	std::vector < CxEdge > vEdges(nNodes*(nNodes-1)/2);

	for(int i =0; i < nNodes; i++)
	{
		for(int j = i+1; j < nNodes; j++)
		{
			float fSim =  cvmGet(pmSim, i, j);
			if(fSim >= fThreshold)
			{
				vEdges[nEdges].nFrom  = i;
				vEdges[nEdges].nTo    = j;
				vEdges[nEdges].fWeight= fSim;
				nEdges++;
			}
		}
	}

	vEdges.resize(nEdges);

	//sort edges from big fWieght to small weight
	sort(vEdges.begin(), vEdges.end(), myCompareEdgeBigger);

	//recursive merge minimum edges if they are in different clusters
	int nNode = 0;
	int nEgde = 0;
	std::vector <int> vNodeParent(nNodes, -1);

	int nClusters = nNodes; 
	while(nNode < nNodes && nEgde < nEdges)
	{
		//if cluster nMinClusterNum
		if(nMinClusterNum> 0 && nMinClusterNum== nClusters)
			break;

		int nRoot1 = mySeekNodeParent (vNodeParent, vEdges[nEgde].nFrom);
		int nRoot2 = mySeekNodeParent (vNodeParent, vEdges[nEgde].nTo);
		if (nRoot1 != nRoot2)//judge loop whether they are from the same cluster
		{
			vNodeParent[nRoot1] = nRoot2;
			nNode++;
			nClusters--;
		}
		nEgde++;
	}

	//assign clusters by vNodeParent
	std::vector <int> vNodeFlag(nNodes, 1);
	std::vector < std::vector <int> > vvTmpClusters(nNodes);
	for(int i =0; i < nNodes; i++)
	{
		if(vNodeFlag[i] == 1) //a seed which is not be processed
		{
			nNode = i;
			std::vector <int> vNodes;
			vNodeFlag[nNode] = 0; //flag it has been added
			vNodes.push_back(nNode);

			//add traced nodes to the root
			while (vNodeParent[nNode] >= 0)
			{
				nNode = vNodeParent[nNode];

				if(vNodeFlag[nNode] == 1)
				{
					vNodeFlag[nNode] = 0;
					vNodes.push_back(nNode);
				}
			}

			vvTmpClusters[nNode].insert(vvTmpClusters[nNode].end(), vNodes.begin(), vNodes.end());
		}
	}

	//add nonempty subcluster to vvClusters
	vvClusters.clear();
	for(int i = 0; i < vvTmpClusters.size(); i++)
	{
		if(vvTmpClusters[i].size() > 0)
		{
			vvClusters.push_back(vvTmpClusters[i]);
		}
	}

	//extract an examplear in each cluster
	int nItem = 0;
	float fSim, fMaxSim;
	int nClusterSize = (int)vvClusters.size();
	if( pvExemplars != NULL)	//Set the avg One as the exemplar
	{
		pvExemplars->resize(nClusterSize);
		for(int i =0; i < nClusterSize; i++)
		{
			nItem = 0;
			fMaxSim = FLT_MIN;
			for(int m = 0; m < vvClusters[i].size(); m++)
			{
				fSim = 0;
				for(int n=0; n < vvClusters[i].size(); n++)
					fSim += cvmGet(pmSim, m, n);

				if(fSim > fMaxSim)
				{
					fMaxSim = fSim;
					nItem = m;
				}
			}
			(*pvExemplars)[i] = nItem;
		}
	}

	//copy vvClusters to vvFaceSet
	char sFaceSetName[16];
	for(int i =0; i < nClusterSize; i++)
	{
		FaceSet curFaceSet;
		sprintf(sFaceSetName, "C%d", i);
		curFaceSet.nFaceSetID = i;
		curFaceSet.szFaceSetName = sFaceSetName;

		for(int j=0; j < vvClusters[i].size(); j++)
		{
			curFaceSet.vKeyFaceIDs.push_back(vvClusters[i][j]);
			curFaceSet.vKeyFaceNames.push_back("");
		}

		vvFaceSet.push_back(curFaceSet);
	}

	return nClusterSize;
}

//copy vvClusters[cB] to vvClusters[cA], then remove vvClusters[cB]
void CxBoostFaceRecog::mergeClusters(vFaceSet& vvClusters, int cA, int cB,  vFaceSet* vvRepClusters /*= NULL*/)
{
	//update propery info 
	int nWeightA = m_vecFaceSet[cA].nFaceSetWeight;
	int nWeightB = m_vecFaceSet[cB].nFaceSetWeight;

	//swap nFaceSetID and szFaceSetName by nFaceSetWeight and szFaceSetName
	if(nWeightB > nWeightA)
	{
		m_vecFaceSet[cA].nFaceSetID    = m_vecFaceSet[cB].nFaceSetID;
	}

	int pos0 =  m_vecFaceSet[cA].szFaceSetName.find("unknown");
	int pos1 =  m_vecFaceSet[cB].szFaceSetName.find("unknown");
	if(pos0 >= 0 && pos1 < 0)
	{
		m_vecFaceSet[cA].szFaceSetName = m_vecFaceSet[cB].szFaceSetName;
	}
	else if(nWeightB > nWeightA)
		m_vecFaceSet[cA].szFaceSetName = m_vecFaceSet[cB].szFaceSetName;

	m_vecFaceSet[cA].nFaceSetWeight= nWeightA + nWeightB;
	m_vecFaceSet[cA].nCurFrameID   = MAX(m_vecFaceSet[cA].nCurFrameID, m_vecFaceSet[cB].nCurFrameID);
	m_vecFaceSet[cA].nCurTrackID   = MAX(m_vecFaceSet[cA].nCurTrackID, m_vecFaceSet[cB].nCurTrackID);
	m_vecFaceSet[cA].szFaceSetName = m_vecFaceSet[cB].szFaceSetName;

	for(int i =0; i < vvClusters[cB].vConflictFaceSetIDs.size(); i++)
		m_vecFaceSet[cA].vConflictFaceSetIDs.push_back( m_vecFaceSet[cB].vConflictFaceSetIDs[i]);

	//copy face images from cluster cB to cluster cA

	//copy faces and features from cluster cB to cluster cA
	int nFacecA = vvClusters[cA].vKeyFaceIDs.size();
	int nFacecB = vvClusters[cB].vKeyFaceIDs.size();
	int nFaceCount = nFacecA + nFacecB;
	for(int i =0; i < vvClusters[cB].vKeyFaceIDs.size(); i++)
	{
		vvClusters[cA].vKeyFaceIDs.push_back(vvClusters[cB].vKeyFaceIDs[i]);
		vvClusters[cA].vKeyFaceNames.push_back(vvClusters[cB].vKeyFaceNames[i]);

		vvClusters[cA].vKeyFaceFeas.push_back(vvClusters[cB].vKeyFaceFeas[i]);
		vvClusters[cB].vKeyFaceFeas[i] = NULL;

		vvClusters[cA].vKeyFaceImgs.push_back(vvClusters[cB].vKeyFaceImgs[i]);
		vvClusters[cB].vKeyFaceImgs[i] = NULL;
	}
	for(int k=0; k < m_fea_dim; k++)
		vvClusters[cA].avgFaceFea[k] = (vvClusters[cA].avgFaceFea[k] * nFacecA + vvClusters[cB].avgFaceFea[k] * nFacecB ) / nFaceCount;
	
	
	//remove cluster cB
	vvClusters.erase(vvClusters.begin()+cB);

	if(vvRepClusters)
	{
		for(int i =0; i < (*vvRepClusters)[cB].vKeyFaceIDs.size(); i++)
		{
			(*vvRepClusters)[cA].vKeyFaceIDs.push_back((*vvRepClusters)[cB].vKeyFaceIDs[i]);
			(*vvRepClusters)[cA].vKeyFaceNames.push_back((*vvRepClusters)[cB].vKeyFaceNames[i]);

			(*vvRepClusters)[cA].vKeyFaceFeas.push_back((*vvRepClusters)[cB].vKeyFaceFeas[i]);
			(*vvRepClusters)[cB].vKeyFaceFeas[i] = NULL;
		}
		vvRepClusters->erase(vvRepClusters->begin()+cB);
	}
}

/////////////////////////////////////////////////////////
//rank Order clustering
int findOrder(std::vector < CxEdge > &vSet, int nItem)
{
	int nSize = (int)vSet.size();
	for(int i=0 ; i < nSize; i++)
	{
		if(vSet[i].nTo == nItem)
			return i;
	}
	return -1;
}

float CxBoostFaceRecog::nearestCluseterDist(CvMat* pmSim, vFaceSet& vvClusters, int cA, int cB, CvMat* pmTmpClusterDist /*=NULL*/)
{
	float  sim = 0;
	float  simMax  = 0;
	float  distMin = FLT_MAX;

	if(pmTmpClusterDist)
	{
		distMin = cvmGet(pmTmpClusterDist, cA, cB);
		if(distMin > 0)  return distMin;
	}
	

	int Na = (int)vvClusters[cA].vKeyFaceIDs.size();
	int Nb = (int)vvClusters[cB].vKeyFaceIDs.size();

	for(int i =0; i < Na; i++)
		for(int j =0; j < Nb; j++)
		{
			if(pmSim)
				sim = cvmGet(pmSim, vvClusters[cA].vKeyFaceIDs[i], vvClusters[cB].vKeyFaceIDs[j]);
			
			if(sim > simMax)
				simMax = sim;
		}

	distMin = 1- simMax;
	
	if(pmTmpClusterDist)
	{
		cvmSet(pmTmpClusterDist, cA, cB, distMin);
	}

	return distMin;
}

float CxBoostFaceRecog::rankOrderCluseterDist(CvMat* pmSim, vFaceSet& vvClusters, int cA, int cB, CvMat* pmTmpClusterDist /*=NULL*/)
{
	int nNodes = (int)vvClusters.size();

	//Oa, Order of face a
	int   nEdgesA = 0, nEdgesB = 0;
	std::vector < CxEdge > vEdgesA(nNodes);
	std::vector < CxEdge > vEdgesB(nNodes);
	for(int j = 0; j < nNodes; j++)
	{
		float fDist =  nearestCluseterDist(pmSim, vvClusters, cA, j, pmTmpClusterDist);
		vEdgesA[nEdgesA].nFrom  = cA;
		vEdgesA[nEdgesA].nTo    = j;
		vEdgesA[nEdgesA].fWeight= fDist;
		nEdgesA++;
	}

	vEdgesA.resize(nEdgesA);
	//sort edges from big fWieght to small fWieght
	sort(vEdgesA.begin(), vEdgesA.end(), cxCompareEdge);

	for(int j = 0; j < nNodes; j++)
	{
		float fDist =  nearestCluseterDist(pmSim, vvClusters, cB, j, pmTmpClusterDist);
		vEdgesB[nEdgesB].nFrom  = cB;
		vEdgesB[nEdgesB].nTo    = j;
		vEdgesB[nEdgesB].fWeight= fDist;
		nEdgesB++;
	}
	vEdgesB.resize(nEdgesB);
	//sort edges from big fWieght to small fWieght
	sort(vEdgesB.begin(), vEdgesB.end(), cxCompareEdge);

	//Ob, Order of face b
	int nOb_fa = 0;
	int nOa_fb = 0;
	int nOa_b = findOrder(vEdgesA, cB);
	int nOb_a = findOrder(vEdgesB, cA);
	for(int i = 0; i < nOa_b; i++)
	{
		nOb_fa += findOrder(vEdgesB, vEdgesA[i].nTo);
	}

	for(int i = 0; i < nOb_a; i++)
	{
		nOa_fb += findOrder(vEdgesA, vEdgesB[i].nTo);
	}

	float  rankDist = 1.0f*(nOb_fa+nOa_fb)/ MIN(nOa_b, nOb_a);
	return rankDist;
}

float CxBoostFaceRecog::normalizedCluseterDist(CvMat* pmSim, vFaceSet& vvClusters, int cA, int cB, CvMat* pmTmpClusterDist /*=NULL*/)
{
	float nearDist = nearestCluseterDist(pmSim, vvClusters, cA, cB, pmTmpClusterDist);

	float phi  = 0;
	int Na = (int)vvClusters[cA].vKeyFaceIDs.size();
	int Nb = (int)vvClusters[cB].vKeyFaceIDs.size();
	int Nnode = Na+Nb;
	int K  = MIN(Nnode, 20);

	farray vSim;
	vSim.resize(Na+Nb);
	/*
	for(int i = 0; i < Na; i++ )
	{
		for(int j =0; j < Na; j++)
			 vSim[j] =  cvmGet(pmSim, vvClusters[cA].vKeyFaceIDs[i], vvClusters[cA].vKeyFaceIDs[j]);
		
		for(int j =0; j < Nb; j++)
			vSim[j+Na] =  cvmGet(pmSim, vvClusters[cA].vKeyFaceIDs[i], vvClusters[cB].vKeyFaceIDs[j]);

		if(Nnode > K)
			std::partial_sort( vSim.begin(), vSim.begin()+K, vSim.end(), std::greater<float>()); 
		
		for(int j =0; j < K; j++)
			phi += vSim[j];
	}

	for(int i = 0; i < Nb; i++ )
	{
		for(int j =0; j < Na; j++)
			vSim[j] =  cvmGet(pmSim, vvClusters[cB].vKeyFaceIDs[i], vvClusters[cA].vKeyFaceIDs[j]);
		for(int j =0; j < Nb; j++)
			vSim[j+Na] =  cvmGet(pmSim, vvClusters[cB].vKeyFaceIDs[i], vvClusters[cB].vKeyFaceIDs[j]);
	
		if(Nnode > K+1)
			std::partial_sort( vSim.begin(), vSim.begin()+K, vSim.end(), std::greater<float>()); 

		for(int j =0; j < K; j++)
			phi += vSim[j];
	}
	*/
	K   = K-1;
	phi = phi-Nnode;

	phi = Nnode-phi/K;
	phi = phi/Nnode;

	float normDist = 0.0f;
	if(phi >FLT_MIN) normDist = nearDist /phi;

	return normDist;
}

// main API of rankOrder dist
float CxBoostFaceRecog::rankOrderCluster(CvMat* pmSim, vFaceSet& vvClusters, 
										 float rankDistThresh /*= 12*/, float normDistThresh /*=1.02*/)
{
	//first HAC clustering
	//clusterHAC(pmSim, vvClusters, 0.65);

	//then rankOrder clustering
	int nFace = pmSim->rows;
	int nFaceSet = (int)vvClusters.size();
	char sName[256];

	if( nFaceSet == 0)
	{
		FaceSet faceset;
		for(int i = 0; i < nFace; i++)
		{
			faceset.nFaceSetID = i;
			faceset.szFaceSetName = "C";
			faceset.vKeyFaceIDs.resize(1);
			faceset.vKeyFaceIDs[0] = i;

			vvClusters.push_back(faceset);
		}
	}
	
	//compute all pairs of cA, Cb
	CxEdge edge;
	std::vector < CxEdge > vMergePairs;
	bool bMerge = true;
	vFaceSet vvClustersTmp;

	while(bMerge)
	{
		vMergePairs.clear();
		for(int cA = 0; cA < vvClusters.size(); cA++)
		{
			if(vvClusters[cA].nFaceSetID != cA) continue;

			for(int cB = cA+1; cB < vvClusters.size(); cB++)
			{
				if(vvClusters[cB].nFaceSetID != cB) continue;

				float rankDist = rankOrderCluseterDist(pmSim, vvClusters, cA, cB);
				float normDist = normalizedCluseterDist(pmSim, vvClusters, cA, cB);
				if(rankDist < rankDistThresh && normDist < normDistThresh)
				{
					edge.nFrom = cA;
					edge.nTo   = cB;
					edge.fWeight = normDist;

					vMergePairs.push_back(edge);
				}
			}
		}

		bMerge = vMergePairs.size() > 0;
		if(bMerge)
		{
			//merge subclusters
			for(int i =0; i< vMergePairs.size(); i++)
			{
				edge = vMergePairs[i];
				int nFrom = vvClusters[edge.nFrom].nFaceSetID;
				int nTo   = vvClusters[edge.nTo].nFaceSetID;

				if(nFrom != edge.nFrom && nTo != edge.nTo) continue; //has been merged
					
				while(nFrom != vvClusters[nFrom].nFaceSetID )
					nFrom = vvClusters[nFrom].nFaceSetID;
				
				while(nTo != vvClusters[nTo].nFaceSetID)
					nTo = vvClusters[nTo].nFaceSetID;

				if(nFrom == nTo) continue; //has been merged

				int sizeFrom = (int)vvClusters[nFrom].vKeyFaceIDs.size();
				int sizeTo   = (int)vvClusters[nTo].vKeyFaceIDs.size();

				if(sizeTo < sizeFrom ) //assign nTo with more items
				{
					int tmp = nTo;
					nTo = nFrom;
					nFrom = tmp;
				}

				for(int j = 0; j < vvClusters[nFrom].vKeyFaceIDs.size(); j++)
					vvClusters[nTo].vKeyFaceIDs.push_back(vvClusters[nFrom].vKeyFaceIDs[j]);
				
				 vvClusters[nFrom].nFaceSetID = nTo;
			}

			//remove the merged sub-clusters
			vvClustersTmp.clear();
			for(int i = 0, j = 0; i < vvClusters.size(); i++)
			{
				if(vvClusters[i].nFaceSetID == i)
				{
					vvClustersTmp.push_back(vvClusters[i]);
					vvClustersTmp[j].nFaceSetID = j;
					j++;
				}
			}
			vvClusters = vvClustersTmp;
		}
	}

	//remove the merged sub-clusters
	vvClustersTmp.clear();
	for(int i = 0, j = 0; i < vvClusters.size(); i++)
	{
		if(vvClusters[i].nFaceSetID == i)
		{
			vvClustersTmp.push_back(vvClusters[i]);
			vvClustersTmp[j].nFaceSetID = j;
			j++;
		}
	}
	vvClusters = vvClustersTmp;

	//merge all single element clusters into the un-grouped cluster Cun
	FaceSet Cun;
	int     removeThresh = 1;
	Cun.nFaceSetID = -1;
	Cun.szFaceSetName = "unKonwn";

	for(int i =0; i < vvClusters.size(); i++)
	{
		int nFace = (int)vvClusters[i].vKeyFaceIDs.size();
		if(nFace <= removeThresh)
		{
			for(int j = 0; j < nFace; j++)
				Cun.vKeyFaceIDs.push_back(vvClusters[i].vKeyFaceIDs[j]);

			vvClusters.erase(vvClusters.begin()+i);
			i--;
		}
	}

	for(int i =0; i < vvClusters.size(); i++)
	{
		vvClusters[i].nFaceSetID = i;

		sprintf(sName, "C%d", i);
		vvClusters[i].szFaceSetName = sName;

		sort(vvClusters[i].vKeyFaceIDs.begin(), vvClusters[i].vKeyFaceIDs.end());
	}
	vvClusters.push_back(Cun);

	return vvClusters.size();
}
