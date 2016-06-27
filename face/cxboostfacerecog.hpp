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

#pragma once
#include <vector>
#include <string>

#include "opencv/cv.h"
#include "opencv/highgui.h"

#include "cxrecognizerbase.hpp"
#include "cxboost.hpp"
#include "cxcasboost.hpp"
#include "facelistxml.hpp"


typedef struct STRU_EDGE
{
	int nFrom;
	int nTo;
	float fWeight;
} CxEdge;


class CxBoostFaceRecog
{
public:
	CxBoostFaceRecog()
	{
		m_recognizerType = RECOGNIZER_CAS_GLOH;
		m_pFea = NULL;
		m_casDetect = NULL;

		m_fea_dim  = 0;
		m_fea_type = -1;
		m_vTrainID.clear();
		m_votebuff_idx = 0;
		
		for(int i =0; i < NVOTEBUFF; i++)
		{
			m_votebuff_face_id[i] = -1;
			m_votebuff_face_label_vote[i].resize(NVOTEPERSON);
		}
		m_nFaces = 0;

		m_oldface_trackid = -1;
		m_oldfaceset_id   = -1;
		m_oldframe_id     = -1;
		m_maxfaceset_id   = 0;
	};

	~CxBoostFaceRecog(void);

		// clear memory
	void clear();

	// for load XML/YAML format model
	int  load(int recognizerType, const char* path, const char* filename, int cutimg_size=128);
	int  loadFaceModelXML(const char *sPathXML = NULL, bool bLoadExtFea = true);
	
	void getMergedFaceSet(vFaceSet& vvClusters, int weigthThreshold = 0);
	void saveFaceModelXML(const char *sPathXML, vFaceSet *pvecFaceSet =NULL);
	
	int  insertEmptyFaceSet(char *faceName, bool createFolder = true, int nFaceSetID = -1 );
	bool tryInsertFace(IplImage* pCutFace, int nFaceSetIdx, bool bForceInsert = false);
	int  removeFaceSet(int nFaceSetIdx);
	int  removeFace(int nFaceSetIdx, int faceIdx);

	char     *getFaceImgDBPath() { return m_szImgDBPath; }
	vFaceSet *getFaceSets()  { return &m_vecFaceSet; }
	std::vector<std::string> *getKeyFacePaths(int faceSetIdx);
	int   getFaceSetSize(int nFaceSetIdx = -1) { if(nFaceSetIdx == -1) return (int)m_vecFaceSet.size();  return (int)m_vecFaceSet[nFaceSetIdx].vKeyFaceNames.size(); } 
	const char* getKeyFacePath(int nFaceSetIdx, int nFaceIdx = -1) {  if(nFaceIdx == -1) return m_vecFaceSet[nFaceSetIdx].szFaceSetName.c_str(); return m_vecFaceSet[nFaceSetIdx].vKeyFaceNames[nFaceIdx].c_str();}

	int   getDefRound()     { return m_defRound; }
	float getDefThreshold() { return m_defThreshold; }
	int   getFeatureDim()   { return m_fea_dim; } 
	int   getFeatureType()  { return m_fea_type; } 
	

	int   predict(IplImage* pCutFaceImg, float *prob =NULL, bool bAutoCluster = false, int face_trackid = -1, int frameid = -1);
	int   predict(float* pFea, float *prob =NULL);
	int   voteLabel(int face_trackid, int label); 

	float clusterSim(float *pFea, int idx);
	float clusterSim(int idx1, int idx2);
	float clusterAvgSim(int idx1, int idx2);
	bool  isSimilarFaces(float* pFea1, float* pFea2, float *pProb =NULL);
	bool  isSimilarToFaceSet(IplImage* pCutFace, int nFaceSetIdx);
	
	void  extFeature(IplImage* pCutFace, float* pFea);

	const char* getFaceName(int nFaceSetID);
	int   getFaceSetIdx(int nFaceSetID);
	int   getFaceSetID(int nFaceSetIdx);

	////////////////////////////////////////////////////////////////////
	//face clustering but need further refinement
	int forwardCluster(float* pFea, int faceID, char *sCutFaceImg, vFaceSet& vvClusters, vFaceSet& vvRepClusters, float fThreshold =-1);
	
	int clusterHAC(CvMat* pmSim, vFaceSet& vvFaceSet, float fThreshold = -1,
		int nMinClusterNum =-1, std::vector<int> *pvExemplars =NULL);

	CvMat* clacSimMat(std::vector <std::string> vFaceImgList, CvMat* &pmSim);
	CvMat* clacSimMat(std::vector <CvMat*> matFea, CvMat* &pmSim);

	void  mergeClusters(vFaceSet& vvClusters, int cA, int cB, vFaceSet* vvRepClusters = NULL);

	float nearestCluseterDist(CvMat* pmSim, vFaceSet& vvClusters, int cA, int cB, CvMat* pmTmpClusterDist = NULL);
	float rankOrderCluseterDist(CvMat* pmSim, vFaceSet& vvClusters, int cA, int cB, CvMat* pmTmpClusterDist = NULL);
	float normalizedCluseterDist(CvMat* pmSim, vFaceSet& vvClusters, int cA, int cB, CvMat* pmTmpClusterDist = NULL);
	float rankOrderCluster(CvMat* pmSim, vFaceSet& vvClusters, float rankDistThresh = 12, float normDistThresh = 1.02);
	////////////////////////////////////////////////////////////////////
private:
	vFaceSet m_vecFaceSet;
	int      m_nFaces;
	std::vector <int>    m_vTrainID;

	int   m_fea_dim;
	int   m_fea_type;
	int   m_cutimg_size;  //tested cutface image size
	float m_defThreshold;
	int   m_defRound;

	CxRecognizerBase *m_casDetect; //use cascade or boosting detector to recognize face
	int   m_recognizerType;
	int   m_oldface_trackid;
	int   m_oldfaceset_id;
	int   m_oldframe_id;
	int   m_maxfaceset_id;

	float* m_pFea;

	// for vote label
	static const int NVOTEBUFF   		= 16;    // voting face label buff
	static const int NVOTEPERSON   		= 4;     // voting max person number
	int		 m_votebuff_idx;
	int      m_votebuff_face_id[NVOTEBUFF];
	int      m_votebuff_face_label[NVOTEBUFF][NVOTEPERSON]; 
	std::vector<int>  m_votebuff_face_label_vote[NVOTEBUFF];
	
	//face exemplar image folder path
	char m_szImgDBPath[256];
};
