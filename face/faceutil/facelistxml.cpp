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

#include <opencv/cxcore.h>
#include <string>
#include <stdio.h>
#include "facelistxml.hpp"

using namespace std;

CFaceListXML::CFaceListXML()
{

}


CFaceListXML::~CFaceListXML()
{

}

// how to validate the file format?
// what if the format is not correct?
bool CFaceListXML::load(const char* fname, vFaceSet& vFaceSets)
{
	if(fname == NULL)
		return false;
    
	CvFileStorage* fs = cvOpenFileStorage( fname, 0, CV_STORAGE_READ );
	if( fs == NULL )
		return false;

    vFaceSets.clear();

    //into the root
	CvFileNode *pRootNode = cvGetRootFileNode(fs);
	
	/* new format using OpeCN XML I/O
	<?xml version="1.0"?>
	<opencv_storage>
	<FaceFeaPath>PIROFileList_Face.fea</FaceFeaPath>
	<ImgDBPath>\\waqp01\ARL-PIRO\PIRO\carole_images_only\IPTC_labels_lowres_dec15</ImgDBPath>
	<FaceSetNum>2</FaceSetNum>
	<FaceSet_0>
		<ID>0<ID>
		<Name>Mike</Name>
		<Num>2</Num>
		<KeyFace_0>0</KeyFace_0>
		<KeyFace_1>1</KeyFace_1>
	</FaceSet_0>
	<FaceSet_1>
		<ID>1<ID>
		<Name>Tom</Name>
		<Num>3</Num>
		<KeyFace_0>2</KeyFace_0> 
		<KeyFace_1>10</KeyFace_0>
		<KeyFace_2>15</KeyFace_0>
	</FaceSet_1>
	</opencv_storage>

	/* old format
	<FaceRetrieval>
	  <FaceSet id="0" Name="Mike">
		<KeyFace id="0"/>
		<KeyFace id="1"/>
	  </FaceSet>
	  <FaceSet id="1"  Name="Tom">
		<KeyFace id="2"/>
		<KeyFace id="10"/>
		<KeyFace id="15"/>
	  </FaceSet>
	</FaceRetrieval>
	*/

	m_szFeaName = cvReadStringByName( fs, pRootNode, "FaceFeaPath", NULL );
	m_szImgDBPath = cvReadStringByName( fs, pRootNode, "ImgDBPath", NULL );
	m_iCurFaceIndex = cvReadIntByName( fs, pRootNode, "CurFaceIndex", 0 );
	int nFaceSetNum = cvReadIntByName( fs, pRootNode, "FaceSetNum", 0 );

	CvFileNode *pFaceSetNode;
	char strBuff[256];
	FaceSet curFaceSet;

	for(int i=0; i < nFaceSetNum; i++)
	{
		sprintf(strBuff, "FaceSet_%d", i);
		pFaceSetNode = cvGetFileNodeByName( fs, pRootNode, strBuff);
				
		if(pFaceSetNode)
		{
			// curFaceSet.nFaceSetID    = cvReadIntByName( fs, pFaceSetNode, "ID", -1 );
			curFaceSet.nFaceSetID = i;// ID should be the input order
			curFaceSet.szFaceSetName  = cvReadStringByName( fs, pFaceSetNode, "Name");
			curFaceSet.nFaceSetWeight = cvReadIntByName( fs, pFaceSetNode, "Weight");
			int nKeyFaceNum  = cvReadIntByName( fs, pFaceSetNode, "Num", 0);
			curFaceSet.vKeyFaceIDs.resize(nKeyFaceNum,  -1);
			curFaceSet.vKeyFaceNames.resize(nKeyFaceNum, "");

			for(int j=0; j< nKeyFaceNum; j++)
			{
				sprintf(strBuff, "KeyFace_%d", j);
				int faceID = curFaceSet.vKeyFaceIDs[j] = cvReadIntByName( fs, pFaceSetNode, strBuff, -1 );
				//if(faceID == -1) 
				{
					sprintf(strBuff, "KeyFaceName_%d", j);
					curFaceSet.vKeyFaceNames[j] = cvReadStringByName( fs, pFaceSetNode, strBuff); //relative path
					//curFaceSet.vKeyFaceNames[j] =  m_szImgDBPath + curFaceSet.vKeyFaceNames[j]; //full path
				}
			}
			vFaceSets.push_back(curFaceSet);
		}
	}

	cvReleaseFileStorage( &fs );

	return true;
}

bool CFaceListXML::save(const char* fname, vFaceSet& vFaceSets,  int iCurFaceIndex /*=0*/)
{
    if(fname == NULL)
        return false;

    CvFileStorage* fs = cvOpenFileStorage( fname, 0, CV_STORAGE_WRITE );
    if( fs == NULL )
        return false;

    // write the root child node
    // write each FaceSet struct
    cvWriteString(fs, "FaceFeaPath", m_szFeaName.c_str());
    cvWriteString(fs, "ImgDBPath", m_szImgDBPath.c_str());
	cvWriteInt(fs, "CurFaceIndex", iCurFaceIndex);
	cvWriteInt(fs, "FaceSetNum", (int)vFaceSets.size());

    char strBuff[256];
    for(int i=0; i < (int)vFaceSets.size(); i++)
    {
        sprintf(strBuff, "FaceSet_%d", i);
        cvStartWriteStruct(fs, strBuff, CV_NODE_MAP);
        cvWriteInt(fs, "ID", vFaceSets[i].nFaceSetID);
		//cvWriteInt(fs, "ID", i);
        cvWriteString(fs, "Name", vFaceSets[i].szFaceSetName.c_str());
		cvWriteInt(fs, "Weight", (int)vFaceSets[i].nFaceSetWeight);
        cvWriteInt(fs, "Num", (int)vFaceSets[i].vKeyFaceNames.size());
		
        for(int j=0; j< (int)vFaceSets[i].vKeyFaceIDs.size(); j++)
        {
			if(vFaceSets[i].vKeyFaceIDs[j] >= 0)
			{
				sprintf(strBuff, "KeyFace_%d", j);
				cvWriteInt(fs, strBuff, vFaceSets[i].vKeyFaceIDs[j]);
			}
        }

		for(int j=0; j< (int)vFaceSets[i].vKeyFaceNames.size(); j++)
		{
			if(vFaceSets[i].vKeyFaceNames[j] != "")
			{
				sprintf(strBuff, "KeyFaceName_%d", j);
				cvWriteString(fs, strBuff, vFaceSets[i].vKeyFaceNames[j].c_str());
			}
		}

        cvEndWriteStruct( fs );
    }

    cvReleaseFileStorage( &fs );

    return true;
}

bool CFaceListXML::saveTxtListFile(const char* fname, vFaceSet &vFaceSets)
{
	if(fname == NULL)
		return false;

	FILE *fp = fopen(fname, "wt");
	if( fp == NULL )
		return false;

	// write each record, facesetID faceName imagePath
	for(int i=0; i < (int)vFaceSets.size(); i++)
	{
		for(int j=0; j< (int)vFaceSets[i].vKeyFaceNames.size(); j++)
		{
			if(m_szImgDBPath.empty())
				fprintf(fp, "%06d %s\t %s\n", vFaceSets[i].nFaceSetID, vFaceSets[i].szFaceSetName.c_str(), vFaceSets[i].vKeyFaceNames[j].c_str());
			else
				fprintf(fp, "%06d %s\t %s//%s\n", vFaceSets[i].nFaceSetID, vFaceSets[i].szFaceSetName.c_str(), m_szImgDBPath.c_str(), vFaceSets[i].vKeyFaceNames[j].c_str());
		}
	}

	fclose(fp);

	return true;
}

// here the fname is not path, it is only the file title, 
// e.g abc, not abc.xml
bool CFaceListXML::create(const char* fname)
{
	if(fname == NULL)
		return false;

	std::string sFullName;
	sFullName = fname;
	sFullName.append(".xml");
	CvFileStorage* fs = cvOpenFileStorage( sFullName.c_str(), 0, CV_STORAGE_WRITE );
	if( fs == NULL )
		return false;

	m_szFeaName = fname;
	m_szImgDBPath = fname;
	m_iCurFaceIndex = 0;
	// write the root child node
	cvWriteString(fs, "FaceFeaPath", m_szFeaName.c_str());
	cvWriteString(fs, "ImgDBPath", m_szImgDBPath.c_str());
	cvWriteInt(fs, "CurFaceIndex", m_iCurFaceIndex);
	cvWriteInt(fs, "FaceSetNum", 0);

	cvReleaseFileStorage( &fs );

	return true;
}

///////////////////////////////////////////////////////////////
