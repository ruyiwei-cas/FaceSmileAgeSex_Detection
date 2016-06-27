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

#ifndef _FACE_XML_HPP
#define _FACE_XML_HPP

#include "basetypes.hpp"

class CFaceListXML
{
public:
	CFaceListXML();
	~CFaceListXML();

	bool load(const char* fname, vFaceSet &vFaceSets);
	bool save(const char* fname, vFaceSet &vFaceSets, int iCurFaceIndex = 0 );
	bool saveTxtListFile(const char* fname, vFaceSet &vFaceSets);
	bool create(const char* fname);

	std::string m_szFeaName;
	std::string m_szImgDBPath;
	int m_iCurFaceIndex;

	std::string getFaceFeaPath() { return m_szFeaName;}
	std::string getImgDBPath()   { return m_szImgDBPath;}
	int  getCurFaceIndex()       { return m_iCurFaceIndex;}
	void setImgDBPath(char *sImgDBPath) { m_szImgDBPath = sImgDBPath; }
};

#endif
