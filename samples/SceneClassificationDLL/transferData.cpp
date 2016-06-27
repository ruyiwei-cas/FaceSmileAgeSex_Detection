/**
*** Copyright (C) 1985-2014 Intel Corporation.  All rights reserved.
***
*** The information and source code contained herein is the exclusive
*** property of Intel Corporation and may not be disclosed, examined
*** or reproduced in whole or in part without explicit written authorization
*** from the company.
***
*** Perceptual Application Innovation Lab, Intel Labs China.
**/

/*!
*   @file        scene.cpp
*   @brief       utility function for scene classification
*   @author      LI Xue, Intel Labs China
*   copyright reserved 2014, please do not remove this head
*/

#include <opencv2/opencv.hpp>  
#include <iostream>
#include <string>
#include <vector>
#include <iterator>
#include <fstream>
#include "transferData.h"

using namespace cv;
using namespace std;




////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the Photo indexing Utility
///		written by Li, Xue
///   
/// Acknowledge     :
///
/// Function name   : loadFile2Mat
/// Description     : load the specified .dat file into a Mat. 
///
/// Argument        : filename -- name of the specified .dat file.
///
/// Return type     : Mat -- contains data saved in the .dat file.
///
/// Create Time     : 2014-12-5  12:45
///
///
/// Side Effect     : 
///
///////////////////////////////////////////////////////////////////////////////////////////////
Mat loadFile2Mat(string filename)
{
	ifstream infile;
	infile.open(filename.c_str(), ios_base::binary);
	if(!infile)
	{
		cout<<"cannot open file!"<<endl;
		cout<<filename<<endl;
		Mat myMat;
		return myMat;
	}
	int sz[3];
	infile.read((char*)&sz,sizeof(sz));
	
	Mat myMat(sz[0],sz[1],sz[2]);
	if(CV_64FC1==sz[2])
	{
		for(int i=0;i<myMat.rows;++i)
		{
			double *p = myMat.ptr<double>(i);
			for(int j=0;j<myMat.cols;++j)	
					infile.read((char*)&p[j],sizeof(p[j]));
		}
	}
	if(CV_64FC3==sz[2])
	{
		for(int k=0;k<myMat.channels();++k)
			for(int i=0;i<myMat.rows;++i)
				for(int j=0;j<myMat.cols;++j)				
					infile.read((char*)&myMat.at<Vec3d>(i,j)[k],sizeof(myMat.at<Vec3d>(i,j)[k]));
	}
	if(CV_8UC1==sz[2])
	{
		for(int i=0;i<myMat.rows;++i)
		{
			uchar *p = myMat.ptr<uchar>(i);
			for(int j=0;j<myMat.cols;++j)	
					infile.read((char*)&p[j],sizeof(p[j]));
		}
	}
	if(CV_8UC3==sz[2])
	{
		for(int k=0;k<myMat.channels();++k)
			for(int i=0;i<myMat.rows;++i)
				for(int j=0;j<myMat.cols;++j)				
					infile.read((char*)&myMat.at<Vec3b>(i,j)[k],sizeof(myMat.at<Vec3b>(i,j)[k]));
	}
	if(CV_32SC1==sz[2])
	{
		for(int i=0;i<myMat.rows;++i)
		{
			int *p = myMat.ptr<int>(i);
			for(int j=0;j<myMat.cols;++j)	
					infile.read((char*)&p[j],sizeof(p[j]));
		}
	}	
	if(CV_32FC1==sz[2])
	{
		for(int i=0;i<myMat.rows;++i)
		{
			float *p = myMat.ptr<float>(i);
			for(int j=0;j<myMat.cols;++j)	
					infile.read((char*)&p[j],sizeof(p[j]));
		}
	}	
	infile.close();
	return myMat;
}




////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the Photo indexing Utility
///		written by Li, Xue
///   
/// Acknowledge     :
///
/// Function name   : writeMat2File
/// Description     : write the contents of the Mat into a .dat file. 
///
/// Argument        : filename -- name of the specified .dat file.
/// Argument        : data     -- the Mat to be saved.
///
/// Return type     : none.
///
/// Create Time     : 2014-12-5  12:48
///
///
/// Side Effect     : 
///
///////////////////////////////////////////////////////////////////////////////////////////////
void writeMat2File(string filename, Mat data)
{
	ofstream outfile;	
	outfile.open(filename.c_str(), ios_base::binary);
	if(!outfile)
	{
		cout<<"cannot open file!"<<endl;
		return;
	}
	int sz[3];
	sz[0] = int(data.rows);
	sz[1] = int(data.cols);
	sz[2] = int(data.type());
	
	outfile.write((char*)&sz,sizeof(sz));

	if(CV_8UC1==sz[2])
	{
		for(int i=0;i<data.rows;++i)
		{
			uchar *p = data.ptr<uchar>(i);
			for(int j=0;j<data.cols;++j)			
				outfile.write((char*)&p[j],sizeof(p[j]));
		}
	}
	if(CV_8UC3==sz[2])
	{
		for(int k=0;k<data.channels();++k)
			for(int i=0;i<data.rows;++i)
				for(int j=0;j<data.cols;++j)			
					outfile.write((char*)&data.at<Vec3b>(i,j)[k],sizeof(data.at<Vec3b>(i,j)[k]));	
	}
	if(CV_64FC1==sz[2])
	{
		for(int i=0;i<data.rows;++i)
		{
			double *p = data.ptr<double>(i);
			for(int j=0;j<data.cols;++j)			
				outfile.write((char*)&p[j],sizeof(p[j]));
		}
	}
	if(CV_64FC3==sz[2])
	{
		for(int k=0;k<data.channels();++k)
			for(int i=0;i<data.rows;++i)
				for(int j=0;j<data.cols;++j)			
					outfile.write((char*)&data.at<Vec3d>(i,j)[k],sizeof(data.at<Vec3d>(i,j)[k]));	
	}
	if(CV_32SC1==sz[2])
	{
		for(int i=0;i<data.rows;++i)
		{
			int *p = data.ptr<int>(i);
			for(int j=0;j<data.cols;++j)			
				outfile.write((char*)&p[j],sizeof(p[j]));
		}
	}
	if(CV_32FC1==sz[2])
	{
		for(int i=0;i<data.rows;++i)
		{
			float *p = data.ptr<float>(i);
			for(int j=0;j<data.cols;++j)			
				outfile.write((char*)&p[j],sizeof(p[j]));
		}
	}

	outfile.close();
}




////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the Photo indexing Utility
///		written by Li, Xue
///   
/// Acknowledge     :
///
/// Function name   : display_Mat
/// Description     : display the entries of a Mat, which is useful when test some of the functions. 
///
/// Argument        : data     -- the Mat to be displayed.
///
/// Return type     : none.
///
/// Create Time     : 2014-12-5  12:50
///
///
/// Side Effect     : 
///
///////////////////////////////////////////////////////////////////////////////////////////////
void display_Mat(Mat data)
{
	if(CV_8UC1==data.type())	
		for(int i=0;i<data.rows;++i)
		{
			uchar *p = data.ptr<uchar>(i);
			for(int j=0;j<data.cols;++j)
				cout<<int(p[j])<<" ";
			cout<<endl;
		}
		
	
	if(CV_8UC3==data.type())	
		for(int k=0;k<data.channels();++k)
		{
			cout<<"next:"<<endl;
			for(int i=0;i<data.rows;++i)
			{
				for(int j=0;j<data.cols;++j)
					cout<<int(data.at<Vec3b>(i,j)[k])<<" ";
				cout<<endl;
			}
		}
	
	if(CV_64FC1==data.type())	
		for(int i=0;i<data.rows;++i)
		{
			double *p = data.ptr<double>(i);
			for(int j=0;j<data.cols;++j)
				cout<<p[j]<<" ";
			cout<<endl;
		}
		
	
	if(CV_64FC3==data.type())	
		for(int k=0;k<data.channels();++k)
		{
			cout<<"next:"<<endl;
			for(int i=0;i<data.rows;++i)
			{
				for(int j=0;j<data.cols;++j)
					cout<<data.at<Vec3d>(i,j)[k]<<" ";
				cout<<endl;
			}
		}
	if(CV_32SC1 == data.type())
		for(int i=0;i<data.rows;++i)
		{
			int *p = data.ptr<int>(i);
			for(int j=0;j<data.cols;++j)
				cout<<p[j]<<" ";
			cout<<endl;
		}
	
}
