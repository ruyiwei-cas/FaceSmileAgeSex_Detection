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

#include <stdlib.h>
#include <iostream>
#include <string>
#include <cmath>
#include <time.h>
#include <fstream>
#include <math.h>
#include <io.h>

#include <opencv2/opencv.hpp>

#include "compute_feature.h"
#include "transferData.h"
#include "constant.h"
#include "myhog.h"


using namespace cv;
using namespace std;




////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the Photo indexing Utility
///		written by Li, Xue
///   
/// Acknowledge     :
///
/// Function name   : generatePath
/// Description     : use this function when you need to access data 
///
/// Argument        : none
///
/// Return type     : struct path -- the structure with useful paths as fields, see compute_feature.h
///
/// Create Time     : 2014-12-2  15:55
///
///
/// Side Effect     : 
///
///////////////////////////////////////////////////////////////////////////////////////////////
struct path generatePath()
{
	struct path myPath;	

	myPath._path       = root_path + "\\scene_15class";
	myPath.vocab_path  = myPath._path + "\\vocabulary";
	myPath.img_path    = myPath._path + string("\\image");
	myPath.feat_path   = myPath._path + string("\\feature");
	myPath.result_path = myPath._path + string("\\result");
	myPath.kernel_path = myPath._path + string("\\kernel");	

	return myPath;
}




////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the Photo indexing Utility
///		written by Li, Xue
///   
/// Acknowledge     :
///
/// Function name   : get_class_names
/// Description     : as an initial step, load all the strings specifying category names  
///
/// Argument        : none
///
/// Return type     : string* -- a string array with each entry containing a category name 
///
/// Create Time     : 2014-12-2  16:02
///
///
/// Side Effect     : 
///
///////////////////////////////////////////////////////////////////////////////////////////////
string* get_class_names()
{
	struct path myPath1 = generatePath();
	string filename = myPath1._path + "\\class_name.txt";
	ifstream infile;
	char buffer[100];

	infile.open(filename.c_str(), ios_base::in);	

	string * class_name = new string[17];
	for(int i=0;i<NUM_CLASS_USE+1;++i)	
	{
		infile.getline(buffer,100);
		class_name[i] = string(buffer);		
	}	

	infile.close();
	return class_name;
}




////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the Photo indexing Utility
///		written by Li, Xue
///   
/// Acknowledge     :
///
/// Function name   : get_class_num
/// Description     : as an initial step, load the number of samples for each category
///
/// Argument        : none
///
/// Return type     : int* -- an array with each entry specifying the number of samples for the corresponding category 
///
/// Create Time     : 2014-12-2  16:05
///
///
/// Side Effect     : 
///
///////////////////////////////////////////////////////////////////////////////////////////////
int* get_class_num()
{
	struct path myPath1 = generatePath();
	string filename = myPath1._path + "\\class_num.dat";
	Mat num = loadFile2Mat(filename);
	int *class_num = new int[num.cols];
	int *p = num.ptr<int>(0);
	for(int i=0;i<num.cols;++i)
	{
		class_num[i] = p[i];
	}
	
	return class_num;
}




////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the Photo indexing Utility
///		written by Li, Xue
///   
/// Acknowledge     :
///
/// Function name   : extractFeatures_colorhist_v3
/// Description     : partition the image into 3 segments vertically, compute color histogram in HSV space 
///                   for each part, then concatenate them as the final result and return.
///
/// Argument        : src -- input RGB image
///
/// Return type     : Mat -- the obtained feature vector 
///
/// Create Time     : 2014-12-2  16:20
///
///
/// Side Effect     : 
///
///////////////////////////////////////////////////////////////////////////////////////////////
Mat extractFeatures_colorhist_v3(Mat src)
{
	Mat img;
	cvtColor(src,img,CV_BGR2HSV);		

	// compute hist
	int dims[3] = {8,8,8};
	int dim = dims[0]*dims[1]*dims[2];
	int step = dims[1]*dims[2];	
	
	double s0 = 180.0/double(dims[0]);
	double s1 = 256.0/double(dims[1]);
	double s2 = 256.0/double(dims[2]);	

	Mat hist(1,dim*3,CV_32FC1);// step * bin_l + dims[2] * bin_a + bin_b
	float *ph = hist.ptr<float>(0);
	hist.setTo(0);

	uchar *p;
	int bin_l, bin_a, bin_b;
	int dh = img.rows/3;
	int range[4]={0,dh,2*dh,img.rows};
	
	for(int l=0;l<3;++l)
	{		
		for(int i=range[l];i<range[l+1];++i)
		{
			p = img.ptr<uchar>(i);
			for(int j=0;j<img.cols;++j)
			{
				p += 3;
				bin_l = p[0]/s0;
				bin_a = p[1]/s1;
				bin_b = p[2]/s2;
				ph[dim * l+bin_l *step + bin_a * dims[2] + bin_b] ++;
			}
		}
		double t = 0;
		for(int i=dim*l;i<dim*(l+1);++i)		
			t += ph[i];
		
		t = 1.0/(t+EPS);
		for(int i=dim*l;i<dim*(l+1);++i)
			ph[i] = ph[i]*t;

	}

	/*for(int i=0;i<img.rows;++i)
	{
		p = img.ptr<uchar>(i);
		for(int j=0;j<img.cols;++j)
		{
			p += 3;
			bin_l = p[0]/s0;
			bin_a = p[1]/s1;
			bin_b = p[2]/s2;
			ph[dim * 3+bin_l *step + bin_a * dims[2] + bin_b] ++;
		}
	}
	double t = 0;
	for(int i=dim*3;i<dim*4;++i)		
		t += ph[i];		
		
	for(int i=dim*3;i<dim*4;++i)
		ph[i] = ph[i]/(t+EPS);*/

	return hist;
}




////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the Photo indexing Utility
///		written by Li, Xue
///   
/// Acknowledge     :
///
/// Function name   : extractFeatures_hog
/// Description     : extract hog2x2 feature for a single image
///
/// Argument        : img -- input RGB image
///
/// Return type     : Mat -- the obtained hog2x2 vector
///
/// Create Time     : 2014-12-2  16:26
///
///
/// Side Effect     : 
///
///////////////////////////////////////////////////////////////////////////////////////////////
Mat extractFeatures_hog(Mat img)
{		
	struct path myPath = generatePath();
	clock_t t1 = clock();	

	//t1 = clock();
	
	//cout<<"Time for resize = "<<clock()-t1<<"ms"<<endl;

	

	//extract hog2x2 descriptors	
	t1 = clock();
	Mat words = myhog2x2_(img);
	//Mat words = myhog2x2_flann(img);
	//cout<<"Time for hog2x2 = "<<clock()-t1<<"ms"<<endl;
			
	// Compute hists
	Mat hist;	
	//t1 = clock();
	hist = SPMpooling_(words);	
	//cout<<"time for pooling = "<<clock()-t1<<"ms"<<endl<<endl;

	return hist;			
}




////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the Photo indexing Utility
///		written by Li, Xue
///   
/// Acknowledge     :
///
/// Function name   : myhog2x2_
/// Description     : generate 124d (concatenating 4 vectors, each is 31-d) descriptor,
///                   and project into the nearest word.
///
/// Argument        : im -- input RGB image
///
/// Return type     : Mat -- the obtained words
///
/// Create Time     : 2014-12-2  16:30
///
///
/// Side Effect     : 
///
///////////////////////////////////////////////////////////////////////////////////////////////
Mat myhog2x2_(Mat im)
{		
	clock_t t1;
	int xmax,ymax,zmax;
	//struct feature feat;


	// Extract descriptors
	t1 = clock();
	Mat d = features_hog_1(im, INTERVAL);// d is 3-d matrix	
	//cout<<"features_hog = "<<clock()-t1<<"ms"<<endl;
	
	xmax = d.size[0];
	ymax = d.size[1];
	zmax = d.size[2];	

	Mat words(xmax-1,ymax-1,CV_32SC1);//int	
	Mat tmp(1,zmax*4,CV_8UC1);//uchar

	uchar * p_tmp = tmp.ptr<uchar>(0);
	double *pdd; 

	t1 = clock();
	int * p_w, step = xmax * ymax, s2 = 2*zmax, s3 = 3*zmax, s4 = 4*zmax;
	for(int xx=0;xx<xmax-1;++xx) //row
	{
		
		p_w = words.ptr<int>(xx);
		for(int yy=0;yy<ymax-1;++yy)// col
		{
			pdd = (double*)d.data;
			pdd += xx*ymax + yy; //locate to (xx,yy)

			for(int j=0;j<zmax;++j)
			{
				p_tmp[j] = uchar(*(pdd) * 255 + 0.5);
				pdd += step;					
			}

			pdd = (double*)d.data;
			pdd += (xx+1)*ymax + yy; //locate to (xx+1,yy)

			for(int j=zmax;j<s2;++j)
			{
				p_tmp[j] = uchar(*(pdd) *255+0.5);
				pdd += step;
			}

			pdd = (double*)d.data;
			pdd += xx*ymax + yy+1; //locate to (xx,yy+1)

			for(int j=s2;j<s3;++j)
			{
				p_tmp[j] = uchar(*(pdd) *255+0.5);
				pdd += step;
			}

			pdd = (double*)d.data;
			pdd += (xx+1)*ymax + yy+1; //locate to (xx+1,yy+1)

			for(int j=s3;j<s4;++j)
			{
				p_tmp[j] = uchar(*(pdd) *255+0.5);
				pdd += step;
			}			

			// project tmp into a word			
			p_w[yy] = computeWords_(tmp,Vocabulary);
			
		}
	}
	//cout<<"compute total words = "<<clock()-t1<<"ms"<<endl;

	return words;
}




////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the Photo indexing Utility
///		written by Li, Xue
///   
/// Acknowledge     :
///
/// Function name   : img_resize
/// Description     : resize the image so that its maximum size is maxSize.
///
/// Argument        : img     -- input RGB image
/// Argument        : maxSize -- pre-specified maximum size
///
/// Return type     : Mat -- the resized image
///
/// Create Time     : 2014-12-2  16:35
///
///
/// Side Effect     : 
///
///////////////////////////////////////////////////////////////////////////////////////////////
Mat img_resize(Mat img, int maxSize)
{
	int h,w;
	h = img.rows;
	w = img.cols;
	double fx = 1.0, fy = 1.0;
	if(h>=w&&h>maxSize)
	{
		fx = double(maxSize)/double(h);
	}
	if(w>h&&w>maxSize)
	{
		fx = double(maxSize)/double(w);
	}
	
	if(fx == 1 && fy == 1)
		return img;
	
	Mat dst;
	resize(img,dst,Size(),fx,fx,1);
	return dst;
}




////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the Photo indexing Utility
///		written by Li, Xue
///   
/// Acknowledge     :
///
/// Function name   : computeWords_
/// Description     : projects a descriptor into its nearest word.
///
/// Argument        : descrs -- input descriptor
/// Argument        : V      -- vocabulary, with each column specifying a word
///
/// Return type     : int -- the index of the nearest word
///
/// Create Time     : 2014-12-2  16:44
///
///
/// Side Effect     : 
///
///////////////////////////////////////////////////////////////////////////////////////////////
int computeWords_(Mat descrs,Mat V)//pass test; this is 0 based
{
	int n_v = V.rows;	
	bool f1;	
	double minVal = 256*256*256,val;
	int idx = 0;		
		
	uchar * p_d = descrs.ptr<uchar>(0);
	int * p_v,dif;
	for(int j=0;j<n_v;++j)
	{			
		p_v = V.ptr<int>(j);
		f1  = 0;			
		val = 0;
		for(int k=0;k<V.cols;++k)
		{
			dif = p_v[k] - p_d[k];
			val += (dif*dif);
			if(val>=minVal)
			{
				f1 = 1;
				break;
			}
		}
		if(f1)
			continue;
		if(val<minVal)
		{
			minVal = val;
			idx = j;
		}
	}			
	
	return idx;
}




////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the Photo indexing Utility
///		written by Li, Xue
///   
/// Acknowledge     :
///
/// Function name   : SPMpooling_
/// Description     : SPM pooling (average), with each layer split by 2, resulting in 21 (1+4+16) cells.
///
/// Argument        : words -- words of the image
///
/// Return type     : int -- the index of the nearest word
///
/// Create Time     : 2014-12-2  16:48
///
///
/// Side Effect     : 
///
///////////////////////////////////////////////////////////////////////////////////////////////
 Mat SPMpooling_(Mat words)//pass test
 { 	 	
	 int numWords = Vocabulary.rows; //300
	 int len = (int(pow(4.0,LAYERS))-1)/3;
	 
	 float *hist = new float[numWords*len];
	 for(int i=0;i<numWords*len;++i)
		 hist[i] = 0;

	 int c1 = 0,c2 = 0;	
	 int cnt = 0, numTiles, numTiles2, dx, dy, *bound_x, *bound_y, * pw;
	 for(int L=0;L<LAYERS;++L)
	 {
		 numTiles  = int(pow(2.0,L));
		 numTiles2 = numTiles * numTiles;
		
		 c1 = c2;
		 c2 = c1 + numTiles2 * numWords;

		 dx = words.rows/numTiles;
		 dy = words.cols/numTiles;

		 bound_x = new int[numTiles+1];
		 bound_y = new int[numTiles+1];

		 bound_x[0] = 0; bound_x[numTiles] = words.rows;
		 bound_y[0] = 0; bound_y[numTiles] = words.cols;

		 for(int i=1;i<numTiles;++i)
		 {
		 	 bound_x[i] = dx*i;
			 bound_y[i] = dy*i;
		 }		 
		 
		 for(int x=0;x<numTiles;++x)
		 {
			 for(int y=0;y<numTiles;++y)
			 {	
				 for(int i=bound_x[x];i<bound_x[x+1];++i)
				 {
					 pw = words.ptr<int>(i);
					 for(int j=bound_y[y];j<bound_y[y+1];++j)
					 {
						 hist[cnt+pw[j]]++;
					 }
				 }
				 
				 cnt += numWords;
			 }
		 }

		 // normalization for each layer
		 double total = 0;
		 for(int i=c1;i<c2;++i)
			 total += hist[i];
		 float tmp = 1.0 / (total + EPS);
		 for(int i=c1;i<c2;++i)
			 hist[i] *= tmp;

		 delete [] bound_x;
		 delete [] bound_y;
	 }	 

	 Mat output(1,numWords*len,CV_32FC1,hist);
	
	 return output;
 }