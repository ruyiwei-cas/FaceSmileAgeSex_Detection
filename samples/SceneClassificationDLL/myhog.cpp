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

#include <iostream>
#include <opencv2/opencv.hpp> 
#include "myhog.h"
#include "constant.h"
#include "transferData.h"

using namespace std;
using namespace cv;


////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the Photo indexing Utility
///		written by Li, Xue
///   
/// Acknowledge     :
///
/// Function name   : features_hog_1
/// Description     : This is a re-implementation of the hog2x2 feature, and returns a 3-d matrix.
///
/// Argument        : img -- the input image.
/// Argument        : sbin -- the patch sizes.
///
/// Return type     : Mat -- this is a (n1 x n2 x 31) 3-d matrix, where n1 = max(cell_h-2,0), 
///                   and n2 = max(cell_w-2,0).
///
/// Create Time     : 2014-12-5  11:14
///
///
/// Side Effect     : note that the value of sbin(denoted as s) and the minimal image size(denoted as w) 
///                   must satisfy the following inequality:
///
///                                w/(w/s-2) + 1 < w/4 + 0.5
///
///////////////////////////////////////////////////////////////////////////////////////////////
Mat features_hog_1(Mat img, int sbin)
{		
	// compute number of cells
	int cell_h = (double)img.rows/(double)sbin+0.5;
	int cell_w = (double)img.cols/(double)sbin+0.5;
	
	int n_cell = cell_h*cell_w;	

	// compute weighted hist, which contains n_cell subarrays, each has 18 entries; 
	double *w_hist = new double[n_cell*18];
	compute_weighted_hist(w_hist, img, sbin);
	

	// values for n_cell cells;	
	double *C = new double[n_cell]; 
	for(int i=0;i<n_cell;++i)
	{
		int ss = 18*i;
		C[i] = 0;	
		for(int j=0;j<9;++j)
		{
			double tmp = w_hist[ss+j]+w_hist[ss+j+9];
			C[i] += tmp*tmp;
		}		
	}	
	
	
	// compute a 31-d feature vector for each cell
	Mat output = compute_hog2x2_final(w_hist, cell_h, cell_w, C);
	

	delete [] w_hist;
	delete [] C;

	return output;
}




////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the Photo indexing Utility
///		written by Li, Xue
///   
/// Acknowledge     :
///
/// Function name   : compute_bin
/// Description     : given the pair of dx and dy, discretize the computed angle into an index netween 0 and 17.
///                   This index is used for accessing values in a subarray of the weighted hist, which has 18 entries. 
///
/// Argument        : dx -- the gradient component in x direction;
/// Argument        : dy -- the gradient component in y direction.
///
/// Return type     : int -- the index between 0 and 17.
///
/// Create Time     : 2014-12-5  10:31
///
///
/// Side Effect     : 
///
///////////////////////////////////////////////////////////////////////////////////////////////
int compute_bin(double dx, double dy)
{	
	double pi = 3.141592653;
	double s = pi/9;

	if(dx > -EPS && dx <EPS)
		dx = EPS;

	// compute angle
	double angle = atan(dy/(dx));		
	
	if(dx>0 && dy<0)			
		angle = 2*pi + angle;		
	
	if(dx<0)	
		angle = pi + angle;		

	int bin = angle/s+0.5;
	bin = bin%18;

	return bin;

}




////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the Photo indexing Utility
///		written by Li, Xue
///   
/// Acknowledge     :
///
/// Function name   : interpolation_bilinear
/// Description     : perform bi-linear interpolation for each pixel, thus it contributes to 4 adjacent cells.
///                   (including the current, the right, the below, and the bottom right cells, respectively.)
///
/// Argument        : hist   -- the weighted hist;
/// Argument        : cx     -- cell index of x direction for the current pixel;
/// Argument        : cy     -- cell index of y direction for the current pixel;
/// Argument        : cell_h -- total cell number along the y direction;
/// Argument        : cell_w -- total cell number along the x direction;
/// Argument        : val    -- energy (sqrt(dx^2 + dy^2)) of the current pixel;
/// Argument        : bin    -- bin index(0<=bin<=17) of the current pixel.
///
/// Return type     : none.
///
/// Create Time     : 2014-12-5  10:44
///
///
/// Side Effect     : 
///
///////////////////////////////////////////////////////////////////////////////////////////////
void interpolation_bilinear(double *hist, double cx, double cy, int cell_h, int cell_w, double val, int bin)
{		
	int cx_p = int(cx);
	int cy_p = int(cy);
	double a0 = cx-cx_p;
	double b0 = cy-cy_p;
	double a1 = 1.0-a0;
	double b1 = 1.0-b0;		

	int idx = 18*(cy_p*cell_w + cx_p) + bin;

	if (cx_p >= 0 && cy_p >= 0)   //(cy_p,cx_p)	
		hist[idx] += a1*b1*val;

	if (cx_p+1 < cell_w && cy_p >= 0)	//(cy_p,cx_p+1)		 
		hist[idx+18] += a0*b1*val;			 

	if (cx_p >= 0 && cy_p+1 < cell_h) 	//(cy_p+1,cx_p)		 
		hist[idx+18*cell_w] += a1*b0*val;			 

	if (cx_p+1 < cell_w && cy_p+1 < cell_h)   //(cy_p+1,cx_p+1)		 
		hist[idx+18*(cell_w + 1)] += a0*b0*val;

}




////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the Photo indexing Utility
///		written by Li, Xue
///   
/// Acknowledge     :
///
/// Function name   : compute_weighted_hist
/// Description     : compute weighted hist, which contains n_cell subarrays, each has 18 entries;
///                   (n_cell is the total number of cells for the current image.)
///                   main steps: for each picel, compute gradients, discretize the angle of the gradient, and perform 
///                   bi-linear interpolation to the adjacent cells.
///
/// Argument        : hist -- the weighted hist;
/// Argument        : img  -- the input image;
/// Argument        : sbin -- the patch sizes.
///
/// Return type     : double * -- hist is also an output.
///
/// Create Time     : 2014-12-5  10:29
///
///
/// Side Effect     : 
///
///////////////////////////////////////////////////////////////////////////////////////////////
void compute_weighted_hist(double *hist, Mat img, int sbin)
{
	int h = img.rows, w = img.cols, ch = img.channels();

	int cell_h = (double)img.rows/(double)sbin+0.5;
	int cell_w = (double)img.cols/(double)sbin+0.5;
	
	int n_cell = cell_h*cell_w;

	// pixels we can manipulate
	int range_h = cell_h*sbin;
	int range_w = cell_w*sbin;	

	for(int i=0;i<n_cell*18;++i)
		hist[i] = 0;

	for (int y = 1; y < range_h-1; y++)	
	{	
		uchar *im  = img.ptr<uchar>(min(y,h-2));   // the current row
		uchar *im1 = img.ptr<uchar>(min(y,h-2)-1); // the previous row
		uchar *im2 = img.ptr<uchar>(min(y,h-2)+1); // the next row

		for (int x = 1; x < range_w-1; x++) 
		{				
			// compute gradient using mask [-1,0,1]
			uchar *s  = im  + ch * min(x,w-2);
			uchar *s1 = im1 + ch * min(x,w-2);
			uchar *s2 = im2 + ch * min(x,w-2);

			// color channel 1
			double dx = double(*(s+ch) - *(s-ch));
			double dy = double(*(s2) - *(s1));
			double v = dx*dx + dy*dy;

			// color channel 2
			s++;s1++;s2++;			
			double dx2 = *(s+ch) - *(s-ch);
			double dy2 = *(s2) - *(s1);
			double v2 = dx2*dx2 + dy2*dy2;

			// color channel 3
			s++;s1++;s2++;
			double dx3 = *(s+ch) - *(s-ch);
			double dy3 = *(s2) - *(s1);			
			double v3 = dx3*dx3 + dy3*dy3;

			// select the channel with the largest magnitude
			if (v2 > v) 
			{
				v  = v2;
				dx = dx2;
				dy = dy2;
			 } 
			 if (v3 > v) 
			 {
				 v  = v3;
				 dx = dx3;
				 dy = dy3;
			 }

			 // 0 <= bin <= 17 
			 int bin = compute_bin(dx, dy);

			 // bilinear interpolation			
			 double cx = (double(x)+0.5)/(double)sbin-0.5;
			 double cy = (double(y)+0.5)/(double)sbin-0.5;
			 interpolation_bilinear(hist, cx, cy,cell_h, cell_w, sqrt(v), bin); 			
			 
		 }
    }	
}




////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the Photo indexing Utility
///		written by Li, Xue
///   
/// Acknowledge     :
///
/// Function name   : compute_hog2x2_final
/// Description     : given the weighted hist, compute the final 3-d matrix. 
///
/// Argument        : hist   -- the weighted hist;
/// Argument        : cell_h -- total cell number along the vertical direction;
/// Argument        : cell_w -- total cell number along the horizontal direction;
/// Argument        : C      -- an array has (cell_h x cell_w) entries.
///
/// Return type     : Mat -- this is a (n1 x n2 x 31) 3-d matrix, where n1 = max(cell_h-2,0), and n2 = max(cell_w-2,0).
///                   31 = 18 sensitive bins + 9 insensitive bins + 4 energy entries.
///
/// Create Time     : 2014-12-5  10:53
///
///
/// Side Effect     : 
///
///////////////////////////////////////////////////////////////////////////////////////////////
Mat compute_hog2x2_final(double *hist, int cell_h,int cell_w, double *C)
{
	int dim[3];
	dim[0] = max(cell_h-2,0);
	dim[1] = max(cell_w-2,0);
	dim[2] = 31;
	int step = dim[0]*dim[1];

	Mat output(3,dim,CV_64F);
	double *pp = (double*)output.data;
	for(int i=0;i<step*dim[2];++i)
		*(pp++) = 0;


	for (int y = 0; y < dim[0]; y++) 
	{
		for (int x = 0; x < dim[1]; x++) 
		{
		    double *p_feat = (double*)output.data + y*dim[1] + x; 
			
			int idx = (y+1)*cell_w + (x+1);
		    double n1 = 1.0 / sqrt(C[idx] + C[idx+1] + C[idx +cell_w] + C[idx+cell_w+1] + EPS);				
			idx = y*cell_w + (x+1);
			double n2 = 1.0 / sqrt(C[idx] + C[idx+1] + C[idx +cell_w] + C[idx+cell_w+1] + EPS);				
			idx = (y+1)*cell_w + x;
		    double n3 = 1.0 / sqrt(C[idx] + C[idx+1] + C[idx +cell_w] + C[idx+cell_w+1] + EPS);	   
			idx = y*cell_w + x;
		    double n4 = 1.0 / sqrt(C[idx] + C[idx+1] + C[idx +cell_w] + C[idx+cell_w+1] + EPS);

			// sum along 18 orientations
		    double s1 = 0; double s2 = 0; double s3 = 0; double s4 = 0;

		    // 18-d
			idx = 18*((y+1)*cell_w + (x+1));
		    for (int i = 0; i < 18; i++) 
		    {
				double tmp = hist[idx+i];			    

				double h1 = min(tmp * n1, 0.2);
				double h2 = min(tmp * n2, 0.2);
			    double h3 = min(tmp * n3, 0.2);
			    double h4 = min(tmp * n4, 0.2);

				*p_feat = min(1.0,0.5 * (h1 + h2 + h3 + h4));
				s1 += h1; s2 += h2;
			    s3 += h3; s4 += h4;
			    p_feat += step;
		    }

		    // 9-d            
		    for (int i = 0; i < 9; i++) 
		    {
			    double sum = hist[idx+i] + hist[idx+i+9];
			    double h1 = min(sum * n1, 0.2);
			    double h2 = min(sum * n2, 0.2);
			    double h3 = min(sum * n3, 0.2);
			    double h4 = min(sum * n4, 0.2);
                *p_feat = min(1.0,0.5 * (h1 + h2 + h3 + h4));
			    p_feat += step;
		    }

		    // 4-d
		    *p_feat = min(1.0,0.2357 * s1);   p_feat  += step;
		    *p_feat = min(1.0,0.2357 * s2);   p_feat  += step;
		    *p_feat = min(1.0,0.2357 * s3);   p_feat  += step;
		    *p_feat = min(1.0,0.2357 * s4);
		}
	}	

	return output;
}