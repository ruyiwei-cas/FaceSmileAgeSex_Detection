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

/*!
*	@file		postmerge.hpp
*	@brief		inline functions for post-merge of detection results
*	@author		Jianguo Li, Intel Labs China
*	copyright reserved 2010, please do not remove this head
*/
#ifndef _POST_FILTER_HPP
#define _POST_FILTER_HPP

#include <math.h>
#include <vector>

#include "cxcore.h"

#include "det_types.hpp"

//////////////////////////////////////////////////////////////////////////
// disjoint-set forests using union-by-rank and path compression (sort of).
typedef struct{
	int rank;
	int p;
	int size;
}uni_elt;

class disjointset 
{
public:
	disjointset(int elements)
	{
		elts = new uni_elt[elements];
		num = elements;
		for (int i = 0; i < elements; i++) 
		{
			elts[i].rank = 0;
			elts[i].size = 1;
			elts[i].p = i;
		}
	}
	~disjointset()
	{
		delete [] elts;
	}

	int size(int x) const { return elts[x].size; }
	int num_sets() const { return num; }

	inline int find(int x)
	{
		int y = x;
		while (y != elts[y].p)
			y = elts[y].p;
		elts[x].p = y;
		return y;
	}
	inline void join(int x, int y)
	{
		if (elts[x].rank > elts[y].rank) 
		{
			elts[y].p = x;
			elts[x].size += elts[y].size;
		} 
		else 
		{
			elts[x].p = y;
			elts[y].size += elts[x].size;
			if (elts[x].rank == elts[y].rank)
				elts[y].rank++;
		}
		num--;
	}
private:
	uni_elt *elts;
	int num;
};

//////////////////////////////////////////////////////////////////////////
inline float computRectJoinUnion(CvRect rc1, CvRect rc2, float& AJoin, float& AUnion)
{
	CvPoint p1, p2;
	p1.x = MAX(rc1.x, rc2.x);
	p1.y = MAX(rc1.y, rc2.y);

	p2.x = MIN(rc1.x +rc1.width, rc2.x +rc2.width);
	p2.y = MIN(rc1.y +rc1.height, rc2.y +rc2.height);

	AJoin = 0;
	if( p2.x > p1.x && p2.y > p1.y )
	{
		AJoin = float(p2.x - p1.x)*(p2.y - p1.y);
	}
	float A1  = float(rc1.width * rc1.height);
	float A2  = float(rc2.width * rc2.height);
	AUnion = (A1 + A2 - AJoin);

	if( AUnion > 0 )
		return (AJoin/AUnion);
	else
		return 0;
}

// overlap ratio is the only tuning parameters
// around 0.6~0.7 = 0.7746^2 ~ 0.8367^2 are good choice
static inline int is_rect_equal(CvMVRect r1, CvMVRect r2, float overlapratio = 0.6f)
{
	float AJoin, AUnion;
	bool bequal = (r1.vid == r2.vid &&	  // view-id must be the same
		(r1.angle == r2.angle) &&		  // in-plane-angle of the rectangle must be the same
		(r1.stage == r2.stage) &&		  // stage must be the same
		computRectJoinUnion(r1.rc, r2.rc, AJoin, AUnion)>overlapratio ); // overlap ratio

	return bequal;
}

inline int icxRectFilter(std::vector<CvMVRect>& rcList, int min_neighbors=2, float overlapratio=0.7f, int roc = 0)
{
	int i, j;
	// S1: group (closed) rectangles using disjoint-set
	int n = (int)rcList.size();
	// min_neighbors = 0, do nothing and return
	if( n < 1 || min_neighbors < 1 )
		return n;

	disjointset *ds = new disjointset(n);
	for(i=0; i<n; ++i)
	{
		for(j=i+1; j<n; ++j)
		{
			int a = ds->find(i);
			int b = ds->find(j);
			if( a != b )
			{
				if( is_rect_equal(rcList[i], rcList[j], overlapratio) )
				{
					ds->join(a, b);
				}
			}
		}
	}
	int ncomp = ds->num_sets();

	int idx = 0;
   std::map<int,int> rcmap;
   std::map<int, int>::const_iterator mIt;
	for(i=0; i<n; ++i)
	{
		int a = ds->find(i);
      // a - key, idx - value
      mIt = rcmap.find(a);
      if(mIt==rcmap.end()) {
         rcmap.insert(std::pair<int, int>(a, idx));
         idx++;
      }
	}
	std::vector<CvAvgRect> comps(ncomp);
	memset(&comps[0], 0, sizeof(CvAvgRect)*ncomp);
	for(i=0; i<n; ++i)
	{
		CvRect r1 = rcList[i].rc;
		float prob = rcList[i].prob;
		int a = ds->find(i);

      mIt = rcmap.find(a);
      idx = mIt->second;

		comps[idx].vid = rcList[i].vid;
		comps[idx].angle = rcList[i].angle;
		comps[idx].stage = rcList[i].stage;
		comps[idx].neighbor++;
		// maximum is better than average
		comps[idx].prob = MAX(prob, comps[idx].prob);
		comps[idx].rc.x += r1.x;
		comps[idx].rc.y += r1.y;
		comps[idx].rc.width += r1.width;
		comps[idx].rc.height += r1.height;
	}
   rcmap.clear();

	// S2: merge grouped rectangles by average
	std::vector<CvAvgRect> seq2;
	seq2.reserve(ncomp);
	for(i = 0; i < ncomp; i++)
	{
		int kk = comps[i].neighbor;
		float deltaprob = MIN(MAX((kk - min_neighbors)/25.0f, -0.08f), 0.5f);
		// comps[i].prob /= kk;

		if( kk >= min_neighbors || roc > 0 )
		{
			CvAvgRect comp;
			int kk2 = 2*kk;
			// laplace estimation
			comp.rc.x = int((comps[i].rc.x*2 + kk +0.0)/kk2);
			comp.rc.y = int((comps[i].rc.y*2 + kk +0.0)/kk2);
			comp.rc.width = int((comps[i].rc.width*2 + kk +0.0)/kk2);
			comp.rc.height = int((comps[i].rc.height*2 + kk +0.0)/kk2);
			comp.neighbor = kk;
			comp.prob = comps[i].prob + deltaprob;
			comp.vid = comps[i].vid;
			comp.angle = comps[i].angle;
			comp.stage = comps[i].stage;

			seq2.push_back(comp);
		}
	}

	// S3: NMS (non-maximum supression) to filter largely-overlapped rects
	int sz = (int)seq2.size();
	std::vector<uchar> flag(sz);
	// flag = 0 for delete
	for(i=0; i <sz; i++)
		flag[i] = 1;

	for(i=0; i <sz; i++)
	{
		if( flag[i] == 0 )
			continue;

		CvAvgRect r1 = seq2[i];
		CvRect rc1 = r1.rc;
		float A1 = float(rc1.width * rc1.height);

		for(j=0; j <sz; j++)
		{
			if( i == j || flag[j] == 0 )
				continue;

			// check whether r1 inside r2, or r2 inside r1
			CvAvgRect r2 = seq2[j];
			CvRect rc2 = r2.rc;
			float A2 = float(rc2.width * rc2.height);

			float AJoin = 0, AUnion = 0;
			float aOverlap = computRectJoinUnion(rc1, rc2, AJoin, AUnion);
			float aRatio1 = AJoin/A1;
			float aRatio2 = AJoin/A2;

			// winner-take-all strategy for overlap-filter
			// replace previous area-ratio based, WTA is much better
			if( aRatio1 >= 0.5 || aRatio2 >= 0.5 || aOverlap > 0.499 )
			{
				if( r1.prob >= r2.prob )
				{
					flag[j] = 0;
					continue;
				}
				else
				{
					flag[i] = 0;
					break;
				}
			}
		} // end j
	}
	rcList.clear();
	for(i=0; i<sz; i++)
	{
		if( flag[i] == 1 )
		{
			CvMVRect arc;
			arc.rc = seq2[i].rc;
			arc.prob = seq2[i].prob;
			arc.vid = seq2[i].vid;
			arc.angle = seq2[i].angle;
			rcList.push_back( arc );
		}
	}

	delete ds;
	return (int)(rcList.size());
}

#endif
