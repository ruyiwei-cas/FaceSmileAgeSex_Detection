//FaceDetection.cpp: implementation of the FaceDetection class.
//
//////////////////////////////////////////////////////////////////////

#include <assert.h>
#include <opencv\cv.h>

#ifdef WIN32
	#include <windows.h>
#else
	#include <sys/time.h>
#endif

#include "FaceDetection.h"

#ifndef WIN32
	#define nullptr NULL
#endif

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////
//int xstep_array[21] = {0, 0,  2,  2,   2,   2,   2,   3,   3,   3,   4,    4,    4,   5,   5,   5,   6,   6,   6,   6,    7};
//int ystep_array[21] = {0, 0,  1,  1,   1,   2,   2,   2,   2,   2,   3,    3,    3,   3,   4,   4,   4,   4,   6,   6,    7};
int xstep_array[21] = {0, 0,  2,  2,   2,   2,   3,   4,   4,   4,   5,    5,    5,   6,   6,   6,   7,   7,   7,   7,    8};
int ystep_array[21] = {0, 0,  1,  1,   1,   2,   2,   2,   3,   3,   3,    3,    4,   4,   4,   5,   5,   5,   6,   6,    7};

#define max(a,b)    (((a) > (b)) ? (a) : (b))
#define min(a,b)    (((a) < (b)) ? (a) : (b))



void Get_xystep(int nFlag, int nWinSize, int *xstep, int *ystep)
{
	if(nFlag==0)
	{
		*xstep = 2;
		*ystep = 1;
		return ;
	}

	int nNum = nWinSize/10;
	if(nNum<0) nNum = 0;
	if(nNum>30)
	{
		*xstep = 15;
		*ystep = 10;
		return ;
	}
	if(nNum>20) nNum = 20;
	*xstep = xstep_array[nNum];
	*ystep = ystep_array[nNum];
}

void DataTransfer(CComp * Src, CComp *Des)
{
	Des->rect = Src->rect;
	Des->eyeLocation = Src->eyeLocation;
	Des->confidence = Src->confidence;
	Des->neighbors = Src->neighbors;
	Des->view = Src->view;
}


__inline int Round( double val )
{
//    double temp = val + 6755399441055744.0;
//    return (int)*((__int64*)&temp);
	return (int)(val+.5);
}

__inline int is_equal(FdRect *r1, FdRect *r2)
{
	int distance = Round(r1->width * 0.2);
	
    return r2->x <= r1->x + distance &&
		   r2->x >= r1->x - distance &&
		   r2->y <= r1->y + distance &&
		   r2->y >= r1->y - distance &&
		   r2->width <= Round( r1->width * 1.2) &&	
		   Round(r2->width * 1.2) >= r1->width;		
}

__inline int is_included(FdRect r, CPoint_Ren p) 
{
    return r.x <= p.x && r.y <=p.y && r.x + r.width >= p.x && r.y + r.height >= p.y;		
}

__inline int is_folded(FdRect r1, FdRect r2) 
{
	CPoint_Ren p;
	p.x = r1.x; p.y = r1.y;
	if ( is_included(r2, p) )
		return 1;
	else
	{
		p.x = r1.x + r1.width; p.y = r1.y;
		if ( is_included(r2, p) )
			return 1;
		else
		{
			p.x = r1.x; p.y = r1.y + r1.height;
			if ( is_included(r2, p) )
				return 1;
			else
			{
				p.x = r1.x + r1.width; p.y = r1.y+r1.height;
				if ( is_included(r2, p) )
					return 1;
				else
					return 0;
			}
		}
	}
}


// if r1 is inside of r2
__inline bool is_inside(FdRect* r1, FdRect *r2) 
{
	if( r1->x < r2->x)
		return false;

	if( r1->y < r2->y)
		return false;

	if( r1->x+r1->width > r2->x +r2->width)
		return false;

	if( r1->y + r1->height > r2->y+r2->height)
		return false;

	return true;
}

bool Overlap_Equal(FdRect* r1, FdRect *r2, double dError = 0.5)
{
/// Mayong's algorithm
/// the overlap area > dError*detected face area
/// the overlap area > dError*groundtruth face area

	/// get the common area
	int nCommon_lx, nCommon_ly, nCommon_rx, nCommon_ry;
	nCommon_lx = max(r1->x, r2->x);
	nCommon_ly = max(r1->y, r2->y);
	nCommon_rx = min(r1->x + r1->width, r2->x+r2->width);
	nCommon_ry = min(r1->y + r1->height, r2->y+r2->height);
	if((nCommon_lx>nCommon_rx) || (nCommon_ly > nCommon_ry))
		return false;
	double nCommon_Area = (nCommon_rx-nCommon_lx)*(nCommon_ry - nCommon_ly);
///	if(nCommon_Area < dError * r1.width*r1.height) return 0;
///	if(nCommon_Area < dError * r2.width*r2.height) return 0;
	double f1 = nCommon_Area / (r1->width*r1->height);
	//if(f1<dError) return false;
	if (f1 > dError) return true;
	double f2 = nCommon_Area / (r2->width*r2->height);
	//if(f2<dError) return false;
	if (f2 > dError) return true;
	if(f1+f2<1.0) return false;
	else return true;
}

bool Overlap(FdRect* r1, FdRect *r2, double dScale = 0.5)
{
/// Mayong's algorithm
/// the overlap area > dError*detected face area
/// the overlap area > dError*groundtruth face area

	/// get the common area
	int nCommon_lx, nCommon_ly, nCommon_rx, nCommon_ry;
	nCommon_lx = max(r1->x, r2->x);
	nCommon_ly = max(r1->y, r2->y);
	nCommon_rx = min(r1->x + r1->width, r2->x+r2->width);
	nCommon_ry = min(r1->y + r1->height, r2->y+r2->height);
	if((nCommon_lx>nCommon_rx) || (nCommon_ly > nCommon_ry))
		return false;
	double nCommon_Area = (nCommon_rx-nCommon_lx)*(nCommon_ry - nCommon_ly);
///	if(nCommon_Area < dError * r1.width*r1.height) return 0;
///	if(nCommon_Area < dError * r2.width*r2.height) return 0;
	double f1 = nCommon_Area / (r1->width*r1->height);
	if(f1 > dScale) return true;
	double f2 = nCommon_Area / (r2->width*r2->height);
	if(f2 > dScale) return true;
	return false;
}
/*
CComp::CComp()
{
	eyeLocation = 0;
	neighbors = 0; 
	rect.height = 0;
	rect.width = 0;
	rect.x = 0;
	rect.y = 0;
}

CComp::~CComp()
{
}
*/
FaceDetection::FaceDetection(int nWinWidth, int nWinHeight)
{
	s = nullptr;
	sum_image = nullptr;
	tilt_image = nullptr;
	sqsum_image = nullptr;
	sq = nullptr;
	m_nImageWidth = m_nImageHeight = m_nImageScale = 0;
	m_lpImage = nullptr;
	m_nWinWidth = 0;
	m_nWinHeight = 0;
	m_dWinScale = 0.0f;

	m_bFaceSizeRange_Set = false;
	m_bROI_Set = false;
	Init(nWinWidth, nWinHeight, 1.0);	

	//m_FaceDepthImage = new CFaceDepthImage();

	m_nTotalWinNum = 0;
	m_nAdaCompute_Num =0;
}

 FaceDetection::~ FaceDetection()
{
	Release();
	if(m_CascadeClassifier != nullptr)
	{
		delete m_CascadeClassifier;
		m_CascadeClassifier = nullptr;
	}
}
void FaceDetection::Init(int nWinWidth, int nWinHeight, double dWinScale)
{
	m_nWinWidth = nWinWidth;
	m_nWinHeight = nWinHeight;
	m_dWinScale = dWinScale;
	m_CascadeClassifier = new HvChainCascadeClassifier(m_nWinWidth, m_nWinHeight);
	assert(m_CascadeClassifier);
}

////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of theFaceDetection Class
///  written by Ren Haibing 
/// Acknowledge		: None
///
/// Function name	: SetParameter
/// Description	    : 
///
/// Argument		: (width, height) -- image size
/// Argument		: pic_scale  -- image scale
///
/// Return type		:  the number
///
/// Create Time		: 2005-12-27  16:14
///
///
/// Side Effect		: 
///
///////////////////////////////////////////////////////////////////////////////////////////////
void FaceDetection::SetParameter(int width, int height, int pic_scale)
{

	m_nImageHeight = height;
	m_nImageWidth = width;
	m_nImageScale = pic_scale;
	
	Release();
	MemoryAllocation();
}

void FaceDetection::MemoryAllocation()
{
    //if (!m_lpImage)
	//    m_lpImage = (BYTE *)malloc(m_nImageWidth* m_nImageHeight);
	//ASSERT(m_lpImage);

	orisize = fdSize(m_nImageWidth, m_nImageHeight);
	size = fdSize(m_nImageWidth/m_nImageScale, m_nImageHeight/m_nImageScale);
	sumsize = fdSize(size.width + 1, size.height + 1 );
	size_space = size.width * size.height;
	sumsize_space = sumsize.width * sumsize.height;

	s = (int *)calloc(sizeof(int), size_space);
	memset(s, 0, sizeof(int)*size_space);

	sum_image = (int *)calloc(sizeof(int), sumsize_space);
	memset(sum_image, 0, sizeof(int)*sumsize_space);

	tilt_image = (int *)calloc(sizeof(int), sumsize_space);
	memset(tilt_image, 0, sizeof(int)*sumsize_space);

	sqsum_image = (double *)calloc(sizeof(double), sumsize_space);
	memset(sqsum_image, 0, sizeof(double)*sumsize_space);

	sq = (int *)calloc(sizeof(int), size_space);
	memset(sq, 0, sizeof(int)*size_space);
}

void FaceDetection::Release()
{
    /*
	if(m_lpImage != NULL)
	{
		free(m_lpImage);
		m_lpImage = NULL;
	}
    */
    
	if(s != NULL) 
	{
		free(s);
		s = NULL;
	}
	if(sum_image != NULL) 
	{
		free(sum_image);
		sum_image = NULL;
	}
	if(tilt_image != NULL) 
	{
		free(tilt_image);
		tilt_image = NULL;
	}
	if(sqsum_image != NULL) 
	{
		free(sqsum_image);
		sqsum_image = NULL;
	}
	if(sq != NULL) 
	{
		free(sq);
		sq = NULL;
	}
}

////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the FaceDetection Class
///  written by Ren Haibing 
/// Acknowledge		: None
///
/// Function name	: Cal_IntegralImage
/// Description	    : calculate the integral image
///
///
/// Return type		:  
///
/// Create Time		: 2005-12-27  16:14
///
///
/// Side Effect		: 
///
///////////////////////////////////////////////////////////////////////////////////////////////
void FaceDetection::Cal_IntegralImage()
{
 /*   
	CvMat img;
    CvMat sum_array;
    CvMat tilted_array;
    CvMat sqsum_array;

	img = cvMat(m_nImageHeight, m_nImageWidth, CV_8UC1,  m_lpImage);

	sum_array = cvMat(m_nImageHeight+1, m_nImageWidth+1, CV_32SC1,  sum_image);
	sqsum_array = cvMat(m_nImageHeight+1, m_nImageWidth+1, CV_64FC1,  sqsum_image);
	tilted_array = cvMat(m_nImageHeight+1, m_nImageWidth+1, CV_32SC1,  tilt_image);
	
   cvIntegral( &img, &sum_array, &sqsum_array, &tilted_array );  // Integral Image.
*/
#ifndef WIN32
	#define BYTE unsigned char
#endif
	BYTE *lpImage = m_lpImage;
	int i, j;

	// sum, sqsum
	s[0] = 0;
	sq[0] = 0;
	for (j = 0; j < size.height; j++){
		for (i = 0; i < size.width; i++){
			s[i + 1] = s[i] + lpImage[j*size.width + i];
			sum_image[((j + 1)*sumsize.width) + (i + 1)] = sum_image[j*sumsize.width + (i + 1)] + s[i + 1];
			sq[i + 1] = sq[i] + lpImage[j*size.width + i] * lpImage[j*size.width + i];
			sqsum_image[((j + 1)*sumsize.width) + (i + 1)] = sqsum_image[j*sumsize.width + (i + 1)] + sq[i + 1];
		}
	}
// tilted
	short r, l, distance;
	int nTemp = m_nImageHeight + m_nImageWidth - 1;
	int *tr = new int[nTemp];
	int *tl = new int[nTemp];
	memset(tr, 0, nTemp*sizeof(int));
	memset(tl, 0, nTemp*sizeof(int));

	memset(tilt_image, 0, sizeof(int)*sumsize_space);
			
	distance = m_nImageHeight - 1;
	for(j=0 ; j<size.height ; j++)
	{
		r = j;		l = distance - r;
		tilt_image[(j+1)*sumsize.width] = tilt_image[j*sumsize.width+1];
		for(i=0 ; i<size.width ; i++)
		{
			tilt_image[(j+1)*sumsize.width+(i+1)] = tilt_image[j*sumsize.width+(i+1)] + lpImage[j*size.width+i] + tr[r] + tl[l];
			tr[r] += lpImage[j*size.width+i];
			tl[l] += lpImage[j*size.width+i];
			r++;	l++;
		}
	}
	delete tr;
	delete tl;
}

void FaceDetection::Cal_IntegralImage_one()
{
#ifndef WIN32
#define BYTE unsigned char
#endif
	BYTE *lpImage = m_lpImage;
	register int i, j;

	// sum, sqsum
	s[0] = 0;
	sq[0] = 0;
	for (j = 0; j < size.height; j++){
		register int j_sumsize = j *sumsize.width;
		register int j_size = j *size.width;

		for (i = 0; i < size.width; i++){
			s[i + 1] = s[i] + lpImage[j_size + i];
			sum_image[j_sumsize + sumsize.width + (i + 1)] = sum_image[j_sumsize + (i + 1)] + s[i + 1];
			sq[i + 1] = sq[i] + lpImage[j*size.width + i] * lpImage[j_size + i];
			sqsum_image[j_sumsize + sumsize.width + (i + 1)] = sqsum_image[j_sumsize + (i + 1)] + sq[i + 1];
		}
	}
}
void FaceDetection::Cal_IntegralImage_two()
{
#ifndef WIN32
#define BYTE unsigned char
#endif
	BYTE *lpImage = m_lpImage;
	register int i, j;
	// tilted
	short r, l, distance;
	int nTemp = m_nImageHeight + m_nImageWidth - 1;
	int *tr = new int[nTemp];
	int *tl = new int[nTemp];
	memset(tr, 0, nTemp*sizeof(int));
	memset(tl, 0, nTemp*sizeof(int));

	memset(tilt_image, 0, sizeof(int)*sumsize_space);

	distance = m_nImageHeight - 1;
	for (j = 0; j<size.height; j++)
	{
		r = j;		
		l = distance - r;
		register int j_sumsize = j * sumsize.width;
		register int j_size = j * size.width;

		tilt_image[j_sumsize + sumsize.width] = tilt_image[j_sumsize + 1];
		for (i = 0; i<size.width; i++)
		{
			tilt_image[j_sumsize + sumsize.width + (i + 1)] = tilt_image[j_sumsize + (i + 1)] + lpImage[j_size + i] + tr[r] + tl[l];
			tr[r] += lpImage[j_size + i];
			tl[l] += lpImage[j_size + i];
			r++;	l++;
		}
	}
	delete tr;
	delete tl;
}

////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the FaceDetection Class
///  written by Ren Haibing 
/// Acknowledge		: None
///
/// Function name	: DetectFace
/// Description	    : detect the face image
///
/// Argument		:  lpImage -- image buffer
/// Argument		:  lpDepthImage -- depth image buffer
/// Argument		:  faces -- face result
///
/// Return type		:  the face number
///
/// Create Time		: 2005-12-27  16:40
///
///
/// Side Effect		: 
///
///////////////////////////////////////////////////////////////////////////////////////////////

int FaceDetection::DetectFace(IplImage *lpImage, unsigned short* lpDepthImage, FdAvgComp *faces)
{
	//if(lpDepthImage != nullptr){
	//	m_FaceDepthImage->Init(lpDepthImage, m_nImageWidth, m_nImageHeight);
	//	this->lpDepthImage = lpDepthImage;
	//}
	//int nOffset_Des = 0;
	int nSrcWidth=m_nImageWidth;
	nSrcWidth += (nSrcWidth%4==0 ? 0 : 4-nSrcWidth%4);

        m_lpImage = (uchar*)lpImage->imageData;


/* add on 2015-5-18 */
#ifdef _DEBUG_TIME
	tick_count t_start = tick_count::now();
#endif
#ifdef _PARALLEL_INTEGRAL_IMAGE_
//	cout<<"[+] parallel..."<<endl;
	task_group tg;
	tg.run([&]{Cal_IntegralImage_two(); });
	tg.run([&]{Cal_IntegralImage_one(); });
	tg.wait();
#else
	Cal_IntegralImage();
#endif
#ifdef _DEBUG_TIME
	tick_count t_end = tick_count::now();
	cout << "[+] Cal_IntegralImage() spend time:: " << (t_end - t_start).seconds() * 1000 << " milliseconds" << endl;
#endif

	int idx = 0;
	int ystep = 1;
	int xstep = 2;

	int limit_win = min (size.width, size.height);
	FdSize winsize = fdSize(m_nWinWidth, m_nWinHeight);


#ifdef _DEBUG_TIME
	tick_count t0 = tick_count::now();
#endif

//#ifdef _PARALLEL_TASK_GROUP_
//	/////////////////////////////task_group version////////////////
//	cout << "[+] <using task_group...>" << endl;
//	cout << "[+] cpu cores: " << get_cpu_core_num() << endl;
//	int thread_num = get_cpu_core_num() - 1;////set thread num = core_num - 1
//
//	vector< vector<dc_param> > params(thread_num);
//	
//	int temp;
//	int i_counter = 0;
//	while(winsize.width <= limit_win)
//	{
//		if(((m_bFaceSizeRange_Set) && (winsize.width>=m_nMinSize)&&(winsize.width<=m_nMaxSize))
//			|| (!m_bFaceSizeRange_Set))
//		{
//			Get_xystep(1, winsize.height, &xstep, &ystep);
//			ystep = max(2, ystep);
//			params[(i_counter++) % thread_num].push_back(dc_param(winsize, xstep, ystep));
//		}
//		temp = int(m_dDetectionScale * winsize.width + 0.5);
//		winsize = fdSize(temp, temp);
//	}
//	/*
//	for (int i = 0; i < thread_num; i++){
//			cout << "params[i].size=" << params[i].size() << endl;
//	}
//	
//	vector<CComp *> vc;
//	DFTask& root = *new(task::allocate_root()) DFTask(this, params, m_nTotalWinNum, m_nAdaCompute_Num, vc);
//	task::spawn_root_and_wait(root);
//
//	m_nSeq_Candidate_Num = vc.size();
//	for (size_t i = 0; i < vc.size(); i++){
//		m_Seq[i] = vc[i];
//	}
//	*/
//
//	task_group					g;
//	vector<int >				v_wn(thread_num + 1);
//	vector<int >				v_cn(thread_num + 1);
//	vector<vector<CComp *> >	v_seq(thread_num + 1);
//	vector<CComp *>				vc;
//
//	for (int i = 0; i < thread_num; i++){
//		//use vp to deliver params[i][]
//		vector<dc_param> &vp		= params[i];
//		vector<CComp *>  &rv_seq	= v_seq[i];
//		g.run([&]()mutable{
//				DFTask *th = new DFTask(this, vp, v_wn[i], v_cn[i], rv_seq); 
//				th->run();
//		});
//	}
//
//	g.wait();
//	cout<<"task_group end, begin merging"<<endl;
//	size_t k = 0;
//	this->m_nTotalWinNum = 0;
//	this->m_nAdaCompute_Num = 0;
//	this->m_nSeq_Candidate_Num = 0;
//	for (size_t j = 0; j < thread_num; j++){
//		this->m_nTotalWinNum += v_wn[j];
//		this->m_nAdaCompute_Num += v_cn[j];
//		this->m_nSeq_Candidate_Num += v_seq[j].size();
//
//		for (size_t i = 0; i < v_seq[j].size(); i++){
//			this->m_Seq[k++] = v_seq[j][i];
//		}
//	}
//	this->m_nSeq_Candidate_Num = k - 1;

#ifdef _PARALLEL_DETECT_CANDIDATE_
	/////////////////////////////parallel version////////////////
//	cout<<"[+] <parallel version...>"<<endl;

	vector<dc_param> params;
	int temp;
	while(winsize.width <= limit_win)
	{
		if(((m_bFaceSizeRange_Set) && (winsize.width>=m_nMinSize)&&(winsize.width<=m_nMaxSize))
			|| (!m_bFaceSizeRange_Set))
		{
			Get_xystep(1, winsize.height, &xstep, &ystep);
			ystep = max(2, ystep);
			params.push_back(dc_param(winsize, xstep, ystep));
			//cout<<"winsize:"<<winsize.width<<"\t xstep:"<<xstep<<"\tystep:"<<ystep<<endl;
		}
		temp = int(m_dDetectionScale * winsize.width + 0.5);
		winsize = fdSize(temp, temp);
	}

	//parallel_reduce
	DFBody dfb(this, params);
	//parallel_reduce(blocked_range<size_t>(0, params.size()), dfb, simple_partitioner());
	parallel_reduce(blocked_range<size_t>(0, params.size()), dfb);//, affinity_partitioner());

	m_nTotalWinNum = dfb.m_nTotalWinNum;
	m_nAdaCompute_Num = dfb.m_nAdaCompute_Num;
	m_nSeq_Candidate_Num = dfb.m_Seq.size();

	for (size_t i = 0; i<dfb.m_Seq.size(); i++){
		m_Seq[i] = dfb.m_Seq[i];
	}
	
#else
	/////////////////////////////serial version////////////////
	m_nSeq_Candidate_Num = 0;

	while(winsize.width <= limit_win)
	{
		//cout<<"winsize.width="<<winsize.width<<"\tlimit_win="<<limit_win<<endl;
		if(((m_bFaceSizeRange_Set) && (winsize.width>=m_nMinSize)&&(winsize.width<=m_nMaxSize))
			|| (!m_bFaceSizeRange_Set))
			// face size range
		{
			Get_xystep(1, winsize.height, &xstep, &ystep);
			ystep = max(2, ystep);
			DetectCandidates(winsize, ystep, xstep);
		}
		winsize = fdSize(int( m_dDetectionScale * winsize.width+0.5), int( m_dDetectionScale * winsize.height+0.5));
	}

#endif
#ifdef _DEBUG_TIME
	tick_count t1 = tick_count::now();
	cout << "[+] DetectCandidates() spend time:: " << (t1-t0).seconds()*1000 << " milliseconds" << endl;
#endif
	//idx is the number of face
	idx = Ren_MergeCandidates(faces, 1); 

	return idx;
}
////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the FaceDetection Class
///  written by Ren Haibing 
/// Acknowledge		: None
///
/// Function name	: DetectFace
/// Description	    : detect the face image
///
/// Argument		:  lpImage -- image buffer
/// Argument		:  lpDepthImage -- depth image buffer
/// Argument		:  faces -- face result
/// Argument		:  parallel_flag ---control parallel or serial
///
/// Return type		:  the face number
///
/// Create Time		: 2005-12-27  16:40
///
///
/// Side Effect		: 
///
///////////////////////////////////////////////////////////////////////////////////////////////
int FaceDetection::DetectFace(IplImage *lpImage, unsigned char* lpSkinImage, FdAvgComp *faces, int parallel_flag)
{
	if (lpSkinImage != nullptr){
		m_FaceSkinImage.Init(lpSkinImage, m_nImageWidth, m_nImageHeight);
		m_bWithSkinColor = true;
	}
	else m_bWithSkinColor = false;
	//int nOffset_Des = 0;
	int nSrcWidth = m_nImageWidth;
	nSrcWidth += (nSrcWidth % 4 == 0 ? 0 : 4 - nSrcWidth % 4);

	m_lpImage = (uchar*)lpImage->imageData;

	int idx = 0;
	int ystep = 1;
	int xstep = 2;

	int limit_win = min(size.width, size.height);
	FdSize winsize = fdSize(m_nWinWidth, m_nWinHeight);

	Cal_IntegralImage();
	m_nSeq_Candidate_Num = 0;

	while (winsize.width <= limit_win)
	{
		//cout<<"winsize.width="<<winsize.width<<"\tlimit_win="<<limit_win<<endl;
		if (((m_bFaceSizeRange_Set) && (winsize.width >= m_nMinSize) && (winsize.width <= m_nMaxSize))
			|| (!m_bFaceSizeRange_Set))
			// face size range
		{
			Get_xystep(1, winsize.height, &xstep, &ystep);
			ystep = max(2, ystep);
			DetectCandidates(winsize, ystep, xstep);
		}
		winsize = fdSize(int(m_dDetectionScale * winsize.width + 0.5), int(m_dDetectionScale * winsize.height + 0.5));
	}

	//idx is the number of face
	idx = Ren_MergeCandidates(faces, 1);

	return idx;
}

////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the FaceDetection Class
///  written by Ren Haibing 
/// Acknowledge		: None
///
/// Function name	: DetectFace
/// Description	    : detect the face image
///
/// Argument		:  lpImage -- image buffer
/// Argument		:  lpDepthImage -- depth image buffer
/// Argument		:  faces -- face result
/// Argument		:  parallel_flag ---control parallel or serial
///
/// Return type		:  the face number
///
/// Create Time		: 2005-12-27  16:40
///
///
/// Side Effect		: 
///
///////////////////////////////////////////////////////////////////////////////////////////////
int FaceDetection::DetectFace_Partial(IplImage *lpImage, unsigned char* lpSkinImage, FdAvgComp *faces,
	int nTotal_Part_Num, int nCurrentPartNo)
{
	if (lpSkinImage != nullptr){
		m_FaceSkinImage.Init(lpSkinImage, m_nImageWidth, m_nImageHeight);
		m_bWithSkinColor = true;
	}
	else m_bWithSkinColor = false;
	//int nOffset_Des = 0;
	int nSrcWidth = m_nImageWidth;
	nSrcWidth += (nSrcWidth % 4 == 0 ? 0 : 4 - nSrcWidth % 4);

	m_lpImage = (uchar*)lpImage->imageData;

	int idx = 0;
	int ystep = 1;
	int xstep = 2;

	int limit_win = min(size.width, size.height);
	FdSize winsize = fdSize(m_nWinWidth, m_nWinHeight);

	Cal_IntegralImage();
	m_nSeq_Candidate_Num = 0;

	int nPartNo = -1;
	while (winsize.width <= limit_win)
	{
		nPartNo++;

		int nFlag = nPartNo % nTotal_Part_Num;
		if (nFlag != nCurrentPartNo)
			continue;
		//cout<<"winsize.width="<<winsize.width<<"\tlimit_win="<<limit_win<<endl;
		if (((m_bFaceSizeRange_Set) && (winsize.width >= m_nMinSize) && (winsize.width <= m_nMaxSize))
			|| (!m_bFaceSizeRange_Set))
			// face size range
		{
			Get_xystep(1, winsize.height, &xstep, &ystep);
			ystep = max(2, ystep);
			DetectCandidates(winsize, ystep, xstep);
		}
		winsize = fdSize(int(m_dDetectionScale * winsize.width + 0.5), int(m_dDetectionScale * winsize.height + 0.5));
	}

	//idx is the number of face
	idx = Ren_MergeCandidates(faces, 1);
	return idx;
}

int FaceDetection::DetectFacePyramid(IplImage *lpImage, FdAvgComp *faces)
{
	m_bWithSkinColor = false;
	m_nSeq_Candidate_Num = 0;

	int level, max_level = 27;
	int idx = 0;
	double factor = 1.0;
	double ystep = 1.0;
	double xstep = 2.0;
	if (m_lpImage)
		free(m_lpImage);

	int nTotalNum = 0;

	FdSize winsize = fdSize(m_nWinWidth, m_nWinHeight);
	IplImage* curImage = cvCloneImage(lpImage);

	m_lpImage = NULL;

	char text[200];
	int nCurrentFaceSize;
	int nCurrentImage_Width, nCurrentImage_Height;
	nCurrentImage_Height = curImage->height;
	nCurrentImage_Width = curImage->width;

	for (level = 0; level < max_level; ++level)
	{
		factor = lpImage->width *1.0 / nCurrentImage_Width;
		nCurrentFaceSize = winsize.width *factor;
		if (((m_bFaceSizeRange_Set) && (nCurrentFaceSize >= m_nMinSize) && (nCurrentFaceSize <= m_nMaxSize))
			|| (!m_bFaceSizeRange_Set))
		{
			SetParameter(nCurrentImage_Width, nCurrentImage_Height, 1);
			m_lpImage = (uchar*)curImage->imageData;
			Cal_IntegralImage();

			SetImageROI(0.8);
			nTotalNum += DetectCandidates(winsize, ystep, xstep, factor);
		}

		// prepar for next level
		nCurrentImage_Width = (int)(nCurrentImage_Width / m_dDetectionScale);
		nCurrentImage_Height = (int)(nCurrentImage_Height / m_dDetectionScale);

		if (nCurrentImage_Width < winsize.width || nCurrentImage_Height < winsize.height)
			break;
		if (nCurrentImage_Width % 4 != 0)
		{
			nCurrentImage_Width += 4 - (nCurrentImage_Width % 4);
			nCurrentImage_Height = (curImage->height * nCurrentImage_Width / m_nImageWidth);
		}
		factor = lpImage->width *1.0 / nCurrentImage_Width;
		nCurrentFaceSize = winsize.width *factor;

		if (((m_bFaceSizeRange_Set) && (nCurrentFaceSize >= m_nMinSize) && (nCurrentFaceSize <= m_nMaxSize))
			|| (!m_bFaceSizeRange_Set))
		{
			IplImage* pImageNew = cvCreateImage(cvSize(nCurrentImage_Width, nCurrentImage_Height), curImage->depth, curImage->nChannels);
			cvResize(lpImage, pImageNew);
			cvReleaseImage(&curImage);
			curImage = pImageNew;
		}
	}

	if (curImage)
		cvReleaseImage(&curImage);
	m_lpImage = NULL;

	idx = Ren_MergeCandidates(faces, 1); // ren

	return idx;
}

float FaceDetection::VerifytFace(IplImage *lpImage, FdRect face)
{
	//m_lpImage = lpImage->GetRowData();
	int nOffset_Des = 0;
	int nSrcWidth=m_nImageWidth;
	nSrcWidth+=(nSrcWidth%4==0 ? 0 : 4-nSrcWidth%4);
    /*
	int i,j;
	BYTE *lpSrc = lpImage->GetRowData();
	BYTE pixel;
	for(i=0;i<m_nImageHeight;i++)
		for(j=0;j<m_nImageWidth;j++)
		{
			lpImage->GetPixel(j, i, &pixel);
			m_lpImage[nOffset_Des] = pixel;
			nOffset_Des ++;
		}
        */
    m_lpImage = (uchar*)lpImage->imageData;
	Cal_IntegralImage();

	if(face.width < face.height)
		m_nWinWidth = face.width;
	else m_nWinWidth = face.height;
	m_nWinHeight = m_nWinWidth;
	FdSize winsize = fdSize(m_nWinWidth, m_nWinHeight);
	
	m_CascadeClassifier->SetImagesForClassifier(winsize.width, winsize.height,
					size.width, size.height, sum_image, tilt_image, sqsum_image);
/*
	double dMinMean = 159.660751*0.6; //0.7
	double dMaxMean = 214.212585*1.2;
	double dMinNorm = 12.940265 *0.8;
	double dMaxNorm = 52.839954 *1.2;
*/
	/*
	double dMinMean = 100.179749*0.9; //0.7
	double dMaxMean = 236.962814*1.04;
	double dMinNorm = 4.784258 *0.9;
	double dMaxNorm = 61.584061 * 1.1;
*/
	bool bFace;
	float fWinMean, fWinNorm, fConfidence;
	//m_CascadeClassifier->CalMeanNorm(face.x, face.y, &fWinMean, &fWinNorm, winsize.width);
    m_CascadeClassifier->CalMeanNorm(face.x, face.y, &fWinMean, &fWinNorm);
	if((fWinMean>dMinMean)&&(fWinMean<dMaxMean)&&(fWinNorm>dMinNorm)&&(fWinNorm<dMaxNorm))
	{
		m_nAdaCompute_Num ++;
		bFace = m_CascadeClassifier->Fast_Evaluate( face.x, face.y, fWinMean, fWinNorm, &fConfidence);
		if(!bFace)
		{
			int mmm=0;
		}
	}
	else 
	{
		bFace = false;
		return 0;
	}

	return fConfidence;
}
////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the FaceDetection Class
///  written by Ren Haibing 
/// Acknowledge		: None
///
/// Function name	: DetectCandidates
/// Description	    	: detect the face candidate areas 
///
/// Argument		:  winsize -- windows size
/// Argument		:  factor -- the window size scale 
/// Argument		:  ystep -- the step on y coordinate
///
/// Return type		:  the face number
///
/// Create Time		: 2005-12-27  16:40
///
///
/// Side Effect		: 
///
///////////////////////////////////////////////////////////////////////////////////////////////
int FaceDetection::DetectCandidates(FdSize winsize, int ystep,  int xstep)
{
//	static long int count=0;
//	printf("Call DetectCandidates() %ld times\n", ++count);
/////////////////////////////////////////	
	CPoint_Ren pt;
	int nX_Start, nX_End, nY_Start, nY_End;
	float fWinMean;
	float Inv_fWinNorm;

	m_CascadeClassifier->SetImagesForClassifier(winsize.width, winsize.height,
					size.width, size.height, sum_image, tilt_image, sqsum_image);

	if (m_bWithSkinColor)
		m_FaceSkinImage.FaceSkin_SetWinSize(winsize.width);
	//m_FaceDepthImage->FaceDepth_SetWinSize(winsize.width);
	if(m_bROI_Set)
	{
		nX_Start = m_ImageROI.x;
		nY_Start = m_ImageROI.y;
		nX_End = m_ImageROI.x + m_ImageROI.width-1;
		if(nX_End>size.width)
			nX_End = size.width;

		nY_End = m_ImageROI.y + m_ImageROI.height -1;
		if(nY_End>size.height)
			nY_End = size.height;
	}
	else
	{
		nX_Start = 0;
		nY_Start = 0;
		nX_End = size.width;
		nY_End = size.height;
	}

    	int stopHeight = int((nY_End - nY_Start - winsize.height - ystep) / ystep);
	int stopWidth  = int((nX_End - nX_Start - winsize.width - xstep) / xstep);
	bool bResult = false;

	int yInt;
	int nNumber=0;
	float fConfidence;
/////////////////////////////////////////	
//printf("for times: %ld\n",stopHeight * stopWidth);

	for( yInt = 0; yInt <= stopHeight; yInt+=1)
	{
		pt.y = int(yInt * ystep + 0.5) + nY_Start;		
		
		for( int xInt = 0; xInt <= stopWidth; xInt+=1)
		{
			pt.x = int( xInt * xstep+0.5) + nX_Start;
			
			m_nTotalWinNum ++;
			
			if ((!m_bWithSkinColor) || (m_FaceSkinImage.FaceSkin_Verification(pt.x, pt.y)))
			{
		        if(m_CascadeClassifier->CalMeanNorm(pt.x, pt.y, &fWinMean, &Inv_fWinNorm))
				{
					m_nAdaCompute_Num ++;
					bResult = m_CascadeClassifier->Fast_Evaluate( pt.x, pt.y, fWinMean, Inv_fWinNorm, &fConfidence);
				}
			}

			if( bResult)		// face window
			{
				CComp *pData = new CComp;
				pData->rect.x = (int)(pt.x);
				pData->rect.y = (int)(pt.y);
				pData->rect.width = (int)(winsize.width);
				pData->rect.height = (int)(winsize.height);
				pData->confidence = fConfidence;
				pData->view = 3;//nViewNo;
				//m_Seq.Add(pData);
				m_Seq[m_nSeq_Candidate_Num] = pData;
				m_nSeq_Candidate_Num++;
				nNumber++;

				bResult = false;
			}
		}
	}
	//cout<<"winsize:"<<winsize.width<<"\t m_Seq.size():"<<m_nSeq_Candidate_Num<<"\t m_nTotalWinNum:"<<m_nTotalWinNum<<endl;
	return nNumber;
}
int FaceDetection::DetectCandidates(FdSize winsize, int ystep, int xstep, double dScale)
{
	//	static long int count=0;
	//	printf("Call DetectCandidates() %ld times\n", ++count);
	/////////////////////////////////////////	
	CPoint_Ren pt;
	int nX_Start, nX_End, nY_Start, nY_End;
	float fWinMean;
	float Inv_fWinNorm;

	m_CascadeClassifier->SetImagesForClassifier(winsize.width, winsize.height,
		size.width, size.height, sum_image, tilt_image, sqsum_image);

	if (m_bWithSkinColor)
		m_FaceSkinImage.FaceSkin_SetWinSize(winsize.width);
	//m_FaceDepthImage->FaceDepth_SetWinSize(winsize.width);
	if (m_bROI_Set)
	{
		nX_Start = m_ImageROI.x;
		nY_Start = m_ImageROI.y;
		nX_End = m_ImageROI.x + m_ImageROI.width - 1;
		if (nX_End>size.width)
			nX_End = size.width;

		nY_End = m_ImageROI.y + m_ImageROI.height - 1;
		if (nY_End>size.height)
			nY_End = size.height;
	}
	else
	{
		nX_Start = 0;
		nY_Start = 0;
		nX_End = size.width;
		nY_End = size.height;
	}

	int stopHeight = int((nY_End - nY_Start - winsize.height - ystep) / ystep);
	int stopWidth = int((nX_End - nX_Start - winsize.width - xstep) / xstep);
	bool bResult = false;

	int yInt;
	int nNumber = 0;
	float fConfidence;
	/////////////////////////////////////////	
	//printf("for times: %ld\n",stopHeight * stopWidth);

	for (yInt = 0; yInt <= stopHeight; yInt += 1)
	{
		pt.y = int(yInt * ystep + 0.5) + nY_Start;

		for (int xInt = 0; xInt <= stopWidth; xInt += 1)
		{
			pt.x = int(xInt * xstep + 0.5) + nX_Start;

		
			if (m_CascadeClassifier->CalMeanNorm(pt.x, pt.y, &fWinMean, &Inv_fWinNorm))
			{
				m_nAdaCompute_Num++;
				bResult = m_CascadeClassifier->Fast_Evaluate(pt.x, pt.y, fWinMean, Inv_fWinNorm, &fConfidence);
			}

			if (bResult)		// face window
			{
				CComp *pData = new CComp;
				pData->rect.x = (int)(pt.x*dScale);
				pData->rect.y = (int)(pt.y*dScale);
				pData->rect.width = (int)(winsize.width*dScale);
				pData->rect.height = (int)(winsize.height*dScale);
				pData->confidence = fConfidence;
				pData->view = 3;//nViewNo;
				//m_Seq.Add(pData);
				m_Seq[m_nSeq_Candidate_Num] = pData;
				m_nSeq_Candidate_Num++;
				nNumber++;

				bResult = false;
			}
		}
	}
	//cout<<"winsize:"<<winsize.width<<"\t m_Seq.size():"<<m_nSeq_Candidate_Num<<"\t m_nTotalWinNum:"<<m_nTotalWinNum<<endl;
	return nNumber;
}
/*
void CCompArray_Remove(CCompArray *Array)
{
	int i;
	CComp * temp;
	for(i=0;i<Array->GetSize();i++)
	{
		temp = Array->GetAt(i);
		if(temp != NULL)
			delete temp;
	}
}
*/
void CCompArray_Remove(CComp **Array,int nNumber)
{
	int i;
	CComp * temp;
	for(i=0;i<nNumber;i++)
	{
		temp = Array[i];
		if(temp != NULL)
			delete temp;
	}
}
#define  FACE_NUMBER 5000

int FaceDetection::Ren_MergeCandidates(FdAvgComp *faces, int min_neighbors)
{
	int i, j;
	int idx_tmp = 0;

	int idx_r[FACE_NUMBER];
	FdAvgComp comps[FACE_NUMBER];
	
	// merge\B5\C9 \C8ĺ\B8 \BF\B5\BF\AA\B5\E9\C0\BB \BC\B3\C1\A4
	int r_cnt = 0;
	//int nCandi = m_Seq.GetSize();
	int nCandi = m_nSeq_Candidate_Num;
//	cout << "Candidate Number " << nCandi <<endl;

	if(nCandi ==0) return 0;
	if(nCandi==1)  
	{
		faces[0].rect.x = m_Seq[0]->rect.x;
		faces[0].rect.y = m_Seq[0]->rect.y;
		faces[0].rect.width = m_Seq[0]->rect.width;
		faces[0].rect.height = m_Seq[0]->rect.height;
		faces[0].confidence =  m_Seq[0]->confidence;
		faces[0].neighbors = m_Seq[0]->neighbors;
		faces[0].view = m_Seq[0]->view;

		//CCompArray_Remove(&m_Seq);
		//m_Seq.RemoveAll();
		CCompArray_Remove(m_Seq, m_nSeq_Candidate_Num);
		return nCandi;
	}
	if(nCandi>=FACE_NUMBER)  //return 1;
		nCandi = FACE_NUMBER;
	/*
	for (int m = 0; m < nCandi; m++)
	{

		faces[m].rect.x = m_Seq[m]->rect.x;
		faces[m].rect.y = m_Seq[m]->rect.y;
		faces[m].rect.width = m_Seq[m]->rect.width;
		faces[m].rect.height = m_Seq[m]->rect.height;
		faces[m].confidence = m_Seq[m]->confidence;
		faces[m].neighbors = m_Seq[m]->neighbors;
		faces[m].view = m_Seq[m]->view;




	}
	CCompArray_Remove(m_Seq, m_nSeq_Candidate_Num);
	return nCandi;
	*/
//---------------------testing  begin--------------------------------------
/*
/// testing
	FILE *fTextfile = fopen("d:\\All_Candidiate.txt", "w");
	if(fTextfile == NULL) return false;	
	for(i=0;i<nCandi;i++)
	{
		faces[i].rect.x = m_Seq[i]->rect.x;
		faces[i].rect.y = m_Seq[i]->rect.y;
		faces[i].rect.width = m_Seq[i]->rect.width;
		faces[i].rect.height = m_Seq[i]->rect.height;
		faces[i].confidence =  m_Seq[i]->confidence;
		/// 2 write file header
		fprintf(fTextfile, "(%d  %d) (%d  %d)  %f\n",faces[i].rect.x,faces[i].rect.y,
			faces[i].rect.width, faces[i].rect.height, faces[i].confidence);
	}
	fclose(fTextfile);	
//	return nCandi;

*/
///-----------------------testing  end--------------------------------------
	//1. merge same size candidates (near to each other)
	memset(idx_r, -1, nCandi*sizeof(int));

	for (i = 0; i < nCandi; i++)
	{
		if(idx_r[i] == -1)
		{
			idx_r[i] = r_cnt;
			for(j=i+1;j<nCandi;j++)
			{
				if(idx_r[j] != -1) continue;
				if(	(m_Seq[i]->rect.width ==m_Seq[j]->rect.width) &&
					(is_equal(& m_Seq[i]->rect, & m_Seq[j]->rect))) 
					idx_r[j] = r_cnt;
			}
			r_cnt++;
		}
	}
	memset( comps, 0, nCandi*sizeof(FdAvgComp));						// comps\B4\C2 \B8\F0\B5\CE 0\C0\B8\B7\CE setting	
	
	int idx;
	
	for(i = 0; i < nCandi; i++)	
	{
		idx = idx_r[i];
		(comps[idx].neighbors)++;							
		comps[idx].rect.x += m_Seq[i]->rect.x;
		comps[idx].rect.y += m_Seq[i]->rect.y;
		comps[idx].rect.width = m_Seq[i]->rect.width;
		comps[idx].rect.height = m_Seq[i]->rect.height;
		comps[idx].confidence += m_Seq[i]->confidence;

/*		if(comps[idx].neighbors > 0)
		{
			comps[idx].neighbors++;							
			comps[idx].rect.x = min(comps[idx].rect.x, m_Seq[i]->rect.x);
			comps[idx].rect.y = min(comps[idx].rect.y, m_Seq[i]->rect.y);
			bottom_x = max(comps[idx].rect.x+comps[idx].rect.width, m_Seq[i]->rect.x+m_Seq[i]->rect.width);
			bottom_y = max(comps[idx].rect.y+comps[idx].rect.height, m_Seq[i]->rect.y+m_Seq[i]->rect.height);
			
			comps[idx].rect.width = bottom_x - comps[idx].rect.x;
			comps[idx].rect.height = bottom_y - comps[idx].rect.y;
			comps[idx].confidence = max(m_Seq[i]->confidence, comps[idx].confidence);
		}
		else
		{
			comps[idx].neighbors++;							
			comps[idx].rect.x =  m_Seq[i]->rect.x;
			comps[idx].rect.y = m_Seq[i]->rect.y;
			
			comps[idx].rect.width = m_Seq[i]->rect.width;
			comps[idx].rect.height = m_Seq[i]->rect.height;
			comps[idx].confidence = m_Seq[i]->confidence;
		}
*/
	}
	for(i = 0; i < r_cnt; i++)	
	{
		comps[i].rect.x = comps[i].rect.x/comps[i].neighbors;
		comps[i].rect.y = comps[i].rect.y/comps[i].neighbors;
	}
		
	//CCompArray_Remove(&m_Seq);
	//m_Seq.RemoveAll();
	CCompArray_Remove(m_Seq, m_nSeq_Candidate_Num);
//---------------------testing  begin--------------------------------------
/*
	/// testing
	FILE * fTextfile = fopen("d:\\Mid_Candidiate.txt", "w");
	if(fTextfile == NULL) return false;	
	for(i=0;i<r_cnt;i++)
	{
		faces[i].rect.x = comps[i].rect.x;
		faces[i].rect.y = comps[i].rect.y;
		faces[i].rect.width = comps[i].rect.width;
		faces[i].rect.height = comps[i].rect.height;
		faces[i].confidence =  comps[i].confidence;
		faces[i].neighbors = comps[i].neighbors;
		/// 2 write file header
		fprintf(fTextfile, "(%d  %d) (%d  %d)  %d  %f\n",faces[i].rect.x,faces[i].rect.y,
			faces[i].rect.width, faces[i].rect.height,faces[i].neighbors++, faces[i].confidence);
	}
	fclose(fTextfile);	
	
	/// testing
//	return r_cnt;
*/
///-----------------------testing  end--------------------------------------	

	memset(idx_r, -1, r_cnt*sizeof(int));
	int Total_Neighbor;
	// 2. merge the candidates which near to each other
	for (i = 0; i < r_cnt; i++)
	{
		if(idx_r[i] == -1)
		{
			for(j=i+1;j<r_cnt;j++)
			{
				if(idx_r[j] > -1) continue;
			//	if(is_equal(&(comps[i].rect), &(comps[j].rect)) )
				if(Overlap_Equal(&(comps[i].rect), &(comps[j].rect), 0.2)) //4) )					
				{
					Total_Neighbor = comps[i].neighbors + comps[j].neighbors;
					comps[i].neighbors = Total_Neighbor;
					comps[j].neighbors = Total_Neighbor;
					
					if(comps[i].confidence<comps[j].confidence)						
					{
						comps[i] = comps[j];
						idx_r[i] = j;
					}
					else
					{
						comps[j] = comps[i];
						idx_r[j] = i;
					}

				}
			}
		}
	}
	/// 3. merge one inside another one
	for (i = 0; i < r_cnt; i++)
	{
		if(idx_r[i] == -1)
		{
			for(j=i+1;j<r_cnt;j++)
			{
				if(idx_r[j] > -1) continue;
				if(is_inside(&(comps[i].rect), &(comps[j].rect)) ||
					is_inside(&(comps[j].rect), &(comps[i].rect)) )
				{
					Total_Neighbor = comps[i].neighbors + comps[j].neighbors;
					comps[i].neighbors = Total_Neighbor;
					comps[j].neighbors = Total_Neighbor;
					
					if(comps[i].confidence<comps[j].confidence)						
					{
						comps[i] = comps[j];
						idx_r[i] = j;
					}
					else
					{
						comps[j] = comps[i];
						idx_r[j] = i;
					}

				}
			}
		}//if(idx_r[i] == -1)
	}//for (i = 0; i < r_cnt; i++)
					 
//	fTextfile = fopen("d:\\Last_Candidiate.txt", "w");
//	if(fTextfile == NULL) return false;	
	i =0;
	int number =0;
	while(i<r_cnt)
	{
		if((idx_r[i]==-1) && (comps[i].neighbors>2))// (comps[i].neighbors>2))
		{
			faces[number].rect.x = comps[i].rect.x;
			faces[number].rect.y = comps[i].rect.y;
			faces[number].rect.width = comps[i].rect.width;
			faces[number].rect.height = comps[i].rect.height;
			faces[number].confidence =  comps[i].confidence;
			faces[number].neighbors = comps[i].neighbors;
			/// 2 write file header
//			fprintf(fTextfile, "(%d  %d) (%d  %d)  %d  %f\n",faces[number].rect.x,faces[number].rect.y,
//				faces[number].rect.width, faces[number].rect.height,faces[number].neighbors++, faces[number].confidence);
			number++;
		}
		i++;
	}
	if((number ==0) && (r_cnt>0))
	{
		i=0;
		while(i<r_cnt)
		{
			if(idx_r[i]==-1)
			{
				faces[number].rect.x = comps[i].rect.x;
				faces[number].rect.y = comps[i].rect.y;
				faces[number].rect.width = comps[i].rect.width;
				faces[number].rect.height = comps[i].rect.height;
				faces[number].confidence =  comps[i].confidence;
				faces[number].neighbors = comps[i].neighbors;
				/// 2 write file header
//				fprintf(fTextfile, "(%d  %d) (%d  %d)  %d  %f\n",faces[number].rect.x,faces[number].rect.y,
//					faces[number].rect.width, faces[number].rect.height,faces[number].neighbors++, faces[number].confidence);
				number++;
			}
			i++;
		}
	}
//	fclose(fTextfile);	
	return number;
	
//-----------------------testing end---------------------------------------
	
}


////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the FaceDetection Class
///  written by Ren Haibing 
/// Acknowledge		: None
///
/// Function name	: LoadDefaultFrontalDetector
/// Description	    : load the default FrontalDetector from the model.h 
///
/// Argument		: 
/// Argument		: 
///
/// Return type		: true: load model successfully
///
/// Create Time		: 2015-3-3  16:59
///
///
/// Side Effect		: 
///
///////////////////////////////////////////////////////////////////////////////////////////////
bool FaceDetection::LoadDefaultFrontalDetector(const unsigned short* nFaceDetector_Int, const double *nFaceDetector_double)
{

	int nStageNum, i;
	bool bOK = true;
	// Get Stage Num
	int nIntNo = 0;
	int nFloatNo = 0;
	int nRead_Int, nRead_Float;

	nStageNum = nFaceDetector_Int[nIntNo];
	nIntNo++;
	if(nStageNum <0) return false;

	m_CascadeClassifier->m_nStageNum = nStageNum;	
	for(i=0;(i<nStageNum) && bOK;i++)
	{
		bOK = m_CascadeClassifier->m_StageClassifier[i].LoadStageClassifier_FromArray(
			&(nFaceDetector_Int[nIntNo]), &(nFaceDetector_double[nFloatNo]), &nRead_Int, &nRead_Float);

		nIntNo += nRead_Int;
		nFloatNo += nRead_Float;
	}

	return bOK;
}

